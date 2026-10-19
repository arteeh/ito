import hashlib
import importlib.util
import json
import os
import platform
import sys
import sysconfig
from functools import wraps
from pathlib import Path


def models_directory():
    release = Path(sys.base_prefix).parent
    if (release / "ito.exe").is_file() and (release / "runtime").is_dir():
        return release / "models"
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file():
        return checkout / "models"
    return Path(sysconfig.get_path("scripts")) / "models"


MODELS = models_directory()
BUNDLE_VERSION = 5
EXTENSIONS = ("lietorch_backends", "mast3r_slam_backends", "curope")
MODEL_FILES = {
    "model.safetensors": "0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb",
    "config.json": "718eb93dc4f9e4332b60cc0041af962d712cbd346d7770ce35c5b22cff68eae4",
}


def abi():
    return [sys.implementation.cache_tag, sys.platform, platform.machine()]


def prepare(report):
    for name, expected in MODEL_FILES.items():
        path = MODELS / name
        try:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        except FileNotFoundError as exc:
            raise RuntimeError(f"MASt3R model missing from {path}") from exc
        except OSError as exc:
            raise RuntimeError(f"MASt3R model unreadable at {path}") from exc
        if digest != expected:
            raise RuntimeError(f"MASt3R model checksum failed at {path}")

    try:
        manifest = json.loads((MODELS / "manifest.json").read_text(encoding="utf-8"))
        if manifest["version"] != BUNDLE_VERSION or manifest["abi"] != abi():
            raise RuntimeError("MASt3R bundle needs rebuilding for current-stream CUDA kernels")
        required = {
            *MODEL_FILES,
            "LICENSE",
            "NOTICE",
            "CHECKPOINTS_NOTICE",
            "ATTRIBUTION.txt",
            "slam/config/base.yaml",
            "slam/mast3r_slam/tracker.py",
            "lie/lietorch/__init__.py",
        }
        if not required <= manifest["files"].keys():
            raise ValueError("missing assets")
        for name, size in manifest["files"].items():
            path = MODELS / name
            if not path.resolve().is_relative_to(MODELS.resolve()) or path.stat().st_size != size:
                raise ValueError("incomplete assets")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError(f"MASt3R files missing or damaged in {MODELS}") from exc
    try:
        if not all(manifest["native"][name] in manifest["files"] for name in EXTENSIONS):
            raise ValueError("missing kernels")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"MASt3R CUDA extensions missing from {MODELS / 'native'}") from exc

    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError("MASt3R CUDA support unavailable") from exc
    if str(torch.__version__) != manifest["torch"]:
        raise RuntimeError("MASt3R CUDA support incompatible with this app")
    if not torch.cuda.is_available():
        raise RuntimeError("MASt3R needs an NVIDIA CUDA device and driver")

    report("Loading MASt3R")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        for name in EXTENSIONS:
            spec = importlib.util.spec_from_file_location(name, MODELS / manifest["native"][name])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[name] = module
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        raise RuntimeError(
            f"MASt3R CUDA extensions could not load from {MODELS / 'native'}"
        ) from exc
    slam = MODELS / "slam"
    sys.path[:0] = [str(slam), str(MODELS / "lie"), str(slam / "thirdparty/mast3r")]
    return slam, MODELS


def load_model(weights, report, *, precision="full", precision_report=None):
    report("Loading MASt3R")
    import torch

    try:
        from mast3r.model import AsymmetricMASt3R
        from safetensors.torch import load_file

        model = AsymmetricMASt3R(**json.loads((weights / "config.json").read_text()))
        result = model.load_state_dict(load_file(weights / "model.safetensors"), strict=False)
        # Safetensors stores each shared tensor once; DPT's layer_rn list aliases layerN_rn.
        missing = [key for key in result.missing_keys if ".dpt.scratch.layer_rn." not in key]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"weights do not match MASt3R: {missing + result.unexpected_keys}")
        adapter = os.environ.get("ITO_MAST3R_ADAPTER")
        if adapter:
            merge_adapter(model, adapter, report)
        model.eval()
        network = NetworkPrecision(model, precision_report)
        if precision == "half":
            network.dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported(including_emulation=False)
                else torch.float16
            )
        model.to(device="cuda", dtype=network.dtype)
        if network.dtype != torch.float32:
            network.fp32_norm_weights()
            network.native_positions()
            network.native_decoder()
            for name in ("_encode_image", "_decoder", "_downstream_head"):
                network.wrap(name)
        if network.dtype == torch.float16:
            network.invalid_count = torch.zeros((), dtype=torch.int64, device="cuda")
        model.network_precision = network
        network.report()
        return model
    except torch.cuda.OutOfMemoryError:
        raise
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"MASt3R model could not load from {weights}") from exc


def merge_adapter(model, path, report):
    """ITO_MAST3R_ADAPTER=file.safetensors (research): a LoRA adapter for MASt3R's decoders
    (GrandCougar/tools/lora_mast3r.py: <module>.down.weight / .up.weight, scale alpha/rank in the
    file's metadata, default 1) merged into the weights: W += scale * up @ down. No runtime cost."""
    import torch
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        scale = float((f.metadata() or {}).get("scale", 1.0))
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    modules = dict(model.named_modules())
    merged = 0
    for key, down in tensors.items():
        if not key.endswith(".down.weight"):
            continue
        name = key[: -len(".down.weight")]
        up = tensors[name + ".up.weight"]
        linear = modules[name.removesuffix(".base")] if name not in modules else modules[name]
        with torch.no_grad():
            linear.weight += (scale * up.float() @ down.float()).to(linear.weight.dtype)
        merged += 1
    report(f"MASt3R adapter {Path(path).name}: {merged} layers merged")


class NetworkPrecision:
    def __init__(self, model, report):
        import torch

        self.model, self.notify = model, report
        self.dtype = torch.float32
        self.invalid_count = None

    def report(self):
        import torch

        active = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}[self.dtype]
        if self.notify:
            self.notify(active)

    def statistics(self):
        return 0 if self.invalid_count is None else int(self.invalid_count.item())

    def sanitize(self, result):
        import torch

        valid = torch.isfinite(result["pts3d"]).all(dim=-1)
        valid &= torch.isfinite(result["desc"]).all(dim=-1)
        valid &= torch.isfinite(result["conf"]) & torch.isfinite(result["desc_conf"])
        self.invalid_count.add_((~valid).sum())
        return {
            key: torch.where(valid[..., None] if value.ndim == valid.ndim + 1 else valid, value, 0)
            for key, value in result.items()
        }

    def fp32_norm_weights(self):
        import torch

        for module in self.model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.float()

    def native_positions(self):
        import torch

        ropes = {
            module.rope
            for module in self.model.modules()
            if getattr(module, "rope", None) is not None
        }
        for rope in ropes:
            original = rope.forward

            def positions(tokens, pixels, *, original=original):
                # cuRoPE has no bf16 dispatch. Native CUDA math stays fp32 in either mode.
                if tokens.dtype == torch.float32:
                    return original(tokens, pixels)
                with torch.autocast("cuda", enabled=False):
                    return original(tokens.float(), pixels).to(tokens.dtype)

            rope.forward = positions

    @staticmethod
    def native_decoder():
        import torch
        from mast3r_slam import mast3r_utils

        @torch.inference_mode()
        def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2):
            dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
            return (
                model._downstream_head(1, dec1, shape1),
                model._downstream_head(2, dec2, shape2),
            )

        mast3r_utils.decoder = decoder

    def wrap(self, name):
        import torch

        original = getattr(self.model, name)

        @wraps(original)
        def inference(*args, **kwargs):
            with torch.autocast("cuda", dtype=self.dtype):
                result = original(*args, **kwargs)
            if name == "_downstream_head":
                result = tensors(result, lambda t: t.float() if t.is_floating_point() else t)
            if name == "_downstream_head" and self.dtype == torch.float16:
                result = self.sanitize(result)
            return result

        setattr(self.model, name, inference)


def tensors(value, function):
    import torch

    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return {key: tensors(item, function) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tensors(item, function) for item in value)
    return value
