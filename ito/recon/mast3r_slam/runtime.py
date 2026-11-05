import hashlib
import importlib
import importlib.util
import json
import os
import platform
import sys
from functools import wraps

from ito.recon import models

# The bundle ito.recon.mast3r_slam.build makes: weights, MASt3R-SLAM sources, CUDA kernels.
MODELS = models() / "mast3r_slam"
# 6: MASt3R-SLAM in full (retrieval, asmk, FactorGraph in fixed buffers); no tracker prior.
BUNDLE_VERSION = 6
# asmk.hamming: asmk's C module, loaded under the name its package imports it by.
EXTENSIONS = ("lietorch_backends", "mast3r_slam_backends", "curope", "asmk.hamming")
MODEL_FILES = {
    "model.safetensors": "0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb",
    "config.json": "718eb93dc4f9e4332b60cc0041af962d712cbd346d7770ce35c5b22cff68eae4",
}
RETRIEVAL = "retrieval/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
CODEBOOK = "retrieval/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"
ASMK = "slam/thirdparty/mast3r/asmk"


def abi():
    return [sys.implementation.cache_tag, sys.platform, platform.machine()]


def verify(models=MODELS):
    """Everything MASt3R-SLAM needs is in MODELS, whole: the manifest ito.build wrote."""
    for name, expected in MODEL_FILES.items():
        path = models / name
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
        manifest = json.loads((models / "manifest.json").read_text(encoding="utf-8"))
        if manifest["version"] != BUNDLE_VERSION or manifest["abi"] != abi():
            raise RuntimeError(
                f"MASt3R bundle in {models} is not bundle {BUNDLE_VERSION} for {abi()}; "
                "rebuild it with ito.build"
            )
        required = {
            *MODEL_FILES,
            "LICENSE",
            "NOTICE",
            "CHECKPOINTS_NOTICE",
            "ATTRIBUTION.txt",
            RETRIEVAL,
            CODEBOOK,
            "slam/config/base.yaml",
            "slam/mast3r_slam/global_opt.py",
            "slam/mast3r_slam/retrieval_database.py",
            "slam/mast3r_slam/tracker.py",
            f"{ASMK}/asmk/kernel.py",
            "lie/lietorch/__init__.py",
        }
        if not required <= manifest["files"].keys():
            raise ValueError("missing assets")
        for name, size in manifest["files"].items():
            path = models / name
            if not path.resolve().is_relative_to(models.resolve()) or path.stat().st_size != size:
                raise ValueError("incomplete assets")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError(f"MASt3R files missing or damaged in {models}") from exc
    try:
        if not all(manifest["native"][name] in manifest["files"] for name in EXTENSIONS):
            raise ValueError("missing kernels")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"MASt3R native extensions missing from {models / 'native'}") from exc
    return manifest


def torch_for(manifest):
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError("MASt3R CUDA support unavailable") from exc
    if str(torch.__version__) != manifest["torch"]:
        raise RuntimeError("MASt3R CUDA support incompatible with this app")
    return torch


def load(models, manifest):
    """Load the bundle's native extensions under the names they are imported by (asmk's
    hamming among them) and put its sources on the path. Needs torch, not a GPU."""
    try:
        for name in EXTENSIONS:
            spec = importlib.util.spec_from_file_location(name, models / manifest["native"][name])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[name] = module
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        raise RuntimeError(
            f"MASt3R native extensions could not load from {models / 'native'}"
        ) from exc
    slam = models / "slam"
    paths = [slam, models / "lie", slam / "thirdparty/mast3r", models / ASMK]
    sys.path[:0] = [str(path) for path in paths if str(path) not in sys.path]


def prepare(report):
    """Load the bundle's native extensions and put its sources on the path. Returns the
    bundle's MASt3R-SLAM directory, the weights directory and the retrieval checkpoint."""
    manifest = verify()
    torch = torch_for(manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("MASt3R needs an NVIDIA CUDA device and driver")
    report("Loading MASt3R")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    load(MODELS, manifest)
    return MODELS / "slam", MODELS, MODELS / RETRIEVAL


# Everything the SLAM worker imports from the bundle and its dependencies: MASt3R-SLAM itself,
# MASt3R's retrieval (which needs faiss) and asmk.
IMPORTS = (
    "faiss",
    "lietorch",
    "asmk.asmk_method",
    "mast3r.retrieval.processor",
    "mast3r_slam.retrieval_database",
    "mast3r_slam.global_opt",
    "mast3r_slam.tracker",
    "mast3r_slam.mast3r_utils",
)


def check_bundle(models=MODELS):
    """The release's check that the default SLAM path will load, without a GPU: every bundle
    file whole (verify), the native extensions, and every import of IMPORTS."""
    manifest = verify(models)
    torch_for(manifest)
    load(models, manifest)
    for name in IMPORTS:
        importlib.import_module(name)


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
