"""Read shipped SLAM assets in the reconstruction worker, never on the display thread."""

import hashlib
import importlib.util
import json
import logging
import os
import platform
import sys
import sysconfig
from functools import wraps
from pathlib import Path


def models_directory():
    # The Windows release: models\ sits beside ito.exe and the runtime\ it starts.
    release = Path(sys.base_prefix).parent
    if (release / "ito.exe").is_file() and (release / "runtime").is_dir():
        return release / "models"
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file():
        return checkout / "models"
    # Console entry points live in bin/ on Unix and Scripts/ on Windows.
    return Path(sysconfig.get_path("scripts")) / "models"


MODELS = models_directory()
BUNDLE_VERSION = 5  # Bump when source pins, layout or kernel adaptations change.
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
    # Instantiate locally below; also disable upstream hub lookups and telemetry.
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
    try:
        import torch
        from mast3r.model import AsymmetricMASt3R
        from safetensors.torch import load_file

        # No from_pretrained/hub path, pickle checkpoint, cache or runtime writes.
        model = AsymmetricMASt3R(**json.loads((weights / "config.json").read_text()))
        result = model.load_state_dict(load_file(weights / "model.safetensors"), strict=False)
        # Safetensors stores each shared tensor once; DPT's layer_rn list aliases layerN_rn.
        missing = [key for key in result.missing_keys if ".dpt.scratch.layer_rn." not in key]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"weights do not match MASt3R: {missing + result.unexpected_keys}")
        model.eval()
        network = NetworkPrecision(model, weights, precision_report)
        if precision == "half":
            try:
                network.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                model.to(device="cuda", dtype=network.dtype)
            except (RuntimeError, TypeError, ValueError) as exc:
                network.fallback(exc)
        else:
            model.to("cuda")
        if network.dtype != torch.float32:
            network.native_positions()
            for name in ("_encode_image", "_decoder", "_downstream_head"):
                network.wrap(name)
        network.report()
        return model
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"MASt3R model could not load from {weights}") from exc


class NetworkPrecision:
    """Low precision ends at each network call; tracking and native kernels receive fp32."""

    def __init__(self, model, weights, report):
        import torch

        self.model, self.weights, self.notify = model, weights, report
        self.dtype = torch.float32
        self.reason = ""

    def report(self):
        import torch

        active = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}[self.dtype]
        if self.notify:
            self.notify(active, self.reason)

    def fallback(self, error):
        import torch
        from safetensors.torch import load_file

        self.reason = str(error).splitlines()[0]
        logging.getLogger(__name__).warning(
            "MASt3R Half failed; falling back to Full: %s", self.reason
        )
        # Restore original fp32 weights, rather than promoting rounded Half weights.
        self.model.to(device="cpu", dtype=torch.float32)
        self.model.load_state_dict(load_file(self.weights / "model.safetensors"), strict=False)
        torch.cuda.empty_cache()
        self.model.to("cuda")
        self.dtype = torch.float32
        self.report()

    def native_positions(self):
        import torch

        # MASt3R deep-copies the second decoder, including its separate RoPE module.
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

    def wrap(self, name):
        import torch

        original = getattr(self.model, name)

        @wraps(original)
        def inference(*args, **kwargs):
            if self.dtype == torch.float32:
                return original(*args, **kwargs)

            def invoke():
                with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
                    result = original(*args, **kwargs)
                result = tensors(result, lambda t: t.float() if t.is_floating_point() else t)
                if self.dtype != torch.float32:
                    tensors(result, finite)
                return result

            try:
                return invoke()
            except (RuntimeError, ValueError, TypeError) as exc:
                if self.dtype == torch.float32:
                    raise
                self.fallback(exc)
                return invoke()

        def finite(tensor):
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"{name} produced non-finite network output")
            return tensor

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
