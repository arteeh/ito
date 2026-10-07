"""Load only shipped SLAM assets; fetching and compilation belong to ito.build."""

import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1] / "_slam"
BUILD_COMMAND = "uv run --python 3.12 --extra slam python -m ito.build"
BUNDLE_VERSION = 1
EXTENSIONS = ("lietorch_backends", "mast3r_slam_backends", "curope")


def abi():
    return [sys.implementation.cache_tag, sys.platform, platform.machine()]


def bundle_error(reason):
    return RuntimeError(
        f"MASt3R-SLAM bundle missing or incomplete ({reason}). Build: {BUILD_COMMAND}"
    )


def prepare(report):
    try:
        manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
        if manifest["version"] != BUNDLE_VERSION or manifest["abi"] != abi():
            raise ValueError("incompatible bundle")
        required = {
            "weights/model.safetensors",
            "weights/config.json",
            "weights/LICENSE",
            "weights/NOTICE",
            "weights/CHECKPOINTS_NOTICE",
            "weights/ATTRIBUTION.txt",
            "slam/config/base.yaml",
            "slam/mast3r_slam/tracker.py",
            "lie/lietorch/__init__.py",
        }
        required.update(manifest["native"][name] for name in EXTENSIONS)
        if not required <= manifest["files"].keys():
            raise ValueError("missing assets")
        for name, size in manifest["files"].items():
            path = BUNDLE / name
            if not path.resolve().is_relative_to(BUNDLE) or path.stat().st_size != size:
                raise ValueError(f"incomplete {name}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise bundle_error("rebuild the shipped assets and CUDA kernels") from exc

    try:
        import torch
    except ImportError as exc:
        raise bundle_error("CUDA dependencies unavailable") from exc
    if str(torch.__version__) != manifest["torch"]:
        raise bundle_error("PyTorch version differs from build")
    if not torch.cuda.is_available():
        raise RuntimeError("MASt3R-SLAM needs an NVIDIA CUDA device and driver")

    report("Loading bundled MASt3R-SLAM")
    # The model is instantiated locally below. Also disable upstream hub lookups.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        for name in EXTENSIONS:
            spec = importlib.util.spec_from_file_location(name, BUNDLE / manifest["native"][name])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[name] = module
    except (ImportError, OSError, RuntimeError) as exc:
        raise bundle_error(f"cannot load CUDA kernels: {exc}") from exc
    slam = BUNDLE / "slam"
    sys.path[:0] = [str(slam), str(BUNDLE / "lie"), str(slam / "thirdparty/mast3r")]
    return slam, BUNDLE / "weights"


def load_model(weights, report):
    from mast3r.model import AsymmetricMASt3R
    from safetensors.torch import load_file

    report("Loading bundled MASt3R weights onto CUDA")
    try:
        # No from_pretrained/hub path, pickle checkpoint, cache or runtime writes.
        model = AsymmetricMASt3R(**json.loads((weights / "config.json").read_text()))
        model.load_state_dict(load_file(weights / "model.safetensors"), strict=True)
        return model.eval().to("cuda")
    except (OSError, ValueError, RuntimeError) as exc:
        raise bundle_error(f"cannot load weights: {exc}") from exc
