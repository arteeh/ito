"""Load precompiled gsplat kernels; never invoke a compiler in the pilot app."""

import importlib.util
import json
import sys
import types

from .mast3r_runtime import MODELS, abi

VERSION = "1.5.3"


def prepare():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    manifest_path = MODELS / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("abi") != abi() or manifest.get("torch") != str(torch.__version__):
            raise RuntimeError("gsplat bundle ABI mismatch")
        relative = manifest.get("native", {}).get("gsplat_cuda")
        if not relative or manifest.get("revisions", {}).get("gsplat") != VERSION:
            raise RuntimeError("precompiled gsplat missing from models bundle")
        binary = MODELS / relative
        if not binary.resolve().is_relative_to(
            MODELS.resolve()
        ) or binary.stat().st_size != manifest["files"].get(relative):
            raise RuntimeError("gsplat kernel file damaged")
        spec = importlib.util.spec_from_file_location("gsplat_cuda", binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.path.insert(0, str(MODELS / "gsplat"))
        # Upstream's backend falls back to JIT when importing csrc fails. Install the
        # already loaded extension before importing gsplat, including its wrappers.
        backend = types.ModuleType("gsplat.cuda._backend")
        backend._C = module
        sys.modules[backend.__name__] = backend
    else:
        spec = importlib.util.find_spec("gsplat")
        if spec is None:
            raise RuntimeError("gsplat not installed")
        from importlib.machinery import EXTENSION_SUFFIXES
        from pathlib import Path

        if not any((Path(spec.origin).parent / ("csrc" + s)).is_file() for s in EXTENSION_SUFFIXES):
            raise RuntimeError("precompiled gsplat not installed")
    import gsplat

    if gsplat.__version__ != VERSION:
        raise RuntimeError(f"gsplat {VERSION} required")
    return gsplat.rasterization
