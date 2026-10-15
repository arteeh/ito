"""Ceres experiment bootstrap: reuse native kernels and bound Python warm-up workers."""

import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
os.environ["MAX_JOBS"] = "1"
os.environ["NVCC_THREADS"] = "1"


class Binary(importlib.abc.Loader):
    def create_module(self, spec):
        from torch.version import cuda

        version = f"py{sys.version_info.major}{sys.version_info.minor}_cu{cuda.replace('.', '')}"
        path = Path.home() / ".cache/torch_extensions" / version / "gsplat_cuda/gsplat_cuda.so"
        if not path.is_file():
            raise RuntimeError("Existing gsplat binary required; native builds prohibited")
        native = importlib.util.spec_from_file_location("gsplat_cuda", path)
        module = importlib.util.module_from_spec(native)
        native.loader.exec_module(module)
        return module

    def exec_module(self, module):
        pass


class Prebuilt(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "gsplat.csrc":
            return importlib.util.spec_from_loader(fullname, Binary())


sys.meta_path.insert(0, Prebuilt())
# Release clean cache only for experiment files this process actually opened.
close_snapshot = np.lib.npyio.NpzFile.close


def release_snapshot(self):
    stream = getattr(self, "fid", None)
    if stream is not None:
        try:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (OSError, AttributeError, ValueError):
            pass
    close_snapshot(self)


np.lib.npyio.NpzFile.close = release_snapshot
