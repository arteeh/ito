"""Persistent driver memory query; avoid restarting NVML during display frames.

The stable NVML v1 memory structure includes all device use. WDDM PID counters
remain separate because Windows does not attribute Linux CUDA to those PIDs.
"""

import ctypes as c
from contextlib import contextmanager


class Memory(c.Structure):
    _fields_ = [(name, c.c_ulonglong) for name in ("total", "free", "used")]


def check(status):
    if status:
        raise RuntimeError(f"NVML memory monitor failed with status {status}")


@contextmanager
def device_memory():
    library = c.CDLL("C:/Windows/System32/nvml.dll")
    library.nvmlDeviceGetHandleByIndex_v2.argtypes = (c.c_uint, c.POINTER(c.c_void_p))
    library.nvmlDeviceGetMemoryInfo.argtypes = (c.c_void_p, c.POINTER(Memory))
    check(library.nvmlInit_v2())
    try:
        handle = c.c_void_p()
        check(library.nvmlDeviceGetHandleByIndex_v2(0, c.byref(handle)))

        def used_mib():
            sample = Memory()
            check(library.nvmlDeviceGetMemoryInfo(handle, c.byref(sample)))
            return (sample.used + 1024**2 - 1) // 1024**2

        yield used_mib
    finally:
        check(library.nvmlShutdown())
