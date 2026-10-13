"""One persistent NVML connection; sampling never runs on the display thread."""

import ctypes as ct
import threading

MAX_SLOT_COUNT = (1 << 31) - 1  # ImGui's signed count; slot IDs on the wire are uint32.


class _Memory(ct.Structure):
    _fields_ = [(name, ct.c_ulonglong) for name in ("total", "free", "used")]


class _Utilization(ct.Structure):
    _fields_ = [(name, ct.c_uint) for name in ("gpu", "memory")]


class GPU:
    def __init__(self, index=0):
        self.library = None
        self.handle = ct.c_void_p()
        self.free = self.total = 0
        self.busy = None
        self.stopped = threading.Event()
        self.thread = None
        library = None
        initialized = False
        try:
            import os

            library = ct.CDLL("nvml.dll" if os.name == "nt" else "libnvidia-ml.so.1")
            self._call(library.nvmlInit_v2)
            initialized = True
            self._call(
                library.nvmlDeviceGetHandleByIndex_v2, ct.c_uint(index), ct.byref(self.handle)
            )
            self.library = library
            self.sample()
        except (OSError, AttributeError, RuntimeError):
            if initialized:
                library.nvmlShutdown()
            self.library = None

    @staticmethod
    def _call(function, *args):
        if function(*args):
            raise RuntimeError("NVML query unavailable")

    def sample(self):
        memory, utilization = _Memory(), _Utilization()
        self._call(self.library.nvmlDeviceGetMemoryInfo, self.handle, ct.byref(memory))
        self.free, self.total = memory.free, memory.total
        try:
            self._call(
                self.library.nvmlDeviceGetUtilizationRates, self.handle, ct.byref(utilization)
            )
            self.busy = float(utilization.gpu)
        except RuntimeError:
            self.busy = None

    def used(self):
        """Device-wide (used, total) bytes as last sampled; None without NVML."""
        if self.library is None or not self.total:
            return None
        return self.total - self.free, self.total

    def start(self):
        if self.library is not None and self.thread is None:
            self.thread = threading.Thread(target=self._sample, name="ito-gpu", daemon=True)
            self.thread.start()

    def _sample(self):
        while not self.stopped.wait(0.5):
            try:
                self.sample()
            except RuntimeError:
                self.busy = None
                break

    def close(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
            if self.thread.is_alive():
                return  # Never unload a library while its query is still running.
        if self.library is not None:
            self.library.nvmlShutdown()
            self.library = None
