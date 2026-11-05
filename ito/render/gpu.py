import ctypes as ct
import threading


class _Memory(ct.Structure):
    _fields_ = [(name, ct.c_ulonglong) for name in ("total", "free", "used")]


class _Utilization(ct.Structure):
    _fields_ = [(name, ct.c_uint) for name in ("gpu", "memory")]


def bus_id(xp, device):
    """The CUDA device's PCI bus id from CuPy (Torch has none), or None if this CuPy cannot
    say: then GPU() takes CUDA's index, right on a one-GPU machine."""
    try:
        return xp.cuda.runtime.deviceGetPCIBusId(device)
    except Exception:
        return None


class GPU:
    def __init__(self, index=0, bus_id=None):
        """bus_id: the card's PCI bus id ("0000:01:00.0"), for a CUDA device: CUDA numbers only the
        cards it sees (CUDA_VISIBLE_DEVICES, fastest first), NVML's index counts all of them."""
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
            if bus_id is None:
                self._call(
                    library.nvmlDeviceGetHandleByIndex_v2, ct.c_uint(index), ct.byref(self.handle)
                )
            else:
                bus_id = bus_id if isinstance(bus_id, bytes) else bus_id.encode()
                self._call(
                    library.nvmlDeviceGetHandleByPciBusId_v2,
                    ct.c_char_p(bus_id),
                    ct.byref(self.handle),
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
                return
        if self.library is not None:
            self.library.nvmlShutdown()
            self.library = None
