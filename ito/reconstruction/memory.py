"""VRAM: no limits. What Ito takes is what its settings ask for (precision, Input resolution,
Map size, Max splats); nothing is given up to fit the card. The worker samples free VRAM and the
GPU's load for the stats, and on Windows whether its VRAM overflows into system RAM: there the
driver does not fail an allocation past the card but places it in system RAM ("Shared GPU
memory" in Task Manager), and everything slows. Linux fails it instead (slam.py's out of VRAM)."""

import ctypes as ct
import os

from ito import clock
from ito.render.gpu import GPU, bus_id

INTERVAL = 1.0
# This much more of Ito's GPU memory in system RAM than at its lowest: spilling.
SPILL_BYTES = 128 << 20


class _Value(ct.Structure):
    _fields_ = [("status", ct.c_uint32), ("value", ct.c_int64)]  # PDH_FMT_COUNTERVALUE, large


class _Item(ct.Structure):
    _fields_ = [("name", ct.c_wchar_p), ("value", _Value)]  # PDH_FMT_COUNTERVALUE_ITEM_W


PDH_FMT_LARGE = 0x400
PDH_MORE_DATA = 0x800007D2


class Shared:
    """Windows: how much of this process's GPU memory is in system RAM: the "GPU Process
    Memory" performance counter's Shared Usage, what Task Manager shows as "Shared GPU memory".
    It counts CUDA's spill (12 GiB card: 76 MiB at 10 GiB held, 4 GiB at 15). DXGI's per-process
    non-local usage does not (it stayed at 76 MiB). Only the instances on CUDA's card (by LUID)
    when that is known. query None where it cannot be read."""

    def __init__(self, luid=None):
        self.query = None
        if os.name != "nt":
            return
        self.luid = None if luid is None else f"luid_0x{luid[1]:08x}_0x{luid[0]:08x}"
        try:
            self.pdh = ct.WinDLL("pdh")
            query, counter = ct.c_void_p(), ct.c_void_p()
            if self.pdh.PdhOpenQueryW(None, None, ct.byref(query)):
                return
            self.query = query
            path = f"\\GPU Process Memory(pid_{os.getpid()}_*)\\Shared Usage"
            if self.pdh.PdhAddEnglishCounterW(query, ct.c_wchar_p(path), None, ct.byref(counter)):
                self.close()
                return
            self.counter = counter
            self.used()
        except OSError:
            self.close()

    def used(self):
        """Bytes; OSError when the counter cannot be read."""
        pdh = self.pdh
        if pdh.PdhCollectQueryData(self.query):
            raise OSError("PdhCollectQueryData failed")
        size, count = ct.c_uint32(0), ct.c_uint32(0)
        status = pdh.PdhGetFormattedCounterArrayW(
            self.counter, PDH_FMT_LARGE, ct.byref(size), ct.byref(count), None
        ) & 0xFFFFFFFF
        if status != PDH_MORE_DATA:
            raise OSError(f"GPU Process Memory counter: {status:#x}")
        buffer = ct.create_string_buffer(size.value)
        status = pdh.PdhGetFormattedCounterArrayW(
            self.counter, PDH_FMT_LARGE, ct.byref(size), ct.byref(count), buffer
        )
        if status:
            raise OSError(f"GPU Process Memory counter: {status & 0xFFFFFFFF:#x}")
        items = ct.cast(buffer, ct.POINTER(_Item))
        values = [items[i] for i in range(count.value)]
        mine = [i for i in values if self.luid is not None and self.luid in i.name.lower()]
        return sum(i.value.value for i in (mine or values) if i.value.status in (0, 1))

    def close(self):
        query, self.query = self.query, None
        if query is not None:
            try:
                self.pdh.PdhCloseQuery(query)
            except OSError:
                pass


def cuda_luid(device):
    """CUDA's device LUID (low, high), as DXGI's adapter LUID, from the driver API (CuPy's
    property cuts it at its first zero byte); None where it cannot be read."""
    if os.name != "nt":
        return None
    try:
        cuda = ct.WinDLL("nvcuda")
        handle, mask, raw = ct.c_int(), ct.c_uint(), (ct.c_char * 8)()
        if cuda.cuInit(0) or cuda.cuDeviceGet(ct.byref(handle), device):
            return None
        if cuda.cuDeviceGetLuid(raw, ct.byref(mask), handle):
            return None
        value = int.from_bytes(bytes(raw), "little")
        return value & 0xFFFFFFFF, value >> 32
    except (OSError, AttributeError):
        return None


class VRAM:
    def __init__(self, xp, note):
        self.xp, self.note = xp, note
        device = xp.cuda.runtime.getDevice()
        self.gpu = GPU(device, bus_id(xp, device))
        self.shared = Shared(cuda_luid(device))
        if os.name == "nt" and self.shared.query is None:
            note("Spill check off: the GPU Process Memory counter cannot be read")
        self.lowest = None  # the least of Ito's GPU memory seen in system RAM
        self.spilled = 0  # bytes more than that: VRAM overflowing into system RAM
        self.next_check = 0.0
        self.free_bytes = 0

    def free(self):
        """The lower of CUDA's and NVML's free VRAM: on Windows CUDA's runtime reports this
        process's budget (10.8 GiB free with another program holding 6 GiB of 12; NVML 4.9)."""
        free, _ = self.xp.cuda.runtime.memGetInfo()
        if self.gpu.library is not None:
            try:
                self.gpu.sample()
                free = min(free, self.gpu.free)
            except RuntimeError:
                self.gpu.busy = None
        return free

    def check(self):
        """Sample, at most once an INTERVAL: True if it did."""
        now = clock.now()
        if now < self.next_check:
            return False
        self.next_check = now + INTERVAL
        self.free_bytes = self.free()
        if self.shared.query is not None:
            try:
                used = self.shared.used()
            except OSError:
                self.shared.close()
                return True
            if self.lowest is None:
                self.note(f"GPU memory in system RAM at the start: {used / 2**20:.0f} MiB")
            self.lowest = used if self.lowest is None else min(self.lowest, used)
            spilled = used - self.lowest
            if (spilled >= SPILL_BYTES) != (self.spilled >= SPILL_BYTES):
                self.note(
                    f"VRAM full: {spilled / 2**20:.0f} MiB spilled into system RAM"
                    if spilled >= SPILL_BYTES
                    else "VRAM no longer spilling into system RAM"
                )
            self.spilled = spilled
        return True

    @property
    def spilling(self):
        return self.spilled >= SPILL_BYTES

    def close(self):
        self.shared.close()
        self.gpu.close()
