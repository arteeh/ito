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
NVIDIA = 0x10DE
IID_FACTORY1 = "770aae78-f26f-4dba-a829-253c83d1b387"
IID_ADAPTER3 = "645967a4-1392-4310-a798-8053ce3e93fd"


class _AdapterDesc1(ct.Structure):
    _fields_ = [
        ("description", ct.c_wchar * 128),
        ("vendor", ct.c_uint),
        ("device", ct.c_uint),
        ("subsystem", ct.c_uint),
        ("revision", ct.c_uint),
        ("dedicated_video", ct.c_size_t),
        ("dedicated_system", ct.c_size_t),
        ("shared_system", ct.c_size_t),
        ("luid", ct.c_uint32 * 2),
        ("flags", ct.c_uint),
    ]


class _MemoryInfo(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in ("budget", "usage", "available", "reserved")]


def _guid(text):
    import uuid

    return uuid.UUID(text).bytes_le


def _call(obj, index, *args, result=None):
    """COM method index of obj's vtable; HRESULT failures raise OSError."""
    vtable = ct.cast(obj, ct.POINTER(ct.POINTER(ct.c_void_p)))[0]
    byref = type(ct.byref(ct.c_int()))
    types = [ct.c_void_p if isinstance(arg, byref) else type(arg) for arg in args]
    prototype = ct.WINFUNCTYPE(result or ct.HRESULT, ct.c_void_p, *types)
    return prototype(vtable[index])(obj, *args)


def _release(obj):
    if obj:
        _call(obj, 2, result=ct.c_ulong)


class Shared:
    """Windows: how much of this process's GPU memory on the card CUDA runs on is in system RAM
    (DXGI's non-local segment, per process). adapter None where it cannot be read."""

    def __init__(self, name):
        self.adapter = None
        if os.name != "nt":
            return
        try:
            dxgi = ct.WinDLL("dxgi")
            dxgi.CreateDXGIFactory1.argtypes = [ct.c_char_p, ct.POINTER(ct.c_void_p)]
            dxgi.CreateDXGIFactory1.restype = ct.HRESULT
            factory = ct.c_void_p()
            dxgi.CreateDXGIFactory1(_guid(IID_FACTORY1), ct.byref(factory))
            try:
                self.adapter = self._find(factory, name)
            finally:
                _release(factory)
            if self.adapter is not None:
                self.used()
        except OSError:
            self.close()

    @staticmethod
    def _find(factory, name):
        """The NVIDIA adapter named as CUDA's device, else the only NVIDIA one; None if unsure."""
        found = []
        try:
            for index in range(16):
                adapter = ct.c_void_p()
                try:
                    _call(factory, 12, ct.c_uint(index), ct.byref(adapter))  # EnumAdapters1
                except OSError:
                    break  # DXGI_ERROR_NOT_FOUND: no more
                desc = _AdapterDesc1()
                found.append((adapter, desc))
                _call(adapter, 10, ct.byref(desc))  # GetDesc1
            nvidia = [(a, d) for a, d in found if d.vendor == NVIDIA]
            named = [(a, d) for a, d in nvidia if d.description == name]
            chosen = named if len(named) == 1 else nvidia
            if len(chosen) != 1:
                return None
            adapter3 = ct.c_void_p()
            _call(chosen[0][0], 0, ct.c_char_p(_guid(IID_ADAPTER3)), ct.byref(adapter3))
            return adapter3
        finally:
            for adapter, _ in found:
                _release(adapter)

    def used(self):
        info = _MemoryInfo()
        # QueryVideoMemoryInfo(node 0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)
        _call(self.adapter, 14, ct.c_uint(0), ct.c_int(1), ct.byref(info))
        return info.usage

    def close(self):
        adapter, self.adapter = self.adapter, None
        try:
            _release(adapter)
        except OSError:
            pass


class VRAM:
    def __init__(self, xp, note):
        self.xp, self.note = xp, note
        device = xp.cuda.runtime.getDevice()
        self.gpu = GPU(device, bus_id(xp, device))
        name = xp.cuda.runtime.getDeviceProperties(device)["name"]
        self.shared = Shared(name.decode() if isinstance(name, bytes) else name)
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
        if self.shared.adapter is not None:
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
