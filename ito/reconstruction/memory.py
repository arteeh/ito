"""VRAM: no fixed shares between Torch, CuPy and splats.

Torch/CuPy cached blocks are MASt3R's working set and count as Ito's own reserve; they are
released only when the device is nearly full. Only then are splat slots given up; the map
chooses redundant/unsupported surfaces before useful old ones. Spilling into shared
system memory would stall everything."""

from ito import clock
from ito.render.gpu import GPU

MARGIN_BYTES = 512 << 20
FLOOR_BYTES = 256 << 20
RELEASE_COOLDOWN = 10.0
SLOT_BYTES = 96
MIN_SLOTS = 16_384
INTERVAL = 0.5
# Torch's caching allocator keeps the blocks it frees for reuse. The trainer's tensors grow with
# the splat count, each a little larger than any block it freed, so its cache only grows: on the
# ten-minute tape 4.1 GiB allocated, 13.8 GiB reserved at 400 s. On Linux cudaMalloc fails at the
# card's size and the allocator then frees its cache and retries; on Windows the driver backs memory
# past the card with shared system memory, nothing fails, and the cache spilled into host RAM
# (11.3 GB working set at 452 s, Ceres out of memory). A cap on what Torch may reserve makes it free
# its cache there, as on Linux. HEADROOM is left for CuPy's splats, the renderer and the desktop.
TORCH_HEADROOM = 3 << 30
torch_cap = None  # bytes, once cap_torch() set it


def cap_torch():
    """Cap Torch's reserved CUDA memory (cache included) at the card minus TORCH_HEADROOM, at
    least half the card; returns the cap in bytes."""
    import torch

    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    cap = max(total // 2, total - TORCH_HEADROOM)
    torch.cuda.set_per_process_memory_fraction(cap / total)
    global torch_cap
    torch_cap = cap
    return cap


class VRAM:
    def __init__(self, xp, torch, note):
        self.xp, self.torch, self.note = xp, torch, note
        self.gpu = GPU(xp.cuda.runtime.getDevice())
        self.next_check = 0.0
        self.last_release = -RELEASE_COOLDOWN
        self.pressure = False
        self.fit = None
        self.free_bytes = 0
        self.cached_bytes = 0
        self.releases = 0

    def free(self):
        free, _ = self.xp.cuda.runtime.memGetInfo()
        if self.gpu.library is not None:
            try:
                self.gpu.sample()
                free = min(free, self.gpu.free)
            except RuntimeError:
                self.gpu.busy = None
        if self.torch is not None and torch_cap is not None:
            # Under the cap Torch frees its cache before growing: what it has not allocated is free.
            free = min(free, torch_cap - self.torch.cuda.memory_allocated())
        return free

    def cached(self):
        pool = self.xp.get_default_memory_pool()
        cached = pool.total_bytes() - pool.used_bytes()
        if self.torch is not None:
            cached += self.torch.cuda.memory_reserved() - self.torch.cuda.memory_allocated()
        return max(0, cached)

    def release(self, free, now):
        self.xp.get_default_memory_pool().free_all_blocks()
        if self.torch is not None:
            self.torch.cuda.empty_cache()
        self.last_release = now
        self.releases += 1
        self.note(
            f"VRAM {free / 2**30:.2f} GiB free: released "
            f"{self.cached_bytes / 2**30:.2f} GiB of cached CUDA blocks"
        )

    def check(self, slots, settled):
        now = clock.now()
        if now < self.next_check:
            return self.fit
        self.next_check = now + INTERVAL
        free = self.free_bytes = self.free()
        self.cached_bytes = self.cached()
        self.pressure = free < FLOOR_BYTES
        if free >= MARGIN_BYTES:
            self.fit = slots + (free - MARGIN_BYTES) // SLOT_BYTES
        elif not self.pressure:
            self.fit = slots if self.fit is None else min(self.fit, slots)
        elif self.cached_bytes >= FLOOR_BYTES - free and now - self.last_release >= (
            RELEASE_COOLDOWN
        ):
            self.release(free, now)
            self.fit = slots if self.fit is None else min(self.fit, slots)
        elif settled:
            shortfall = -(-(MARGIN_BYTES - free) // SLOT_BYTES)
            fit = max(MIN_SLOTS, slots - shortfall, slots // 2)
            if self.fit is None or fit < self.fit:
                self.fit = fit
                self.note(
                    f"VRAM {free / 2**30:.2f} GiB free, "
                    f"{self.cached_bytes / 2**30:.2f} GiB cached: splats limited to {fit:,}"
                )
        return self.fit

    def close(self):
        self.gpu.close()
