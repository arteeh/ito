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
