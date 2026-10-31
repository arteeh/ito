"""VRAM: no fixed shares between Torch, CuPy and splats.

Torch/CuPy cached blocks are MASt3R's working set and count as Ito's own reserve; they are
released only when the device is nearly full. Only then are splat slots given up, the
longest-unseen first. Spilling into shared system memory would stall everything."""

from ito import clock
from ito.render.gpu import GPU

MARGIN_BYTES = 512 << 20
FLOOR_BYTES = 256 << 20
RELEASE_COOLDOWN = 10.0
SLOT_BYTES = 96
MIN_SLOTS = 16_384
INTERVAL = 0.5


class VRAM:
    def __init__(self, xp, torch, note, shed=None):
        self.xp, self.torch, self.note = xp, torch, note
        # Gives back more VRAM when the caches were not enough (MASt3R's CUDA graphs): True if
        # it freed anything. Before the splats, which fade for little (96 B a slot).
        self.shed = shed
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
        """Hand the cached blocks back to the driver; the free VRAM after it. What is reported
        released is measured: the estimate (cached) also counts blocks that cannot go, as the
        CUDA graphs' pool's."""
        self.xp.get_default_memory_pool().free_all_blocks()
        if self.torch is not None:
            self.torch.cuda.empty_cache()
        self.last_release = now
        self.releases += 1
        after = self.free()
        self.cached_bytes = self.cached()
        self.note(
            f"VRAM {free / 2**30:.2f} GiB free: released "
            f"{max(0, after - free) / 2**30:.2f} GiB of cached CUDA blocks"
        )
        return after

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
            return self.fit
        if (
            self.pressure
            and self.cached_bytes >= FLOOR_BYTES - free
            and now - self.last_release >= RELEASE_COOLDOWN
        ):
            free = self.free_bytes = self.release(free, now)
            self.pressure = free < FLOOR_BYTES
            if self.pressure and self.shed is not None and self.shed():
                # Measured again: the splats give up only what is still short.
                free = self.free_bytes = self.free()
                self.cached_bytes = self.cached()
                self.pressure = free < FLOOR_BYTES
                self.note(f"VRAM {free / 2**30:.2f} GiB free after dropping MASt3R's CUDA graphs")
        if not self.pressure:
            self.fit = slots if self.fit is None else min(self.fit, slots)
        elif settled:
            # Still short after any release: splats give up slots.
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
