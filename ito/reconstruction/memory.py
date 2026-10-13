"""Keep Ito in dedicated VRAM, with no fixed shares between torch, CuPy and splats.

The worker builds its backend (MASt3R's weights included) before this sizes anything.
Twice a second it reads the device's free memory and how much of it Torch and CuPy hold
as cached, unused blocks. That cache is MASt3R's working set between frames: handing it
back makes every following frame allocate it again, which is slow (on Windows above all),
so it counts as Ito's own reserve and is released only when the device itself is nearly
full, at most once per cooldown. Only if that was not enough does Ito give up splat
slots, the longest-unseen splats first. The splat limit grows while the device has more
than the margin free. Spilling into shared system memory would stall reconstruction and
the display instead.
"""

from ito import clock
from ito.render.gpu import GPU

# Growth stops below this much free device memory; between FLOOR and MARGIN nothing changes.
MARGIN_BYTES = 512 << 20
# Below this the next allocation risks spilling into shared memory: release, then shrink.
FLOOR_BYTES = 256 << 20
RELEASE_COOLDOWN = 10.0
# A slot's GL record (64 B), its power-of-two sort entries (up to 16 B), and the worker's
# per-frame key matching (16 B).
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
        """Bytes Torch and CuPy hold for reuse without using them."""
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
        """Splat slots that fit now, or None before the first sample.

        slots: allocated slots; settled: an earlier shrink has given its slots back.
        """
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
            # Between floor and margin: keep what is allocated, grow no further.
            self.fit = slots if self.fit is None else min(self.fit, slots)
        elif self.cached_bytes >= FLOOR_BYTES - free and now - self.last_release >= (
            RELEASE_COOLDOWN
        ):
            # Cached, unused blocks go back without dropping any geometry.
            self.release(free, now)
            self.fit = slots if self.fit is None else min(self.fit, slots)
        elif settled:
            # At most half per step: the next readings show whether that was enough.
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
