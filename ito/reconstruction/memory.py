"""Keep Ito in dedicated VRAM, with no fixed shares between torch, CuPy and splats.

The worker builds its backend (MASt3R's weights included) before this sizes anything.
Twice a second it reads the device's free memory. Below a small margin it first returns
the allocators' cached blocks, and only if that was not enough gives up splat slots, the
longest-unseen splats first. With room for twice the margin, the splat limit grows again.
Spilling into shared system memory would stall reconstruction and the display instead.
"""

from ito import clock
from ito.render.gpu import GPU

MARGIN_BYTES = 512 << 20
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
        self.released = False
        self.pressure = False
        self.fit = None

    def free(self):
        free, _ = self.xp.cuda.runtime.memGetInfo()
        if self.gpu.library is not None:
            try:
                self.gpu.sample()
                free = min(free, self.gpu.free)
            except RuntimeError:
                self.gpu.busy = None
        return free

    def check(self, slots, settled):
        """Splat slots that fit now, or None before the first sample.

        slots: allocated slots; settled: an earlier shrink has given its slots back.
        """
        now = clock.now()
        if now < self.next_check:
            return self.fit
        self.next_check = now + INTERVAL
        free = self.free()
        began, self.pressure = not self.pressure, free < MARGIN_BYTES
        if not self.pressure:
            self.released = False
            if free >= 2 * MARGIN_BYTES:
                self.fit = slots + (free - 2 * MARGIN_BYTES) // SLOT_BYTES
            else:
                # Between one and two margins: keep what is allocated, grow no further.
                self.fit = slots if self.fit is None else min(self.fit, slots)
        elif not self.released:
            # Cached, unused blocks go back without dropping any geometry.
            self.xp.get_default_memory_pool().free_all_blocks()
            if self.torch is not None:
                self.torch.cuda.empty_cache()
            self.released = True
            if began:
                self.note(f"VRAM {free / 2**30:.2f} GiB free: released cached CUDA blocks")
        elif settled:
            # At most half per step: the next readings show whether that was enough.
            shortfall = -(-(MARGIN_BYTES - free) // SLOT_BYTES)
            fit = max(MIN_SLOTS, slots - shortfall, slots // 2)
            if self.fit is None or fit < self.fit:
                self.fit = fit
                self.note(
                    f"VRAM {free / 2**30:.2f} GiB free after releasing caches: "
                    f"splats limited to {fit:,}"
                )
            self.released = False
        return self.fit

    def close(self):
        self.gpu.close()
