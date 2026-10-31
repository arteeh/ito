"""VRAM: no limits. What Ito takes is what its settings ask for (precision, Input resolution,
Map size, Max splats); nothing is given up to fit the card. The worker only samples free VRAM
and the GPU's load for the stats."""

from ito import clock
from ito.render.gpu import GPU, bus_id

INTERVAL = 1.0


class VRAM:
    def __init__(self, xp):
        self.xp = xp
        device = xp.cuda.runtime.getDevice()
        self.gpu = GPU(device, bus_id(xp, device))
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
        return True

    def close(self):
        self.gpu.close()
