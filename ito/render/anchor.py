import numpy as np

from .scene import FloatArray

LONGEST_INTERVAL = 0.6
LARGEST_STEP = 1.0  # Metres; a bigger move is a relocalization, not motion.


class AnchorGlide:
    def __init__(self):
        self.latest: FloatArray | None = None
        self.arrived = 0.0
        self.interval = 0.0
        self.start = self.target = np.zeros(3, dtype=np.float32)
        self.began = 0.0

    def position(self, now: float) -> FloatArray:
        if self.interval <= 0:
            return self.target
        progress = min(1.0, max(0.0, (now - self.began) / self.interval))
        return self.start + progress * (self.target - self.start)

    def __call__(self, anchor: FloatArray, now: float) -> FloatArray:
        if anchor is not self.latest:
            target = np.asarray(anchor[:3, 3], dtype=np.float32)
            drawn = self.position(now)
            interval = now - self.arrived
            if (
                self.latest is None
                or interval > LONGEST_INTERVAL
                or np.linalg.norm(target - drawn) > LARGEST_STEP
            ):
                self.interval = 0.0
            else:
                self.interval = (
                    interval if self.interval <= 0 else 0.7 * self.interval + 0.3 * interval
                )
            self.start, self.target, self.began = drawn, target, now
            self.latest, self.arrived = anchor, now
        glided = np.array(anchor, dtype=np.float32)
        glided[:3, 3] = self.position(now)
        return glided
