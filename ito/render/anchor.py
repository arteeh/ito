import numpy as np
from scipy.spatial.transform import Rotation

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


# Fusion's corrections are held while the pilot is still and eased during motion.
CORRECTION_TURN = np.radians(2.0)  # rad/s
CORRECTION_SLIDE = 0.05  # m/s
MOVING_TURN = np.radians(5.0)  # rad/s
MOVING_SLIDE = 0.02  # m/s


def _angle(rotation):
    return float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))


def _toward_identity(rotation, angle):
    vector = Rotation.from_matrix(rotation).as_rotvec()
    full = np.linalg.norm(vector)
    if full <= angle or full < 1e-9:
        return np.eye(3)
    return Rotation.from_rotvec(vector * (1 - angle / full)).as_matrix()


class CorrectionEase:
    def __init__(self):
        self.correction = None
        self.turn = np.eye(3)
        self.slide = np.zeros(3)
        self.previous = None

    def __call__(self, correction, anchor, head, now):
        correction = np.asarray(correction, dtype=np.float64)
        robot = np.asarray(anchor[:3, 3], dtype=np.float64)
        eye = correction[:3, :3] @ robot + correction[:3, 3]
        if self.correction is not None and not np.array_equal(correction, self.correction):
            change = correction[:3, :3] @ self.correction[:3, :3].T
            self.turn = self.turn @ change.T
            before = self.correction[:3, :3] @ robot + self.correction[:3, 3]
            self.slide = self.slide + before - eye
        self.correction = correction
        head_rotation = None if head is None else np.asarray(head, np.float64)[:3, :3]
        drawn = eye + self.slide
        if self.previous is not None and now > self.previous[0]:
            dt = now - self.previous[0]
            turned = 0.0
            if head_rotation is not None and self.previous[1] is not None:
                turned = _angle(self.previous[1].T @ head_rotation) / dt
            travelled = np.linalg.norm(drawn - self.previous[2]) / dt
            if turned > MOVING_TURN or travelled > MOVING_SLIDE:
                self.turn = _toward_identity(self.turn, CORRECTION_TURN * dt)
                distance = np.linalg.norm(self.slide)
                if distance > 1e-9:
                    self.slide *= max(0.0, 1 - CORRECTION_SLIDE * dt / distance)
        drawn = eye + self.slide
        self.previous = (now, head_rotation, drawn)
        # The renderer draws world_from_eye = anchor @ head: a turn of the displayed world
        # about the eye is the inverse turn of the pilot's view.
        result = np.eye(4, dtype=np.float32)
        result[:3, :3] = self.turn.T @ np.asarray(anchor[:3, :3], np.float64)
        result[:3, 3] = drawn
        return result
