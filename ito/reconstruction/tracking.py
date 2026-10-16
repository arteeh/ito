"""The visual front end's boundary: all fusion needs from MASt3R today, DPVO or DROID later.

A front end tracks each frame against one of its keyframes, seeded at the pose fusion
predicts and pulled toward it (its prior) where the image says little, and hands back the
relative pose it measured with its confidence, and the frame's points for splats. Keyframes
carry the node ids fusion gives frames, and fusion writes their fused poses back.
Poses are world-from-camera in Ito axes (+Y up, -Z forward).
"""

from typing import NamedTuple, Protocol

import numpy as np


class Cloud(NamedTuple):
    """One frame's pointmap on the device, in the camera frame (Ito axes), in metres."""

    points: object
    colors: object
    valid: object
    shape: tuple[int, int]
    intrinsics: tuple[float, float, float, float]  # fx, fy, cx, cy at that shape


class VisualResult(NamedTuple):
    reference: int | None
    relative: np.ndarray | None  # 4x4 reference-from-frame; None without a usable match
    sigma: np.ndarray  # one sigma of the relative pose: rotation vector (rad), translation (m)
    match: float
    cloud: Cloud | None
    keyframe: bool


class Frontend(Protocol):
    def track(
        self, rgb: np.ndarray, node: int, predicted: np.ndarray, prior_sigma: np.ndarray
    ) -> VisualResult:
        pass

    def correct(self, keyframe_poses: dict[int, np.ndarray], scale: float) -> None:
        pass

    def forget(self, nodes: list[int]) -> None:
        pass


class MapAligner(Protocol):
    def align(
        self, rgb: np.ndarray, predicted: np.ndarray, intrinsics
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        pass
