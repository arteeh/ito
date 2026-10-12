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

    points: object  # (H*W, 3) float32
    colors: object  # (H*W, 3) uint8
    valid: object  # (H*W,) bool
    shape: tuple[int, int]  # height, width of the image the points belong to
    intrinsics: tuple[float, float, float, float]  # fx, fy, cx, cy at that shape


class VisualResult(NamedTuple):
    reference: int | None  # keyframe (node id) the relative pose is against
    relative: np.ndarray | None  # 4x4 reference-from-frame; None without a usable match
    sigma: np.ndarray  # one sigma of the relative pose: rotation vector (rad), translation (m)
    match: float  # fraction of pixels matched, 0 to 1
    cloud: Cloud | None
    keyframe: bool  # this frame became a keyframe, under the node id it was tracked as


class Frontend(Protocol):
    def track(
        self, rgb: np.ndarray, node: int, predicted: np.ndarray, prior_sigma: np.ndarray
    ) -> VisualResult:
        """Track a frame fusion calls `node`, predicted at `predicted` within prior_sigma."""

    def correct(self, keyframe_poses: dict[int, np.ndarray], scale: float) -> None:
        """Fused keyframe poses back in; scale converts the front end's lengths to metres."""

    def forget(self, nodes: list[int]) -> None:
        """Keyframes fusion no longer holds."""


class MapAligner(Protocol):
    def align(
        self, rgb: np.ndarray, predicted: np.ndarray, intrinsics
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        """A frame's pose in the map's world, refined from `predicted` by comparing the map
        rendered there with the image: (world-from-camera, one sigma of its rotation vector
        (rad) and translation (m), pixels that agreed); None where the map cannot say."""
