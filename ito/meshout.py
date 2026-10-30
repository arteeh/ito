"""The worker's feed to the mesh world (ito/render/mesh.py): every placed frame's depth, image
and pose, and the keyframe corrections the map follows. Light on purpose: the worker imports it.

ITO_MESH (an experiment, off by default):
- over: the mesh world's NOW layer (the newest frame as a textured depth mesh) drawn over the
  splats; ITO_MESH_MEMORY picks what it remembers beside them (default none)
- only: the mesh world alone, no splats drawn (ITO_MESH_MEMORY default tsdf)
"""

import os
import queue
from dataclasses import dataclass

import numpy as np

MESH = os.environ.get("ITO_MESH", "0")
if MESH == "1":
    MESH = "over"
ENABLED = MESH in ("over", "only")
# The longest side of the image a frame carries; its depth stays on the network's grid.
RGB_LONG = int(os.environ.get("ITO_MESH_RGB", "1024"))


@dataclass
class MeshFrame:
    rgb: np.ndarray  # H x W x 3 uint8
    depth: np.ndarray  # h x w axial metres (float16/32), 0 where unknown; same view as rgb
    intrinsics: tuple  # fx, fy, cx, cy in depth pixels, pixel (i, j) centred at (i + .5, j + .5)
    pose: np.ndarray  # 4x4 world-from-camera, Ito axes (x right, y up, looking down -z)
    captured: float
    anchor: int = -1  # the keyframe whose corrections this frame follows (-1: none)
    key: bool | None = None  # keep as a memory view; None: the mesh world decides by motion
    scale: float = 1.0  # depth multiplier (a keyframe's similarity correction can rescale it)


class MeshSink:
    """Worker side: frames are dropped when the window has not taken the last few (it never
    waits); corrections wait briefly, since frames placed before them need them."""

    def __init__(self, out):
        self.out = out
        self.dropped = 0
        self.lost_moves = 0

    def frame(self, depth, pose, intrinsics, rgb, captured, anchor):
        import cv2

        h, w = rgb.shape[:2]
        if max(h, w) > RGB_LONG:
            f = RGB_LONG / max(h, w)
            rgb = cv2.resize(rgb, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
        if rgb.dtype != np.uint8:
            rgb = np.clip(np.asarray(rgb, np.float32) * (255 if rgb.max() <= 1.0 else 1), 0, 255)
            rgb = rgb.astype(np.uint8)
        fx, fy, cx, cy = (float(v) for v in intrinsics)
        frame = MeshFrame(
            rgb=np.ascontiguousarray(rgb),
            depth=np.asarray(depth, np.float16),
            intrinsics=(fx, fy, cx + 0.5, cy + 0.5),
            pose=np.asarray(pose, np.float32),
            captured=float(captured),
            anchor=int(anchor),
        )
        try:
            self.out.put_nowait(("frame", frame))
        except queue.Full:
            self.dropped += 1

    def moved(self, deltas):
        """deltas: keyframe -> 4x4 world similarity (Ito axes) its surfaces just followed."""
        moves = {int(k): np.asarray(d, np.float64) for k, d in deltas.items()}
        try:
            self.out.put(("move", moves), timeout=0.5)
        except queue.Full:
            self.lost_moves += 1


def drain(out, limit=16):
    items = []
    for _ in range(limit):
        try:
            items.append(out.get_nowait())
        except queue.Empty:
            break
    return items
