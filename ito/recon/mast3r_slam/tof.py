"""Metres per map unit, once, from the robot's time-of-flight sensor: for display only.

MASt3R-SLAM maps in MASt3R's own units (its first keyframe's). Where the robot has a multizone
ToF beside its camera, each zone's range in metres against the depth MASt3R gives that patch of
the frame says how many metres one map unit is. The median over the first frames whose zones
agree is fixed as the map's display scale: the pilot's view and the splats are drawn at it, so
the room and the robot are the same size in VR. It never enters the solve.
"""

import math
import warnings
from collections import deque

import numpy as np

RANGE_MATCH = 0.035  # s: a grid this close to a frame's capture ranged what the frame shows
FRAMES = 20  # frames whose zones agree, before the scale is fixed
ZONES = 4  # zones a frame needs that see one surface each
SAMPLES = 5  # pixels across and down a zone's footprint
ZONE_SUPPORT = 0.6  # share of a zone's pixels that need a depth
ZONE_SPREAD = 0.15  # a zone's depths within this share of their median: one surface
AGREEMENT = 0.1  # median spread of a frame's zones' log ratios


def ray(across, up):
    """Unit rays at these angles right of and above a sensor's axis (-Z; +X right, +Y up)."""
    return np.stack(
        (np.cos(up) * np.sin(across), np.sin(up), -np.cos(up) * np.cos(across)), axis=-1
    )


class RangeScale:
    def __init__(self, K, shape):
        """K: the camera matrix of the frames MASt3R sees; shape: their height and width."""
        self.fx, self.fy, self.cx, self.cy = (
            float(K[i, j]) for i, j in ((0, 0), (1, 1), (0, 2), (1, 2))
        )
        self.height, self.width = shape
        self.grids = deque(maxlen=16)
        self.ratios = []
        self.value = None

    def add(self, grid):
        """grid: one RangeGrid as the pilot passes it on: time (pilot clock), mount (4x4
        sensor-to-camera, Ito axes), shape (rows, cols), field_of_view, distances (NaN: none)."""
        if self.value is None:
            self.grids.append(grid)

    def zones(self, grid):
        """Each usable zone's depth along the camera's axis in metres, and the flat indices of
        its footprint's pixels in the frame."""
        mount = np.asarray(grid["mount"], dtype=float)
        rows, cols = grid["shape"]
        width, height = grid["field_of_view"]
        steps = (np.arange(SAMPLES) + 0.5) / SAMPLES - 0.5
        targets, pixels = [], []
        for zone, distance in enumerate(grid["distances"]):
            if not math.isfinite(distance):
                continue
            row, col = divmod(zone, cols)
            across = ((col + 0.5) / cols - 0.5) * width
            up = (0.5 - (row + 0.5) / rows) * height
            a, u = np.meshgrid(across + steps * width / cols, up + steps * height / rows)
            spots = ray(a.ravel(), u.ravel()) * distance @ mount[:3, :3].T + mount[:3, 3]
            point = mount[:3, :3] @ (ray(across, up) * distance) + mount[:3, 3]
            if point[2] > -0.05 or (spots[:, 2] > -0.05).any():
                continue
            # Ito's camera axes (+Y up, -Z forward) to the frame's pixels (+y down).
            x = self.fx * spots[:, 0] / -spots[:, 2] + self.cx
            y = self.cy - self.fy * spots[:, 1] / -spots[:, 2]
            if x.min() < 0 or x.max() > self.width - 1 or y.min() < 0 or y.max() > self.height - 1:
                continue
            targets.append(-point[2])
            pixels.append(np.rint(y).astype(int) * self.width + np.rint(x).astype(int))
        return np.asarray(targets), np.asarray(pixels)

    def measure(self, captured, depth):
        """One frame: captured on the pilot clock, depth at its every pixel (flat, along the
        camera's axis, in map units, NaN where unknown; a torch tensor). True when this frame
        fixed the scale."""
        if self.value is not None or not self.grids:
            return False
        grid = min(self.grids, key=lambda g: abs(g["time"] - captured))
        if abs(grid["time"] - captured) > RANGE_MATCH:
            return False
        targets, pixels = self.zones(grid)
        if len(targets) < ZONES:
            return False
        import torch

        index = torch.as_tensor(pixels.ravel(), device=depth.device)
        sampled = depth[index].float().cpu().numpy().reshape(pixels.shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            supported = np.isfinite(sampled).mean(axis=1) >= ZONE_SUPPORT
            middle = np.nanmedian(sampled, axis=1)
            low, high = np.nanpercentile(sampled, (10, 90), axis=1)
        flat = supported & (middle > 0) & ((high - low) < ZONE_SPREAD * middle)
        if flat.sum() < ZONES:
            return False
        ratios = np.log(targets[flat] / middle[flat])
        centre = float(np.median(ratios))
        if float(np.median(np.abs(ratios - centre))) > AGREEMENT:
            return False
        self.ratios.append(centre)
        if len(self.ratios) < FRAMES:
            return False
        self.value = math.exp(float(np.median(self.ratios)))
        self.grids.clear()
        return True
