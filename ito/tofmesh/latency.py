"""How late the camera frames really are, measured from the frames themselves.

A frame is posed at its exposure, which the pilot knows only as its arrival minus the driver's
capture_delay: a fixed guess. The real delay moves (the virtual Microduck's 83-123 ms within one
session, against the driver's 90), and a frame posed 20 ms late while the head turns 100
degrees a second is projected 2 degrees off: the projected view shakes with every head movement.

So the delay is measured. Each frame is shrunk to a grey thumbnail and phase-correlated with the
previous one: how far the image moved. The robot's own orientation track says how far it should
have moved for any delay. The extra delay (on top of capture_delay) whose predictions match the
last WINDOW seconds best is the one used. Only rotation is predicted (far points); walking's
parallax and moving objects are outliers the clipped cost ignores. Without motion there is
nothing to measure and the estimate stays.
"""

from collections import deque

import numpy as np

THUMB_WIDTH = 160
# The worker takes the newest frame whenever it is free (worker.TofMesh.submit): from a 30 Hz
# camera ~25 a second (live sim, 6 Nov), so a window holds ~25 pairs a second. An estimate uses
# the last WINDOW seconds and needs PAIRS of them moving: more than half, a head turning most of
# the window.
WINDOW = 3.0  # seconds of frame pairs an estimate uses
EVERY = 0.5  # seconds between estimates
MOVED = 1.0  # thumbnail pixels a pair must be predicted to move to count
CLIP = 5.0  # thumbnail pixels: a worse match counts no more than this
PAIRS = 40  # moving pairs an estimate needs
# A steady turn says nothing about the delay (every delay predicts the same image shift): the
# costs come out nearly flat. The estimate moves only to a delay that matches clearly better
# than the current one, by this fraction of the median cost.
BETTER = 0.05
# And only when the costs have one clear valley: the best at most this fraction of the median.
# Slow head movements leave image noise of the same size as the motion, and a jagged curve
# whose lowest point means nothing (a live sim run jumped from +0 to +185 ms on one).
VALLEY = 0.65
PEAK = 0.08  # phase-correlation peak below which a pair is no match at all
# Extra delays tried, seconds on top of the driver's capture_delay.
CANDIDATES = np.round(np.arange(-0.09, 0.2101, 0.005), 3)
STEP = 0.005  # seconds between the orientation track's samples for the candidates


def _thumbnail(image):
    """Thumbnail about THUMB_WIDTH wide (block means of the green channel, a fair grey at a
    fifth of the cost), and its scale from `image`."""
    h, w = image.shape[:2]
    k = max(1, w // THUMB_WIDTH)
    grey = image[: h // k * k, : w // k * k]
    if grey.ndim == 3:
        grey = grey[..., min(1, grey.shape[2] - 1)]
    blocks = grey.reshape(h // k, k, w // k, k).sum(axis=(1, 3), dtype=np.uint32)
    return blocks.astype(np.float32) / (k * k), k


def _shift(a, b):
    """(dx, dy) that `b` is `a` moved by, in pixels, and the correlation peak (0..1)."""
    cross = a * np.conj(b)
    cross /= np.abs(cross) + 1e-9
    surface = np.fft.ifft2(cross).real
    h, w = surface.shape
    y, x = np.unravel_index(int(np.argmax(surface)), surface.shape)
    peak = float(surface[y, x])

    def refine(minus, centre, plus):
        d = minus - 2 * centre + plus
        return 0.0 if d >= 0 else 0.5 * (minus - plus) / d

    dx = x + refine(surface[y, x - 1], surface[y, x], surface[y, (x + 1) % w])
    dy = y + refine(surface[y - 1, x], surface[y, x], surface[(y + 1) % h, x])
    dx = dx - w if dx > w / 2 else dx
    dy = dy - h if dy > h / 2 else dy
    return -dx, -dy, peak


def _matrices(q):
    """3x3 rotations of (n, 4) xyzw quaternions."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def _track(times, quaternions, at, gap):
    """Rotations at times `at` from a sampled track (nlerp; NaN across a gap or outside)."""
    i = np.clip(np.searchsorted(times, at), 1, len(times) - 1)
    t0, t1 = times[i - 1], times[i]
    f = np.clip((at - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)[:, None]
    q0, q1 = quaternions[i - 1], quaternions[i].copy()
    q1[np.sum(q0 * q1, axis=1) < 0] *= -1
    out = _matrices(q0 * (1 - f) + q1 * f)
    bad = (at < times[0]) | (at > times[-1]) | (t1 - t0 > gap)
    out[bad] = np.nan
    return out


class FrameDelay:
    def __init__(self):
        self.extra = 0.0  # seconds to add to the driver's capture_delay
        self.previous = None  # (stamp, spectrum)
        self.window = None
        self.pairs = deque()  # (stamp a, stamp b, dx, dy) in thumbnail pixels
        self.scale = None  # thumbnail pixels per frame pixel
        self.last_estimate = -1e9
        self.measured = 0  # estimates made
        self.cost = None  # mean clipped mismatch (thumbnail pixels) of the latest estimate

    def frame(self, image, stamp):
        """A frame as it arrives, at its stamp (arrival minus capture_delay)."""
        thumb, k = _thumbnail(image)
        if self.window is None or self.window.shape != thumb.shape:
            h, w = thumb.shape
            self.window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
            self.previous = None
        self.scale = 1.0 / k
        spectrum = np.fft.fft2(thumb * self.window)
        if self.previous is not None and 0 < stamp - self.previous[0] <= 0.2:
            dx, dy, peak = _shift(self.previous[1], spectrum)
            if peak >= PEAK:
                self.pairs.append((self.previous[0], stamp, dx, dy))
        self.previous = (stamp, spectrum)
        while self.pairs and stamp - self.pairs[0][0] > WINDOW:
            self.pairs.popleft()

    def due(self, now):
        return now - self.last_estimate >= EVERY and len(self.pairs) >= PAIRS

    def estimate(self, now, times, quaternions, intrinsics, gap=0.1):
        """Re-measure every EVERY seconds against an orientation track: `times` (n,), xyzw
        `quaternions` (n, 4) of world_from_camera, the frames' pinhole (fx, fy, cx, cy)."""
        if not self.due(now) or len(times) < 2:
            return
        self.last_estimate = now
        pairs = np.array(self.pairs)
        a, b, measured = pairs[:, 0], pairs[:, 1], pairs[:, 2:]
        # One orientation track sample per STEP covers every candidate of every pair.
        start = a.min() - CANDIDATES.max() - STEP
        grid = start + STEP * np.arange(int((b.max() - CANDIDATES.min() - start) / STEP) + 2)
        rotations = _track(np.asarray(times), np.asarray(quaternions, np.float64), grid, gap)

        def at(t):
            return rotations[np.clip(np.rint((t - start) / STEP).astype(int), 0, len(grid) - 1)]

        fx, fy = intrinsics[0] * self.scale, intrinsics[1] * self.scale
        ahead = np.array([0.0, 0.0, -1.0])
        predictions = []
        for c in CANDIDATES:
            ra, rb = at(a - c), at(b - c)
            # Where a far point straight ahead in frame a lands in frame b.
            d = np.einsum("nji,nj->ni", rb, ra @ ahead)
            with np.errstate(invalid="ignore", divide="ignore"):
                x, y = fx * d[:, 0] / -d[:, 2], -fy * d[:, 1] / -d[:, 2]
            predictions.append(np.stack([x, y], -1))
        predictions = np.array(predictions)  # (candidates, pairs, 2)
        # Every candidate is judged on the same pairs: those the track covers for all of them
        # and that moved enough at the current estimate.
        use = np.isfinite(predictions).all(axis=(0, 2))
        current = int(np.argmin(np.abs(CANDIDATES - self.extra)))
        use &= np.linalg.norm(np.nan_to_num(predictions[current]), axis=1) >= MOVED
        if use.sum() < PAIRS:
            return
        miss = np.linalg.norm(predictions[:, use] - measured[use][None], axis=2)
        costs = np.minimum(miss, CLIP).mean(axis=1)
        best = int(np.argmin(costs))
        self.cost = float(costs[best])
        typical = float(np.median(costs))
        if costs[best] > VALLEY * typical or costs[current] - costs[best] < BETTER * typical:
            return
        offset = 0.0
        if 0 < best < len(costs) - 1:  # the minimum between grid points (parabola)
            minus, centre, plus = costs[best - 1 : best + 2]
            curve = minus - 2 * centre + plus
            offset = 0.5 * (minus - plus) / curve if curve > 0 else 0.0
        self.extra = float(CANDIDATES[best] + offset * (CANDIDATES[1] - CANDIDATES[0]))
        self.measured += 1
