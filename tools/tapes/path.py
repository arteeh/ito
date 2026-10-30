"""The one camera path every candidate is filmed along on a tape: the sim's exact camera pose of
each frame (Ito's world), smoothed with a Gaussian of SIGMA seconds so the film is watchable
while shifts, swims and doubling in the map stay visible. Indexed by tape time (seconds after
the tape's first frame was received)."""

import numpy as np

SIGMA = 0.35  # s


def camera_path(rec, sigma=SIGMA):
    frames = rec.truth_frames()
    t0 = rec.frames[0]["received"]
    times = np.array([f["received"] - t0 for f in frames])
    poses = np.array([f["pose"] for f in frames])
    rate = (len(times) - 1) / max(times[-1] - times[0], 1e-6)
    radius = max(1, int(3 * sigma * rate))
    weights = np.exp(-0.5 * (np.arange(-radius, radius + 1) / (sigma * rate)) ** 2)
    from scipy.spatial.transform import Rotation

    quats = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    n = len(times)
    out = poses.copy()
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        w = weights[lo - i + radius : hi - i + radius]
        w = w / w.sum()
        out[i, :3, 3] = w @ poses[lo:hi, :3, 3]
        q = w @ quats[lo:hi]
        out[i, :3, :3] = Rotation.from_quat(q / np.linalg.norm(q)).as_matrix()
    return times, out
