# ruff: noqa: E402
"""Why do fused poses lose dB? Per-frame error of the poses train.py actually uses
(aligned, interpolated fused) against truth: constant rotation bias, time offset, scale,
interpolation between sparse estimates, and pixel size of the errors."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repo/e2e"))
from truth import align  # noqa: E402

run = sys.argv[1]
out = ROOT / "out" / run
source = ROOT.parent / "rf-run/out" / run
d = np.load(out / "data.npz")
times = d["times"] - d["times"][0]
truth = d["poses"]
e = np.load(out / "fused-extracted.npz")
ids, est = e["ids"], e["estimates"]
est = np.linalg.inv(align(est[0], truth[ids[0]])) @ est
intr = json.loads((source / "rec/camera.json").read_text())
fx = intr["fx"]
n = len(times)
test = np.arange(7, n, 8)


def rot_err(a, b):
    return np.degrees(
        (Rotation.from_matrix(a[:, :3, :3]).inv() * Rotation.from_matrix(b[:, :3, :3])).magnitude()
    )


def interp(t_query, t_known, poses):
    t = np.clip(t_query, t_known[0], t_known[-1])
    out = np.repeat(np.eye(4)[None], len(t), 0)
    out[:, :3, :3] = Slerp(t_known, Rotation.from_matrix(poses[:, :3, :3]))(t).as_matrix()
    for k in range(3):
        out[:, k, 3] = np.interp(t, t_known, poses[:, k, 3])
    return out


# 1. Error at the estimate frames themselves (no interpolation).
_, first = np.unique(ids, return_index=True)
ids_u, est_u = ids[first], est[first]
order = np.argsort(times[ids_u])
ids_u, est_u = ids_u[order], est_u[order]
at_est = rot_err(est_u, truth[ids_u])
# 2. Error of the interpolated poses train.py uses, at every frame and at held-out frames.
used = interp(times, times[ids_u], est_u)
all_err = rot_err(used, truth)
trans_err = np.linalg.norm(used[:, :3, 3] - truth[:, :3, 3], axis=1)
# 3. Truth's own angular speed (deg/s) per frame.
tr = Rotation.from_matrix(truth[:, :3, :3])
speed = np.zeros(n)
speed[1:] = np.degrees((tr[:-1].inv() * tr[1:]).magnitude()) / np.maximum(np.diff(times), 1e-3)
# 4. Constant time offset: compare estimates with truth sampled at t + dt.
offsets = np.arange(-0.15, 0.1501, 0.0025)
med = []
for dt in offsets:
    shifted = interp(times[ids_u] + dt, times, truth)
    med.append(float(np.median(rot_err(est_u, shifted))))
best = float(offsets[int(np.argmin(med))])
# 5. Constant rotation bias in the camera frame (mean residual rotation).
resid = Rotation.from_matrix(truth[ids_u, :3, :3]).inv() * Rotation.from_matrix(est_u[:, :3, :3])
bias = resid.mean()
corrected = np.array(est_u)
corrected[:, :3, :3] = (Rotation.from_matrix(est_u[:, :3, :3]) * bias.inv()).as_matrix()
after_bias = rot_err(corrected, truth[ids_u])
# 6. Translation scale: path length ratio.
path_est = np.linalg.norm(np.diff(est_u[:, :3, 3], axis=0), axis=1).sum()
path_true = np.linalg.norm(np.diff(truth[ids_u, :3, 3], axis=0), axis=1).sum()
# Residual axes in camera frame (x right, y up, z back per Ito convention).
rv = np.degrees(resid.as_rotvec())
slow = speed[ids_u] < 30
fast = speed[ids_u] >= 90
corr = float(np.corrcoef(speed[ids_u], at_est)[0, 1])


def q(x):
    return [round(float(v), 3) for v in np.percentile(x, [50, 95])] if len(x) else None


report = {
    "run": run,
    "frames": n,
    "estimates": len(ids_u),
    "estimate_rate_hz": round(len(ids_u) / (times[ids_u[-1]] - times[ids_u[0]]), 2),
    "fx_px": fx,
    "rot_deg_at_estimates_p50_p95": q(at_est),
    "rot_deg_interpolated_all_p50_p95": q(all_err),
    "rot_deg_interpolated_heldout_p50_p95": q(all_err[test]),
    "rot_px_heldout_p50_p95": q(np.radians(all_err[test]) * fx),
    "trans_m_heldout_p50_p95": q(trans_err[test]),
    "speed_deg_s_p50_p95": q(speed),
    "rot_deg_at_estimates_slow_lt30": q(at_est[slow]),
    "rot_deg_at_estimates_fast_ge90": q(at_est[fast]),
    "corr_speed_vs_error": round(corr, 3),
    "best_time_offset_s": best,
    "median_rot_deg_at_offset_0": round(med[int(np.argmin(np.abs(offsets)))], 3),
    "median_rot_deg_at_best_offset": round(min(med), 3),
    "mean_residual_rotvec_deg_xyz": [round(float(v), 3) for v in rv.mean(0)],
    "residual_rotvec_std_deg_xyz": [round(float(v), 3) for v in rv.std(0)],
    "rot_deg_after_removing_constant_bias_p50_p95": q(after_bias),
    "path_ratio_est_over_true": round(float(path_est / path_true), 4),
}
(out / "pose-gap.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
