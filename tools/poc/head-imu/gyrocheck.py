# ruff: noqa: E501, B905  (analysis script for the PoC, not product code)
"""Head IMU vs the sim's truth on a tape: integrate the gyro (mounted into Ito's camera axes)
between frames and compare with the true relative camera rotation; scan a time offset.
Usage: gyrocheck.py TAPE"""

import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

tape = sys.argv[1]
frames = [json.loads(x) for x in open(f"{tape}/frames.jsonl")]
truth = [f for f in frames if "orientation" in f]
print("frame keys (truth line):", sorted(truth[0].keys()))
stamp = (
    "captured" if "captured" in truth[0] else ("exposed" if "exposed" in truth[0] else "received")
)
lag = np.median([f["received"] - f[stamp] for f in truth])
print("stamp", stamp, "median received - stamp", round(lag, 4))
msgs = [json.loads(x) for x in open(f"{tape}/sensors.jsonl")]
mount = None
ts, w = [], []
for m in msgs:
    for u in m["imu"]:
        mount = Rot.from_quat(u["mount"]["orientation"]).as_matrix()
        for s in u["samples"]:
            ts.append(s["time"])
            w.append(s["angular_velocity"])
ts, w = np.array(ts), np.array(w)
order = np.argsort(ts)
ts, w = ts[order], w[order]
wc = w @ mount.T  # camera (Ito axes) angular velocity


def integrate(t0, t1):
    """R_{C(t0) C(t1)} from body rates, piecewise constant (sample i holds until i+1)."""
    R = np.eye(3)
    i = max(0, np.searchsorted(ts, t0) - 1)
    t = t0
    while t < t1 and i < len(ts):
        end = min(t1, ts[i + 1] if i + 1 < len(ts) else t1)
        if end > t:
            R = R @ Rot.from_rotvec(wc[i] * (end - t)).as_matrix()
        t = end
        i += 1
    return R


def rel_true(a, b):
    Ra = Rot.from_quat(a["orientation"]).as_matrix()
    Rb = Rot.from_quat(b["orientation"]).as_matrix()
    return Ra.T @ Rb


pairs = [(truth[i], truth[i + 4]) for i in range(0, len(truth) - 4, 7)]
for offset in (-0.06, -0.03, 0.0, 0.015, 0.03, 0.06, 0.09):
    errs, angs = [], []
    for a, b in pairs:
        Rt = rel_true(a, b)
        Rg = integrate(a[stamp] + offset, b[stamp] + offset)
        errs.append(np.degrees(np.linalg.norm(Rot.from_matrix(Rt.T @ Rg).as_rotvec())))
        angs.append(np.degrees(np.linalg.norm(Rot.from_matrix(Rt).as_rotvec())))
    errs, angs = np.array(errs), np.array(angs)
    big = angs > 5
    print(
        f"offset {offset:+.3f}: err med {np.median(errs):.2f} deg p95 {np.percentile(errs, 95):.2f}; "
        f"pairs turning >5deg ({big.sum()}): err med {np.median(errs[big]) if big.any() else 0:.2f} "
        f"vs true turn med {np.median(angs[big]) if big.any() else 0:.1f}"
    )
# Rates: how fast does the head turn?
rate = np.degrees(np.linalg.norm(wc, axis=1))
print(
    "gyro |w| deg/s p50 %.0f p95 %.0f p99 %.0f max %.0f"
    % tuple(np.percentile(rate, [50, 95, 99, 100]))
)
