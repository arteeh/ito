"""Pose sets against a recording's exact poses (supporting numbers, never a gate).

    traj.py REC [POSES.npz ...] [--robot] [--labels ...]

POSES.npz: seq + poses (world-from-camera, Ito axes), as gs.py/replay.py/importers write them.
--robot adds the robot's own kinematic camera pose (motion.jsonl) at each frame's exposure.
Per set: absolute error after one rigid alignment (SE3, and Sim3 for vision-only sets), and the
error of RELATIVE poses between frames dt apart (dt 0.1, 1, 5 s) and between co-visible frames
(optical axes within 30 deg, centres within 1 m): what decides whether their splats agree."""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).parent))
from data import Recording  # noqa: E402


def umeyama(src, dst, scale):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d.T @ s / len(src))
    E = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    c = (S * np.diag(E)).sum() / (s**2).sum() * len(src) if scale else 1.0
    t = mu_d - c * R @ mu_s
    return c, R, t


def align(est, truth, scale):
    """Similarity taking est poses onto truth: rotation from the orientations (chordal mean, well
    posed even when the camera barely moves), then scale and translation from the positions."""
    M = np.einsum("nij,nkj->ik", truth[:, :3, :3], est[:, :3, :3])
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    pe = est[:, :3, 3] @ R.T
    pt = truth[:, :3, 3]
    de, dt = pe - pe.mean(0), pt - pt.mean(0)
    c = float((de * dt).sum() / max((de * de).sum(), 1e-12)) if scale else 1.0
    t = pt.mean(0) - c * pe.mean(0)
    return c, R, t


def angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def robot_poses(rec, frames):
    times, poses, _, _ = rec.motion
    rot = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))
    out = []
    for f in frames:
        t = np.clip(f["exposed"], times[0], times[-1])
        m = np.eye(4)
        m[:3, :3] = rot([t]).as_matrix()[0]
        m[:3, 3] = [np.interp(t, times, poses[:, i, 3]) for i in range(3)]
        out.append(m)
    return np.array(out)


def evaluate(name, est, truth, times, scale=False):
    c, R, t = align(est, truth, scale)
    aligned = est.copy()
    aligned[:, :3, :3] = R @ est[:, :3, :3]
    aligned[:, :3, 3] = (c * (R @ est[:, :3, 3].T)).T + t
    pos = np.linalg.norm(aligned[:, :3, 3] - truth[:, :3, 3], axis=1)
    rot = angle(np.einsum("nji,njk->nik", truth[:, :3, :3], aligned[:, :3, :3]))
    line = (
        f"{name:28s} n={len(est):5d} scale={c:.3f}  abs: pos rms {100 * np.sqrt((pos**2).mean()):5.1f} cm"
        f" rot rms {np.sqrt((rot**2).mean()):5.2f} deg"
    )
    for dt in (0.1, 1.0, 5.0):
        j = np.searchsorted(times, times + dt)
        ok = j < len(times)
        i, j = np.flatnonzero(ok), j[ok]
        if not len(i):
            continue
        rel_e = np.linalg.inv(aligned[i]) @ aligned[j]
        rel_t = np.linalg.inv(truth[i]) @ truth[j]
        err = np.linalg.inv(rel_t) @ rel_e
        line += f" | d{dt:g}s {np.median(angle(err[:, :3, :3])):.2f}deg {100 * np.median(np.linalg.norm(err[:, :3, 3], axis=1)):.1f}cm"
    # co-visible pairs
    axes = -truth[:, :3, 2]
    centres = truth[:, :3, 3]
    rng = np.random.default_rng(0)
    a = rng.integers(0, len(est), 4000)
    b = rng.integers(0, len(est), 4000)
    keep = (np.einsum("ij,ij->i", axes[a], axes[b]) > np.cos(np.radians(30))) & (
        np.linalg.norm(centres[a] - centres[b], axis=1) < 1.0
    ) & (np.abs(times[a] - times[b]) > 2.0)
    a, b = a[keep], b[keep]
    if len(a):
        err = np.linalg.inv(np.linalg.inv(truth[a]) @ truth[b]) @ (
            np.linalg.inv(aligned[a]) @ aligned[b]
        )
        r, p = angle(err[:, :3, :3]), np.linalg.norm(err[:, :3, 3], axis=1)
        line += (
            f" | covisible(>2s apart, n={len(a)}): rot p50 {np.median(r):.2f} p90 "
            f"{np.percentile(r, 90):.2f} deg, pos p50 {100 * np.median(p):.1f} cm"
        )
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("poses", nargs="*")
    ap.add_argument("--robot", action="store_true")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--sim3", action="store_true")
    a = ap.parse_args()
    rec = Recording(a.rec)
    frames = rec.truth_frames()
    by_seq = {f["seq"]: f for f in frames}
    if a.robot:
        sel = frames[::3]
        evaluate(
            "robot kinematics",
            robot_poses(rec, sel),
            np.array([f["pose"] for f in sel]),
            np.array([f["exposed"] for f in sel]),
        )
    labels = a.labels or [Path(p).parent.name + "/" + Path(p).stem for p in a.poses]
    for path, label in zip(a.poses, labels, strict=True):
        data = np.load(path)
        keep = [i for i, s in enumerate(data["seq"]) if int(s) in by_seq]
        est = data["poses"][keep]
        sel = [by_seq[int(data["seq"][i])] for i in keep]
        evaluate(
            label,
            est,
            np.array([f["pose"] for f in sel]),
            np.array([f["exposed"] for f in sel]),
            scale=a.sim3,
        )


if __name__ == "__main__":
    main()
