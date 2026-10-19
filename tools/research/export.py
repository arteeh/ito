"""Export a recording for outside SLAM systems, and import their trajectories back.

    export.py frames REC OUT [--size 640x360] [--every 2] [--start S] [--end E]
        OUT/images/NNNNNN.png (rectified pinhole frames, NNNNNN = frame seq), OUT/rgb.txt and
        OUT/times.txt (seq received-time), OUT/calib.txt (fx fy cx cy), OUT/intrinsics.yaml,
        OUT/groundtruth.txt (TUM: time tx ty tz qx qy qz qw, OpenCV camera axes, Ito world)
    export.py import REC TRAJ OUT.npz [--format tum|kitti] [--times OUT/times.txt] [--no-align]
        TRAJ: camera-to-world, OpenCV camera axes, any world/scale; rows matched to frames by time
        (TUM, the received time or the seq) or by order (kitti). Written as seq + Ito-axes poses,
        Sim3-aligned to the exact trajectory (one global similarity: placement only, no change
        to the trajectory's own consistency) so maps land in the room's frame."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from data import AXES, Recording  # noqa: E402
from traj import align  # noqa: E402


def frames(a):
    rec = Recording(a.rec)
    size = tuple(int(v) for v in a.size.split("x"))
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    chosen = rec.between(a.start, a.end, a.every)
    K = rec.K_at(size)
    with (out / "times.txt").open("w") as times, (out / "rgb.txt").open("w") as rgb, (
        out / "groundtruth.txt"
    ).open("w") as gt:
        for f in chosen:
            name = f"{f['seq']:06d}.png"
            cv2.imwrite(str(out / "images" / name), rec.rgb(f, size)[..., ::-1])
            times.write(f"{f['seq']} {f['received']:.6f}\n")
            rgb.write(f"{f['received']:.6f} images/{name}\n")
            c2w = f["pose"] @ AXES
            q = Rotation.from_matrix(c2w[:3, :3]).as_quat()
            gt.write(
                f"{f['received']:.6f} {c2w[0, 3]:.6f} {c2w[1, 3]:.6f} {c2w[2, 3]:.6f} "
                f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n"
            )
    if a.depth:
        # TUM-style exact depth (uint16, metres x 5000) of every frame with depth in the range,
        # for systems whose loaders want depth beside the images (evaluation only).
        (out / "depth").mkdir(exist_ok=True)
        with (out / "depth.txt").open("w") as listing:
            for f in rec.between(a.start, a.end):
                if not f.get("depth"):
                    continue
                d = rec.depth(f, size)
                name = f"{f['seq']:06d}.png"
                cv2.imwrite(str(out / "depth" / name), np.clip(d * 5000, 0, 65535).astype(np.uint16))
                listing.write(f"{f['received']:.6f} depth/{name}\n")
    (out / "calib.txt").write_text(f"{K[0, 0]:.6f} {K[1, 1]:.6f} {K[0, 2]:.6f} {K[1, 2]:.6f}\n")
    (out / "intrinsics.yaml").write_text(
        f"width: {size[0]}\nheight: {size[1]}\n"
        f"calibration: [{K[0, 0]:.6f}, {K[1, 1]:.6f}, {K[0, 2]:.6f}, {K[1, 2]:.6f}]\n"
    )
    print(len(chosen), "frames to", out)


def load_traj(path, fmt):
    rows = np.loadtxt(path)
    if fmt == "kitti":
        poses = np.tile(np.eye(4), (len(rows), 1, 1))
        poses[:, :3, :4] = rows.reshape(-1, 3, 4)
        return None, poses
    t = rows[:, 0]
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(rows[:, 4:8]).as_matrix()
    poses[:, :3, 3] = rows[:, 1:4]
    return t, poses


def import_traj(a):
    rec = Recording(a.rec)
    by_seq = {f["seq"]: f for f in rec.truth_frames()}
    t, poses = load_traj(a.traj, "tum" if a.format == "index" else a.format)
    listing = np.loadtxt(a.times) if a.times else None
    seqs = []
    if a.format == "index":
        # First column: index into the export's frames (times.txt order = sorted image names).
        seqs = listing[t.astype(int), 0].astype(int)
    elif t is None:
        seqs = listing[: len(poses), 0].astype(int)
    else:
        for value in t:
            if listing is not None and np.min(np.abs(listing[:, 1] - value)) < 1e-3:
                seqs.append(int(listing[np.argmin(np.abs(listing[:, 1] - value)), 0]))
            elif listing is not None and np.min(np.abs(listing[:, 0] - value)) < 0.5:
                seqs.append(int(listing[np.argmin(np.abs(listing[:, 0] - value)), 0]))
            else:
                seqs.append(int(round(value)))
        seqs = np.array(seqs)
    ito = poses @ AXES
    keep = [i for i, s in enumerate(seqs) if s in by_seq]
    ito, seqs = ito[keep], np.asarray(seqs)[keep]
    c, R, tt = 1.0, np.eye(3), np.zeros(3)
    if not a.no_align:
        truth = np.array([by_seq[s]["pose"] for s in seqs])
        c, R, tt = align(ito, truth, True)
        ito[:, :3, :3] = R @ ito[:, :3, :3]
        ito[:, :3, 3] = (c * (R @ ito[:, :3, 3].T)).T + tt
        print(f"aligned: scale {c:.4f}")
    # The placement (Ito world = c R x + t of the system's world), for its maps.
    np.savez(a.out, seq=seqs, poses=ito, scale=c, rotation=R, translation=tt)
    print(len(seqs), "poses to", a.out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("frames")
    f.add_argument("rec")
    f.add_argument("out")
    f.add_argument("--size", default="640x360")
    f.add_argument("--every", type=int, default=2)
    f.add_argument("--start", type=float)
    f.add_argument("--end", type=float)
    f.add_argument("--depth", action="store_true", help="also TUM-style exact depth")
    i = sub.add_parser("import")
    i.add_argument("rec")
    i.add_argument("traj")
    i.add_argument("out")
    i.add_argument("--format", default="tum", help="tum (time or seq), kitti, index")
    i.add_argument("--times")
    i.add_argument("--no-align", action="store_true")
    a = p.parse_args()
    frames(a) if a.command == "frames" else import_traj(a)


if __name__ == "__main__":
    main()
