"""DROID-SLAM, unchanged, on an export.py folder (run inside the droid container, repo at /opt/droid-slam):

    python run_droid.py EXPORT OUT [--stride 1] [--weights /opt/droid-slam/droid.pth] [droid args]

Writes OUT/traj.txt (TUM: seq tx ty tz qx qy qz qw, camera-to-world, OpenCV axes, DROID's world),
OUT/keyframes.pt (keyframe seqs, poses, depth) and OUT/stats.json (seconds, peak VRAM)."""

import argparse
import json
import os
import sys
import time

sys.path.append("/opt/droid-slam/droid_slam")
sys.path.append("/opt/droid-slam")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from demo import image_stream  # noqa: E402
from droid import Droid  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("export")
    p.add_argument("out")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--weights", default="/opt/droid-slam/droid.pth")
    p.add_argument("--buffer", type=int, default=512)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--filter_thresh", type=float, default=2.4)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--keyframe_thresh", type=float, default=4.0)
    p.add_argument("--frontend_thresh", type=float, default=16.0)
    p.add_argument("--frontend_window", type=int, default=25)
    p.add_argument("--frontend_radius", type=int, default=2)
    p.add_argument("--frontend_nms", type=int, default=1)
    p.add_argument("--backend_thresh", type=float, default=22.0)
    p.add_argument("--backend_radius", type=int, default=2)
    p.add_argument("--backend_nms", type=int, default=3)
    p.add_argument("--upsample", action="store_true")
    p.add_argument("--frontend_device", default="cuda")
    p.add_argument("--backend_device", default="cuda")
    args = p.parse_args()
    args.stereo = False
    args.disable_vis = True
    args.asynchronous = False
    torch.multiprocessing.set_start_method("spawn")
    os.makedirs(args.out, exist_ok=True)
    imagedir = os.path.join(args.export, "images")
    calib = os.path.join(args.export, "calib.txt")
    names = sorted(os.listdir(imagedir))[:: args.stride]
    seqs = [int(n.split(".")[0]) for n in names]
    droid = None
    began = time.time()
    for t, image, intrinsics in image_stream(imagedir, calib, args.stride):
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        droid.track(t, image, intrinsics=intrinsics)
    tracked = time.time() - began
    traj = droid.terminate(image_stream(imagedir, calib, args.stride))
    total = time.time() - began
    with open(os.path.join(args.out, "traj.txt"), "w") as f:
        for seq, row in zip(seqs, traj, strict=True):
            f.write(f"{seq} " + " ".join(f"{v:.8f}" for v in row) + "\n")
    video = droid.video
    n = video.counter.value
    torch.save(
        dict(
            seq=torch.tensor([seqs[int(i)] for i in video.tstamp[:n].cpu().numpy()]),
            poses=video.poses[:n].cpu(),
            disps=video.disps[:n].cpu(),
            intrinsics=video.intrinsics[:n].cpu(),
        ),
        os.path.join(args.out, "keyframes.pt"),
    )
    json.dump(
        dict(
            frames=len(seqs),
            keyframes=int(n),
            track_seconds=tracked,
            total_seconds=total,
            fps=len(seqs) / tracked,
            peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
        ),
        open(os.path.join(args.out, "stats.json"), "w"),
        indent=1,
    )
    print("done", n, "keyframes", f"{len(seqs) / tracked:.1f} fps")


if __name__ == "__main__":
    main()
