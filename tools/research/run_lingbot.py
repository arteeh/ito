"""LingBot-Map (robbyant/lingbot-map b9dd1b1), unchanged, on an export.py folder:

    LINGBOT=~/Projects/ito/research/systems/lingbot-map
    $LINGBOT/.venv/bin/python run_lingbot.py EXPORT OUT --model_path CKPT [--stride 1] [--use_sdpa]
        [demo.py's other inference options]

Runs demo.py's own loading and streaming inference, then writes OUT/traj.txt (index tx ty tz qx qy
qz qw: camera-to-world, OpenCV axes, its own world and scale; index into the export's frames),
OUT/depth.npy (N x H x W float16, its depth), OUT/intrinsics.npy, OUT/stats.json."""

import json
import os
import sys
import time

LINGBOT = os.environ.get("LINGBOT", os.path.expanduser("~/Projects/ito/research/systems/lingbot-map"))
sys.path.insert(0, LINGBOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


def main():
    import demo

    export, out = sys.argv[1], sys.argv[2]
    sys.argv = [sys.argv[0], "--image_folder", os.path.join(export, "images")] + sys.argv[3:]
    parser_args = []
    # Reuse demo.main's parser by parsing through a patched main: simplest is to mirror it.
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--image_folder")
    p.add_argument("--model_path", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--patch_size", type=int, default=14)
    p.add_argument("--mode", default="streaming")
    p.add_argument("--enable_3d_rope", action="store_true", default=True)
    p.add_argument("--max_frame_num", type=int, default=1024)
    p.add_argument("--num_scale_frames", type=int, default=8)
    p.add_argument("--keyframe_interval", type=int, default=None)
    p.add_argument("--kv_cache_sliding_window", type=int, default=64)
    p.add_argument("--camera_num_iterations", type=int, default=4)
    p.add_argument("--use_sdpa", action="store_true", default=False)
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--offload_to_cpu", action="store_true", default=True)
    p.add_argument("--window_size", type=int, default=64)
    p.add_argument("--overlap_size", type=int, default=16)
    p.add_argument("--overlap_keyframes", type=int, default=None)
    args, unknown = p.parse_known_args()
    os.makedirs(out, exist_ok=True)
    began = time.time()
    images, paths, _ = demo.load_images(
        image_folder=args.image_folder, video_path=None, fps=10, first_k=None, stride=args.stride,
        image_size=args.image_size, patch_size=args.patch_size, rotate_clockwise_90=False,
    )
    device = torch.device("cuda")
    model = demo.load_model(args, device)
    dtype = torch.bfloat16
    if getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)
    n = images.shape[0]
    if args.keyframe_interval is None:
        args.keyframe_interval = (n + 319) // 320 if args.mode == "streaming" and n > 320 else 1
    images = images.to(device)
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images, num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=torch.device("cpu") if args.offload_to_cpu else None,
            )
        else:
            predictions = model.inference_windowed(
                images, window_size=args.window_size, overlap_size=args.overlap_size,
                overlap_keyframes=args.overlap_keyframes, num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=torch.device("cpu") if args.offload_to_cpu else None,
            )
    seconds = time.time() - t0

    c2w, intrinsic = demo.pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    c2w = c2w.float().cpu().numpy().reshape(-1, 3, 4)
    with open(os.path.join(out, "traj.txt"), "w") as f:
        for i, m in enumerate(c2w):
            q = Rotation.from_matrix(m[:, :3]).as_quat()
            f.write(f"{i} {m[0, 3]} {m[1, 3]} {m[2, 3]} {q[0]} {q[1]} {q[2]} {q[3]}\n")
    if "depth" in predictions:
        np.save(os.path.join(out, "depth.npy"),
                predictions["depth"].float().cpu().numpy().squeeze().astype(np.float16))
    np.save(os.path.join(out, "intrinsics.npy"), intrinsic.float().cpu().numpy())
    json.dump(dict(frames=int(n), seconds=seconds, fps=n / seconds, total=time.time() - began,
                   peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                   keys=sorted(predictions.keys())),
              open(os.path.join(out, "stats.json"), "w"), indent=1)
    print("done", n, "frames", f"{n / seconds:.1f} fps")


if __name__ == "__main__":
    main()
