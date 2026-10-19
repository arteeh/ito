"""A live-style Gaussian mapper on a recording, in stream time (E4): keyframes arrive as the robot
saw them; the mapper may spend only RATE optimisation steps per second of stream time.

    stream.py REC OUT.npz [--rate 60] [--keyframe 0.5] [--poses truth|FILE] [--noise R,T[,drift]]
        [--depth truth|DUMP] [--pose-opt] [--recent 8] [--size 640x360] [--snapshots 30]

Each keyframe (every --keyframe seconds of stream time) seeds Gaussians from its depth where the
map, rendered from its pose, is not yet opaque (alpha < 0.5), on a 2-pixel lattice. Steps sample
keyframes half from the --recent newest, half uniformly from all. --pose-opt refines every
keyframe's pose photometrically against the map (MonoGS/3DGS-style joint optimisation; the first
keyframe holds the gauge). Depth DUMP: an ITO_DUMP folder of Ito's per-frame depth (estimated)."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data import AXES, Recording  # noqa: E402
from gs import C0, perturb, rotvec_matrix, save, ssim  # noqa: E402


def dump_depths(folder, size):
    import cv2

    out = {}
    for path in sorted(Path(folder).glob("*.npz")):
        d = np.load(path)
        depth = d["estimated"]
        out[int(path.stem) - 1] = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("rec")
    p.add_argument("out")
    p.add_argument("--rate", type=float, default=60)
    p.add_argument("--keyframe", type=float, default=0.5)
    p.add_argument("--poses", default="truth")
    p.add_argument("--noise")
    p.add_argument("--depth", default="truth")
    p.add_argument("--pose-opt", action="store_true")
    p.add_argument("--pose-lr", type=float, default=2e-4)
    p.add_argument("--recent", type=int, default=8)
    p.add_argument("--size", default="640x360")
    p.add_argument("--snapshots", type=float, default=0)
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    from gsplat import rasterization

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    rec = Recording(a.rec)
    size = tuple(int(v) for v in a.size.split("x"))
    w, h = size
    dumped_seqs = (
        {int(q.stem) - 1 for q in Path(a.depth).glob("*.npz")} if a.depth != "truth" else None
    )
    # Exact depth: frames the sim rendered depth for; Ito's depth: the frames Ito mapped.
    frames = [
        f
        for f in rec.between(a.start, a.end)
        if (f.get("depth") if dumped_seqs is None else f["seq"] in dumped_seqs)
    ]
    if a.poses != "truth":
        # Given poses (any frames): interpolated in time to the frames with exact depth.
        from scipy.spatial.transform import Rotation, Slerp

        given = np.load(a.poses)
        received = {f["seq"]: f["received"] for f in rec.frames}
        order = np.argsort([received[int(q)] for q in given["seq"]])
        gt = np.array([received[int(given["seq"][i])] for i in order])
        gp = np.asarray(given["poses"])[order]
        slerp = Slerp(gt, Rotation.from_matrix(gp[:, :3, :3]))
        frames = [f for f in frames if gt[0] <= f["received"] <= gt[-1]]
        poses = np.tile(np.eye(4), (len(frames), 1, 1))
        times = np.array([f["received"] for f in frames])
        poses[:, :3, :3] = slerp(times).as_matrix()
        for k in range(3):
            poses[:, k, 3] = np.interp(times, gt, gp[:, k, 3])
    else:
        poses = np.array([f["pose"] for f in frames])
    if a.noise:
        parts = a.noise.split(",")
        poses = perturb(poses, float(parts[0]), float(parts[1]),
                        parts[2] if len(parts) > 2 else "white", a.seed)
    dumped = dump_depths(a.depth, size) if a.depth != "truth" else None
    t0 = frames[0]["received"]
    # Keyframes: one per --keyframe seconds of stream time.
    chosen, last = [], -1e9
    for i, f in enumerate(frames):
        if dumped is not None and f["seq"] not in dumped:
            continue
        if f["received"] - last >= a.keyframe:
            chosen.append(i)
            last = f["received"]
    K = torch.tensor(rec.K_at(size), dtype=torch.float32, device="cuda")
    Kn = rec.K_at(size)
    u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
    rays = np.stack([(u - Kn[0, 2]) / Kn[0, 0], (v - Kn[1, 2]) / Kn[1, 1], np.ones_like(u)], -1)
    lattice = np.zeros((h, w), bool)
    lattice[::2, ::2] = True
    params = None
    optimizers = None
    keyframes = []  # dicts: image tensor, c2w tensor, delta param index
    deltas = torch.nn.Parameter(torch.zeros(len(chosen), 6, device="cuda"))
    pose_opt = torch.optim.Adam([deltas], lr=a.pose_lr)
    names = ("means", "scales", "quats", "opacities", "sh0")
    rates = dict(means=1.6e-4 * 2.0, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3)
    steps = 0
    log = dict(args=vars(a), keyframes=len(chosen))
    budget = 0.0
    snapshots = []
    next_snapshot = a.snapshots or 1e9
    began = time.time()

    def camera(k):
        c2w = keyframes[k]["c2w"]
        if a.pose_opt and k > 0:
            R = rotvec_matrix(deltas[k : k + 1, :3])[0]
            top = torch.cat((R, deltas[k, 3:, None]), 1)
            delta = torch.cat((top, torch.tensor([[0.0, 0, 0, 1]], device="cuda")), 0)
            c2w = c2w @ delta
        return c2w

    def render(c2w, grad=True):
        with torch.set_grad_enabled(grad):
            return rasterization(
                params["means"], params["quats"], params["scales"].exp(),
                params["opacities"].sigmoid(), params["sh0"], torch.linalg.inv(c2w)[None], K[None],
                w, h, sh_degree=0, near_plane=0.01, far_plane=100, packed=False, absgrad=True,
            )

    def add(points, colors, spacing):
        nonlocal params, optimizers
        n = len(points)
        new = dict(
            means=torch.tensor(points, device="cuda", dtype=torch.float32),
            scales=torch.log(torch.tensor(spacing, device="cuda", dtype=torch.float32))[:, None]
            .repeat(1, 3),
            quats=torch.tensor([1.0, 0, 0, 0], device="cuda").repeat(n, 1),
            opacities=torch.logit(torch.full((n,), 0.7, device="cuda")),
            sh0=((torch.tensor(colors, device="cuda", dtype=torch.float32) - 0.5) / C0)[:, None],
        )
        if params is None:
            params = {k: torch.nn.Parameter(new[k]) for k in names}
            optimizers = {
                k: torch.optim.Adam([params[k]], lr=rates[k], eps=1e-15) for k in names
            }
            return
        for k in names:
            old = params[k]
            joined = torch.nn.Parameter(torch.cat((old.data, new[k])))
            opt = optimizers[k]
            state = opt.state.pop(old, None)
            opt.param_groups[0]["params"] = [joined]
            if state is not None:
                state["exp_avg"] = torch.cat((state["exp_avg"], torch.zeros_like(new[k])))
                state["exp_avg_sq"] = torch.cat((state["exp_avg_sq"], torch.zeros_like(new[k])))
                opt.state[joined] = state
            params[k] = joined

    def prune(keep):
        nonlocal params
        for k in names:
            old = params[k]
            kept = torch.nn.Parameter(old.data[keep])
            opt = optimizers[k]
            state = opt.state.pop(old, None)
            opt.param_groups[0]["params"] = [kept]
            if state is not None:
                state["exp_avg"] = state["exp_avg"][keep]
                state["exp_avg_sq"] = state["exp_avg_sq"][keep]
                opt.state[kept] = state
            params[k] = kept

    for n, index in enumerate(chosen):
        f = frames[index]
        stream_time = f["received"] - t0
        if keyframes:
            budget += a.rate * (stream_time - keyframes[-1]["time"])
        image = torch.from_numpy(rec.rgb(f, size)).cuda()
        c2w = torch.tensor(poses[index] @ AXES, dtype=torch.float32, device="cuda")
        keyframes.append(dict(image=image, c2w=c2w, time=stream_time, seq=f["seq"]))
        # Seed where the map is not yet opaque, from this keyframe's (possibly refined) pose.
        depth = rec.depth(f, size) if dumped is None else dumped[f["seq"]]
        ok = (depth > 0.05) & (depth < 8) & lattice
        if params is not None:
            with torch.no_grad():
                _, alpha, _ = render(camera(n).detach(), grad=False)
            ok &= alpha[0, ..., 0].cpu().numpy() < 0.5
        if ok.any():
            cam = camera(n).detach().cpu().numpy()
            local = rays[ok] * depth[ok, None]
            world = local @ cam[:3, :3].T + cam[:3, 3]
            rgb = rec.rgb(f, size).astype(np.float32)[ok] / 255
            add(world, rgb, depth[ok] / Kn[0, 0] * 1.5)
        # Spend the budget earned since the last keyframe.
        while budget >= 1:
            budget -= 1
            recent = len(keyframes) - 1 - int(rng.integers(0, min(a.recent, len(keyframes))))
            k = recent if rng.random() < 0.5 else int(rng.integers(0, len(keyframes)))
            rendered, alpha, info = render(camera(k))
            info["means2d"].retain_grad()
            target = keyframes[k]["image"].float()[None] / 255
            loss = 0.8 * (rendered - target).abs().mean() + 0.2 * (
                1 - ssim(rendered.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2))
            )
            loss.backward()
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            if a.pose_opt:
                pose_opt.step()
                pose_opt.zero_grad(set_to_none=True)
            steps += 1
            if steps % 200 == 0:
                with torch.no_grad():
                    keep = params["opacities"].sigmoid() > 0.02
                    if (~keep).any():
                        prune(keep)
        if a.snapshots and stream_time >= next_snapshot:
            next_snapshot += a.snapshots
            snapshots.append(stream_time)
            save_stream(a.out.replace(".npz", f"-{stream_time:05.0f}.npz"), params, keyframes,
                        camera, log)
    log["steps"] = steps
    log["seconds"] = time.time() - began
    log["splats"] = int(len(params["means"]))
    log["snapshots"] = snapshots
    print(json.dumps({k: v for k, v in log.items() if k != "args"}))
    save_stream(a.out, params, keyframes, camera, log)


def save_stream(path, params, keyframes, camera, log):
    full = dict(params)
    full["shN"] = torch.zeros(len(params["means"]), 0, 3, device="cuda")
    with torch.no_grad():
        poses = np.array([camera(k).detach().cpu().numpy() @ AXES for k in range(len(keyframes))])
    save(path, full, 0, poses, keyframes, log)


if __name__ == "__main__":
    main()
