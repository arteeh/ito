"""Offline 3D Gaussian splatting on a driver recording (tools/research/data.py): a standard,
well-tuned gsplat trainer, to find what a given set of poses and initial points can become.

    gs.py train REC OUT.npz [--start S --end E --every K] [--size 640x360] [--poses truth|FILE]
          [--noise ROT_DEG,TRANS_CM[,drift|white]] [--init truth|FILE.npz|none] [--iters N]
          [--sh 0] [--pose-opt] [--holdout 8] [--seed 0]

Poses FILE: npz with 'seq' (frame seq numbers) and 'poses' (N x 4 x 4 world-from-camera, Ito axes);
frames without an estimate are skipped. Init FILE: npz with 'points' (N x 3 world) and 'colors'
(N x 3, 0..1). OUT: Ito splat records (N x (3 + SH) x 4: [xyz, opacity], [scale, 0], wxyz,
[SH rgb, 0]...), sh degree, the poses trained with (after pose optimisation) and the run's log."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data import AXES, Recording  # noqa: E402

C0 = 0.28209479177387814


def ssim(a, b):
    """Mean SSIM of (B, 3, H, W) images, 11-tap Gaussian window."""
    x = torch.arange(11, device=a.device, dtype=a.dtype) - 5
    g = torch.exp(-(x**2) / (2 * 1.5**2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()

    def blur(v):
        return torch.nn.functional.conv2d(v, window, padding=5, groups=3)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return value.mean()


def rotvec_matrix(w):
    """(N, 3) rotation vectors -> (N, 3, 3), differentiable."""
    theta = w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / theta
    K = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    return eye + s * K + (1 - c) * (K @ K)


def perturb(poses, rot_deg, trans_cm, mode, seed):
    """Poses with rotation (deg) and translation (cm) error of that rms per axis. 'white': each
    frame its own error; 'drift': a smooth random walk (1 s correlation at 10 fps)."""
    rng = np.random.default_rng(seed)
    n = len(poses)
    if mode == "white":
        r = rng.normal(0, 1, (n, 3))
        t = rng.normal(0, 1, (n, 3))
    else:
        r = np.cumsum(rng.normal(0, 1, (n, 3)), 0)
        t = np.cumsum(rng.normal(0, 1, (n, 3)), 0)
        k = np.exp(-np.arange(-30, 31) ** 2 / (2 * 10.0**2))
        k /= k.sum()
        r = np.stack([np.convolve(r[:, i] - r[:, i].mean(), k, "same") for i in range(3)], 1)
        t = np.stack([np.convolve(t[:, i] - t[:, i].mean(), k, "same") for i in range(3)], 1)
        r /= r.std(0, keepdims=True) + 1e-12
        t /= t.std(0, keepdims=True) + 1e-12
    from scipy.spatial.transform import Rotation

    out = poses.copy()
    for i in range(n):
        R = Rotation.from_rotvec(np.radians(rot_deg) * r[i]).as_matrix()
        out[i, :3, :3] = poses[i, :3, :3] @ R
        out[i, :3, 3] = poses[i, :3, 3] + trans_cm / 100 * t[i]
    return out


def load_frames(rec, args):
    # Frames with exact depth (every 3rd the sim rendered), then every K-th of those: the same
    # frames serve training and, with their own (possibly perturbed) poses, initialisation.
    frames = [f for f in rec.between(args.start, args.end) if f.get("depth")][:: args.every]
    poses = np.array([f["pose"] for f in frames])
    if args.poses != "truth":
        given = np.load(args.poses)
        lookup = {int(s): p for s, p in zip(given["seq"], given["poses"], strict=True)}
        keep = [i for i, f in enumerate(frames) if f["seq"] in lookup]
        frames = [frames[i] for i in keep]
        poses = np.array([lookup[f["seq"]] for f in frames])
    if args.noise:
        parts = args.noise.split(",")
        poses = perturb(
            poses, float(parts[0]), float(parts[1]), parts[2] if len(parts) > 2 else "white",
            args.seed,
        )
    return frames, poses


def init_points(rec, frames, poses, args, size):
    if args.init == "none":
        return None
    if args.init != "truth":
        data = np.load(args.init)
        return data["points"].astype(np.float32), data["colors"].astype(np.float32)
    K = rec.K_at(size)
    w, h = size
    u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    points, colors = [], []
    deep = [(f, p) for f, p in zip(frames, poses, strict=True) if f.get("depth")]
    step = max(1, len(deep) // 60)
    for f, pose in deep[::step]:
        d = rec.depth(f, size)
        rgb = rec.rgb(f, size).astype(np.float32) / 255
        ok = (d > 0.05) & (d < 8)
        ok[::2] = False
        ok[:, ::2] = False
        p = rays[ok] * d[ok, None]
        c2w = pose @ AXES
        points.append(p @ c2w[:3, :3].T + c2w[:3, 3])
        colors.append(rgb[ok])
    points, colors = np.concatenate(points), np.concatenate(colors)
    # 1 cm voxels: one point each; at most args.init_cap of them (VRAM).
    keys = np.floor(points / 0.01).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    if len(first) > args.init_cap:
        first = np.random.default_rng(0).choice(first, args.init_cap, replace=False)
    return points[first].astype(np.float32), colors[first].astype(np.float32)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    rec = Recording(args.rec)
    size = tuple(int(v) for v in args.size.split("x"))
    w, h = size
    frames, poses = load_frames(rec, args)
    held = np.zeros(len(frames), bool)
    if args.holdout:
        held[args.holdout // 2 :: args.holdout] = True
    log = dict(args=vars(args), frames=len(frames), held=int(held.sum()))
    began = time.time()
    images = torch.stack(
        [torch.from_numpy(rec.rgb(f, size)) for f in frames]
    ).cuda()  # uint8 (N, H, W, 3)
    depths = None
    if args.depth_loss:
        if args.depth_source == "truth":
            depths = torch.stack(
                [torch.from_numpy(rec.depth(f, size)) for f in frames]
            ).cuda().half()
        else:
            import cv2

            dumped = {}
            for path in Path(args.depth_source).glob("*.npz"):
                dumped[int(path.stem) - 1] = cv2.resize(
                    np.load(path)["estimated"], size, interpolation=cv2.INTER_NEAREST
                )
            depths = torch.stack(
                [
                    torch.from_numpy(dumped.get(f["seq"], np.zeros((h, w), np.float32)))
                    for f in frames
                ]
            ).cuda().half()
    K = torch.tensor(rec.K_at(size), dtype=torch.float32, device="cuda")
    c2w = torch.tensor(np.array([p @ AXES for p in poses]), dtype=torch.float32, device="cuda")
    found = init_points(rec, frames, poses, args, size)
    if found is None:
        centre = c2w[:, :3, 3].mean(0)
        points = (torch.rand(100_000, 3, device="cuda") - 0.5) * 6 + centre
        colors = torch.rand(100_000, 3, device="cuda")
    else:
        points = torch.tensor(found[0], device="cuda")
        colors = torch.tensor(found[1], device="cuda")
    n = len(points)
    log["init_points"] = n
    # Initial scale: mean distance to the 3 nearest neighbours (chunked).
    with torch.no_grad():
        dist = torch.empty(n, device="cuda")
        for i in range(0, n, 4096):
            d = torch.cdist(points[i : i + 4096], points)
            d = d.topk(4, largest=False).values[:, 1:]
            dist[i : i + 4096] = d.mean(-1)
    scene_scale = float((c2w[:, :3, 3] - c2w[:, :3, 3].mean(0)).norm(dim=-1).max()) * 1.1
    scene_scale = max(scene_scale, 1.0)
    sh = args.sh
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(points),
            "scales": torch.nn.Parameter(torch.log(dist.clamp(1e-4, 0.1))[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(
                torch.tensor([1.0, 0, 0, 0], device="cuda").repeat(n, 1)
            ),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.5, device="cuda"))),
            "sh0": torch.nn.Parameter(((colors - 0.5) / C0)[:, None, :]),
            "shN": torch.nn.Parameter(torch.zeros(n, (sh + 1) ** 2 - 1, 3, device="cuda")),
        }
    )
    rates = dict(
        means=1.6e-4 * scene_scale,
        scales=5e-3,
        quats=1e-3,
        opacities=5e-2,
        sh0=2.5e-3,
        shN=2.5e-3 / 20,
    )
    optimizers = {
        k: torch.optim.Adam([{"params": params[k], "lr": rates[k], "name": k}], eps=1e-15)
        for k in params
    }
    decay = 0.01 ** (1 / args.iters)
    strategy = DefaultStrategy(
        refine_start_iter=500,
        refine_stop_iter=int(args.iters * 0.6),
        refine_every=100,
        reset_every=3000,
        grow_grad2d=args.grow,
        verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)
    pose_delta = None
    if args.pose_opt:
        pose_delta = torch.nn.Parameter(torch.zeros(len(frames), 6, device="cuda"))
        pose_optimizer = torch.optim.Adam([pose_delta], lr=args.pose_lr)
    train_ids = np.flatnonzero(~held)
    rng = np.random.default_rng(args.seed)
    for step in range(args.iters):
        i = int(rng.choice(train_ids))
        cam = c2w[i]
        if pose_delta is not None:
            R = rotvec_matrix(pose_delta[i : i + 1, :3])[0]
            delta = torch.eye(4, device="cuda")
            delta = torch.cat(
                (torch.cat((R, pose_delta[i, 3:, None]), 1), delta[3:]), 0
            )
            cam = cam @ delta
        viewmat = torch.linalg.inv(cam)[None]
        degree = min(sh, step // 1000)
        colours = torch.cat((params["sh0"], params["shN"]), 1)
        rendered, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(),
            colors=colours,
            viewmats=viewmat,
            Ks=K[None],
            width=w,
            height=h,
            sh_degree=degree,
            near_plane=0.01,
            far_plane=100,
            packed=False,
            render_mode="RGB+ED" if depths is not None else "RGB",
        )
        if depths is not None:
            drawn_depth = rendered[..., 3]
            rendered = rendered[..., :3]
        target = images[i].float()[None] / 255
        l1 = (rendered - target).abs().mean()
        loss = 0.8 * l1 + 0.2 * (
            1 - ssim(rendered.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2))
        )
        if depths is not None:
            known = depths[i].float()[None]
            mask = (known > 0) & (alpha[..., 0].detach() > 0.5)
            if mask.any():
                loss = loss + args.depth_loss * (
                    (drawn_depth[mask] - known[mask]).abs() / known[mask]
                ).mean()
        if args.flat:
            axes = params["scales"].exp().sort(-1).values
            loss = loss + args.flat * (axes[:, 0] / axes[:, 1].clamp_min(1e-6)).mean()
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if pose_delta is not None and step > args.pose_start:
            pose_optimizer.step()
        if pose_delta is not None:
            pose_optimizer.zero_grad(set_to_none=True)
        for group in optimizers["means"].param_groups:
            group["lr"] *= decay
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if len(params["means"]) > args.max_splats and strategy.refine_stop_iter > step:
            # VRAM: 12 GB card, ~5 GB held by Windows; stop densifying past the cap.
            strategy.refine_stop_iter = step
            log["capped_at_step"] = step
        if step % 1000 == 0 or step == args.iters - 1:
            print(f"step {step} loss {loss.item():.4f} splats {len(params['means'])}", flush=True)
    log["train_seconds"] = time.time() - began
    log["splats"] = len(params["means"])
    # Held-out PSNR (supporting evidence only).
    psnrs = []
    with torch.no_grad():
        colours = torch.cat((params["sh0"], params["shN"]), 1)
        for i in np.flatnonzero(held):
            cam = c2w[i]
            if pose_delta is not None:
                cam = cam  # held-out views keep their given pose
            rendered, _, _ = rasterization(
                params["means"], params["quats"], params["scales"].exp(),
                params["opacities"].sigmoid(), colours, torch.linalg.inv(cam)[None], K[None],
                w, h, sh_degree=sh, near_plane=0.01, far_plane=100,
            )
            mse = ((rendered[0] - images[i].float() / 255) ** 2).mean()
            psnrs.append(float(-10 * torch.log10(mse)))
    log["holdout_psnr"] = float(np.mean(psnrs)) if psnrs else None
    print(json.dumps({k: v for k, v in log.items() if k != "args"}), flush=True)
    final = c2w.detach().clone()
    if pose_delta is not None:
        with torch.no_grad():
            R = rotvec_matrix(pose_delta[:, :3])
            d = torch.eye(4, device="cuda").repeat(len(frames), 1, 1)
            d[:, :3, :3] = R
            d[:, :3, 3] = pose_delta[:, 3:]
            final = c2w @ d
    final = final.cpu().numpy() @ AXES  # back to Ito axes (AXES is its own inverse)
    save(args.out, params, sh, final, frames, log)


def save(path, params, sh, poses, frames, log):
    with torch.no_grad():
        n = len(params["means"])
        coefficients = (sh + 1) ** 2
        records = np.zeros((n, 3 + coefficients, 4), np.float32)
        records[:, 0, :3] = params["means"].cpu().numpy()
        records[:, 0, 3] = params["opacities"].sigmoid().cpu().numpy()
        records[:, 1, :3] = params["scales"].exp().cpu().numpy()
        q = torch.nn.functional.normalize(params["quats"], dim=-1)
        records[:, 2] = q.cpu().numpy()
        records[:, 3, :3] = params["sh0"][:, 0].cpu().numpy()
        if coefficients > 1:
            records[:, 4:, :3] = params["shN"].cpu().numpy()
    np.savez(
        path,
        records=records,
        sh=sh,
        poses=poses,
        seq=np.array([f["seq"] for f in frames]),
        log=json.dumps(log),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("rec")
    t.add_argument("out")
    t.add_argument("--start", type=float)
    t.add_argument("--end", type=float)
    t.add_argument("--every", type=int, default=1)
    t.add_argument("--size", default="640x360")
    t.add_argument("--poses", default="truth")
    t.add_argument("--noise")
    t.add_argument("--init", default="truth")
    t.add_argument("--iters", type=int, default=15000)
    t.add_argument("--sh", type=int, default=0)
    t.add_argument("--grow", type=float, default=0.0002)
    t.add_argument("--pose-opt", action="store_true")
    t.add_argument("--depth-loss", type=float, default=0.0)
    t.add_argument("--depth-source", default="truth", help="truth or an ITO_DUMP folder")
    t.add_argument("--flat", type=float, default=0.0)
    t.add_argument("--max-splats", type=int, default=700_000)
    t.add_argument("--init-cap", type=int, default=400_000)
    t.add_argument("--pose-lr", type=float, default=1e-4)
    t.add_argument("--pose-start", type=int, default=500)
    t.add_argument("--holdout", type=int, default=8)
    t.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.command == "train":
        train(args)


if __name__ == "__main__":
    main()
