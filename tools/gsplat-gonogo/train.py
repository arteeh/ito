# ruff: noqa: E402
"""Timed fixed-pose 3DGS experiment. gsplat DefaultStrategy + 3D/2D mip filtering.
Times exclude data/init preparation, CUDA warm-up and checkpoint evaluation; report all.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
sys.path.insert(0, str(ROOT / "repo/e2e"))
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import DefaultStrategy, rasterization
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

torch.set_num_threads(6)
C0 = 0.28209479177387814


def ssim(x, y):
    # Original 11x11 Gaussian-window SSIM, valid boundary, RGB averaged.
    x = x.permute(2, 0, 1)[None]
    y = y.permute(2, 0, 1)[None]
    z = torch.cat([x, y, x * x, y * y, x * y], 0)
    k = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    k = torch.exp(-k * k / (2 * 1.5**2))
    k = k / k.sum()
    z = F.conv2d(z, k.view(1, 1, 1, 11).expand(3, 1, 1, 11), groups=3)
    z = F.conv2d(z, k.view(1, 1, 11, 1).expand(3, 1, 11, 1), groups=3)
    mx, my, xx, yy, xy = z.unbind(0)
    return (
        ((2 * mx * my + 0.01**2) * (2 * (xy - mx * my) + 0.03**2))
        / ((mx * mx + my * my + 0.01**2) * (xx - mx * mx + yy - my * my + 0.03**2))
    ).mean()


# Point clouds per initialisation. "online" is Ito's own fused-mode MASt3R cloud (fused poses,
# fused scale), placed in the fused world; the others are placed with true poses.
FILES = {
    "sparse": "triangulated.npz",
    "calibrated": "calibrated.npz",
    "mast3r": "true-extracted.npz",
    "online": "online-extracted.npz",
}


class Dataset:
    def __init__(self, run, mode="true", initialization="mast3r", estimated_only=False):
        self.run = run
        self.initialization = initialization
        self.source = ROOT.parent / "rf-run/out" / run
        self.out = ROOT / "out" / run
        self.rows = [
            json.loads(s) for s in (self.source / "rec/frames.jsonl").read_text().splitlines()
        ]
        data = np.load(self.out / "data.npz")
        self.times = data["times"]
        self.times = self.times - self.times[0]
        self.truth = data["poses"]
        self.mode = mode
        self.fused = self.truth.copy()
        if (self.out / "fused-extracted.npz").exists():
            from truth import align

            e = np.load(self.out / "fused-extracted.npz")
            ids = e["ids"]
            est = e["estimates"]
            alignment = np.linalg.inv(align(est[0], self.truth[ids[0]]))
            est = alignment @ est
            t = np.clip(self.times, self.times[ids[0]], self.times[ids[-1]])
            self.fused[:, :3, :3] = Slerp(self.times[ids], Rotation.from_matrix(est[:, :3, :3]))(
                t
            ).as_matrix()
            for k in range(3):
                self.fused[:, k, 3] = np.interp(t, self.times[ids], est[:, k, 3])
        elif mode == "fused":
            raise FileNotFoundError("fused-extracted.npz")
        self.poses = (self.fused if mode == "fused" else self.truth).copy()
        intr = json.loads((self.source / "rec/camera.json").read_text())
        self.w = intr["width"]
        self.h = intr["height"]
        self.K = torch.tensor(
            [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]],
            dtype=torch.float32,
            device="cuda",
        )
        cv = self.poses @ np.diag([1, -1, -1, 1])
        self.views = torch.tensor(np.linalg.inv(cv), dtype=torch.float32, device="cuda")
        self.images = torch.from_numpy(
            np.stack(
                [
                    np.asarray(Image.open(self.source / "rec" / r["name"]).convert("RGB"))
                    for r in self.rows
                ]
            )
        ).pin_memory()
        self.test = np.arange(7, len(self.rows), 8)
        self.train = np.setdiff1d(np.arange(len(self.rows)), self.test)
        if estimated_only:
            # Only frames the fused tracker actually estimated (no pose interpolation between
            # estimates through fast head turns): what an online optimiser would get.
            estimated = np.unique(np.load(self.out / "fused-extracted.npz")["ids"])
            self.test = np.intersect1d(self.test, estimated)
            self.train = np.intersect1d(self.train, estimated)

    @staticmethod
    def point_world(initialization):
        """Which pose set a cloud was placed with."""
        return "fused" if initialization == "online" else "true"

    def to_pose_world(self, ids):
        """Per-point transform from the world a cloud was placed in to the training world."""
        source = self.truth if self.point_world(self.initialization) == "true" else self.fused
        return self.poses[ids] @ np.linalg.inv(source[ids])

    def image(self, i):
        return self.images[int(i)].to("cuda", non_blocking=True).float() / 255

    def initial(self, limit=None, maxpoints=40000):
        e = np.load(self.out / FILES[self.initialization])
        p = e["points"]
        c = e["colors"]
        ids = e["point_frames"]
        mask = np.isin(ids, self.train)
        if limit is not None:
            early = mask & (self.times[ids] <= limit)
            if early.sum() < 4:
                # Causal start: the first train frame that has points.
                early = mask & (ids == ids[mask][np.argmin(self.times[ids[mask]])])
            mask = early
        p = p[mask]
        c = c[mask]
        ids = ids[mask]
        if self.point_world(self.initialization) != self.mode:
            transform = self.to_pose_world(ids)
            p = np.einsum("nij,nj->ni", transform[:, :3, :3], p) + transform[:, :3, 3]
        # Sparse cloud, deterministic 1cm voxel dedup and cap; only train-frame MASt3R points.
        _, keep = np.unique(np.floor(p / 0.01).astype(np.int32), axis=0, return_index=True)
        if len(keep) > maxpoints:
            keep = np.random.default_rng(42).choice(keep, maxpoints, replace=False)
        return p[keep].astype(np.float32), c[keep].astype(np.float32) / 255


class Model:
    def __init__(self, data, limit=None, maxpoints=40000):
        self.data = data
        p, c = data.initial(limit, maxpoints)
        self.scene_scale = max(
            0.5,
            float(np.quantile(np.linalg.norm(p - np.median(p, axis=0), axis=1), 0.9)),
        )
        dist = cKDTree(p).query(p, k=4)[0][:, 1:]
        scale = np.clip(np.sqrt((dist**2).mean(1)), 0.001, 0.15)
        n = len(p)
        params = {
            "means": p,
            "scales": np.log(scale)[:, None].repeat(3, 1),
            "quats": np.tile([1.0, 0, 0, 0], (n, 1)),
            "opacities": np.full(n, math.log(0.1 / 0.9)),
            "sh0": ((c - 0.5) / C0)[:, None],
            "shN": np.zeros((n, 15, 3)),
        }
        self.p = torch.nn.ParameterDict(
            {
                k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, device="cuda"))
                for k, v in params.items()
            }
        )
        self.lr = {
            "means": 0.00016 * self.scene_scale,
            "scales": 0.005,
            "quats": 0.001,
            "opacities": 0.05,
            "sh0": 0.0025,
            "shN": 0.0025 / 20,
        }
        self.opt = {k: torch.optim.Adam([v], lr=self.lr[k], eps=1e-15) for k, v in self.p.items()}
        self.strategy = DefaultStrategy(
            refine_start_iter=500,
            refine_stop_iter=15000,
            refine_every=100,
            reset_every=3000,
        )
        self.state = self.strategy.initialize_state(scene_scale=self.scene_scale)
        self.strategy.check_sanity(self.p, self.opt)
        self.degree = 0
        self.filter = None
        self.filter_ids = (
            data.train if limit is None else data.train[data.times[data.train] <= limit]
        )
        self.update_filter()

    @torch.no_grad()
    def update_filter(self):
        # Mip-Splatting 3D smoothing: smallest projected footprint in a sample of training views.
        ids = self.filter_ids[
            np.linspace(0, len(self.filter_ids) - 1, min(64, len(self.filter_ids))).astype(int)
        ]
        means = self.p["means"]
        closest = torch.full((len(means),), float("inf"), device="cuda")
        for i in ids:
            v = self.data.views[i]
            local = means @ v[:3, :3].T + v[:3, 3]
            z = local[:, 2]
            u = self.data.K[0, 0] * local[:, 0] / z + self.data.K[0, 2]
            y = self.data.K[1, 1] * local[:, 1] / z + self.data.K[1, 2]
            valid = (
                (z > 0.01)
                & (u > -0.15 * self.data.w)
                & (u < 1.15 * self.data.w)
                & (y > -0.15 * self.data.h)
                & (y < 1.15 * self.data.h)
            )
            closest = torch.minimum(closest, torch.where(valid, z, float("inf")))
        valid = torch.isfinite(closest)
        fallback = closest[valid].max() if valid.any() else means.new_tensor(1.0)
        closest = torch.where(valid, closest, fallback)
        self.filter = closest / self.data.K[0, 0] * math.sqrt(0.2)

    def activated(self):
        scale2 = self.p["scales"].exp().square()
        filtered = scale2 + self.filter[:, None].square()
        opacity = self.p["opacities"].sigmoid() * torch.sqrt(scale2.prod(-1) / filtered.prod(-1))
        return filtered.sqrt(), opacity

    def render(self, i=None, views=None, K=None, w=None, h=None, mode="antialiased"):
        scales, opa = self.activated()
        if views is None:
            views = self.data.views[i : i + 1]
        if K is None:
            K = self.data.K[None].expand(len(views), -1, -1)
        return rasterization(
            means=self.p["means"],
            quats=self.p["quats"],
            scales=scales,
            opacities=opa,
            colors=torch.cat([self.p["sh0"], self.p["shN"]], 1),
            viewmats=views,
            Ks=K,
            width=w or self.data.w,
            height=h or self.data.h,
            sh_degree=self.degree,
            packed=False,
            rasterize_mode=mode,
            near_plane=0.01,
            far_plane=1000,
        )

    def step(self, i, step):
        self.degree = min(3, step // 1000)
        self.opt["means"].param_groups[0]["lr"] = self.lr["means"] * (
            0.01 ** (min(step, 30000) / 30000)
        )
        pred, _alpha, info = self.render(i)
        self.strategy.step_pre_backward(self.p, self.opt, self.state, step, info)
        target = self.data.image(i)
        pred = pred[0]
        loss = 0.8 * (pred - target).abs().mean() + 0.2 * (1 - ssim(pred, target))
        loss.backward()
        for opt in self.opt.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        self.strategy.step_post_backward(self.p, self.opt, self.state, step, info, packed=False)
        if step > 500 and step % 100 == 0 and step < 15000:
            self.update_filter()
        return float(loss.detach())

    @torch.no_grad()
    def evaluate(self, ids=None):
        if ids is None:
            ids = self.data.test
        metrics = []
        for i in ids:
            r, a, _ = self.render(int(i))
            r = r[0].clamp(0, 1)
            target = self.data.image(i)
            psnr = -10 * torch.log10((r - target).square().mean())
            quality = ssim(r, target)
            covered = a[0, :, :, 0] > 0.05
            cp = (
                -10 * torch.log10((r[covered] - target[covered]).square().mean())
                if covered.any()
                else r.new_tensor(float("nan"))
            )
            metrics.append(
                {
                    "frame": int(i),
                    "psnr": float(psnr),
                    "ssim": float(quality),
                    "coverage": float(covered.float().mean()),
                    "covered_psnr": float(cp),
                }
            )
        return {
            "psnr_mean": float(np.mean([x["psnr"] for x in metrics])),
            "psnr_median": float(np.median([x["psnr"] for x in metrics])),
            "ssim_mean": float(np.mean([x["ssim"] for x in metrics])),
            "frames": metrics,
        }

    @torch.no_grad()
    def save(self, out):
        torch.save(
            {
                "params": {k: v.detach().cpu() for k, v in self.p.items()},
                "filter": self.filter.cpu(),
                "degree": self.degree,
            },
            out / "model.pt",
        )
        scale, opacity = self.activated()
        sh = torch.cat([self.p["sh0"], self.p["shN"]], 1)
        n = len(scale)
        # Bake 3D filter into standard PLY; 2D opacity compensation remains a renderer feature.
        arrays = [
            self.p["means"],
            torch.zeros_like(scale),
            sh[:, 0],
            sh[:, 1:].transpose(1, 2).reshape(n, -1),
            torch.logit(opacity.clamp(1e-7, 1 - 1e-7))[:, None],
            scale.log(),
            F.normalize(self.p["quats"], dim=-1),
        ]
        names = (
            ["x", "y", "z", "nx", "ny", "nz"]
            + [f"f_dc_{i}" for i in range(3)]
            + [f"f_rest_{i}" for i in range(45)]
            + ["opacity"]
            + [f"scale_{i}" for i in range(3)]
            + [f"rot_{i}" for i in range(4)]
        )
        values = torch.cat(arrays, 1).cpu().numpy()
        vertices = np.empty(n, dtype=[(k, "f4") for k in names])
        for j, k in enumerate(names):
            vertices[k] = values[:, j]
        PlyData([PlyElement.describe(vertices, "vertex")]).write(str(out / "scene.ply"))

    @torch.no_grad()
    def benchmark(self):
        # Two distinct eye views, 64mm IPD, 90deg vertical FOV, 1832x1920 each.
        w, h = 1832, 1920
        K = torch.tensor(
            [[h / 2, 0, w / 2], [0, h / 2, h / 2], [0, 0, 1]],
            device="cuda",
            dtype=torch.float32,
        )[None].repeat(2, 1, 1)
        results = []
        for i in self.data.test[np.linspace(0, len(self.data.test) - 1, 5).astype(int)]:
            c = self.data.poses[i] @ np.diag([1, -1, -1, 1])
            eyes = np.stack([c.copy(), c.copy()])
            eyes[0, :3, 3] -= c[:3, 0] * 0.032
            eyes[1, :3, 3] += c[:3, 0] * 0.032
            v = torch.tensor(np.linalg.inv(eyes), dtype=torch.float32, device="cuda")
            for _ in range(5):
                self.render(views=v, K=K, w=w, h=h)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(30):
                self.render(views=v, K=K, w=w, h=h)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            results.append(
                {
                    "frame": int(i),
                    "stereo_fps": 30 / elapsed,
                    "ms_pair": elapsed / 30 * 1000,
                }
            )
        return {
            "eye_size": [w, h],
            "vertical_fov_degrees": 90,
            "ipd_m": 0.064,
            "views": results,
            "stereo_fps_median": float(np.median([x["stereo_fps"] for x in results])),
        }


@torch.no_grad()
def sheets(model, out):
    d = model.data
    for i in d.test[np.linspace(0, len(d.test) - 1, 5).astype(int)]:
        render = model.render(int(i))[0][0].clamp(0, 1).cpu().numpy()
        real = d.images[i].numpy()
        pair = np.concatenate([real, (render * 255).astype(np.uint8)], 1)
        im = Image.fromarray(pair)
        ImageDraw.Draw(im).text(
            (8, 8),
            f"REAL | 3DGS {d.mode}; held-out frame {i + 1}",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
        im.save(out / f"heldout-{i + 1:06d}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--pose", default="true")
    parser.add_argument("--init", default="mast3r", choices=list(FILES))
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--name", default=None)
    parser.add_argument("--maxpoints", type=int, default=40000)
    parser.add_argument("--estimated-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    start = time.perf_counter()
    data = Dataset(args.run, args.pose, args.init, args.estimated_only)
    model = Model(data, maxpoints=args.maxpoints)
    out = data.out / (args.name or args.pose)
    out.mkdir(exist_ok=True, parents=True)
    torch.cuda.synchronize()
    prep = time.perf_counter() - start
    # Warm forward/backward kernels without modifying parameters, optimizer or strategy.
    pred, _, _ = model.render(int(data.train[0]))
    (pred.mean() + ssim(pred[0], data.image(data.train[0]))).backward()
    for v in model.p.values():
        v.grad = None
    torch.cuda.synchronize()
    warm = time.perf_counter() - start - prep
    torch.cuda.reset_peak_memory_stats()
    milestones = sorted({x for x in [5, 15, 30, 60, 180, args.seconds] if x <= args.seconds})
    records = []
    elapsed = 0.0
    step = 0
    cursor = 0
    rng = np.random.default_rng(42)
    order = rng.permutation(data.train)
    print(
        json.dumps(
            {
                "event": "ready",
                "prep_s": prep,
                "warm_s": warm,
                "initial_points": len(model.p["means"]),
                "scene_scale": model.scene_scale,
            }
        ),
        flush=True,
    )
    while elapsed < args.seconds:
        if cursor == len(order):
            order = rng.permutation(data.train)
            cursor = 0
        torch.cuda.synchronize()
        began = time.perf_counter()
        loss = model.step(int(order[cursor]), step)
        torch.cuda.synchronize()
        elapsed += time.perf_counter() - began
        step += 1
        cursor += 1
        if milestones and elapsed >= milestones[0]:
            target = milestones.pop(0)
            evaluation_start = time.perf_counter()
            metrics = model.evaluate()
            evaluation = time.perf_counter() - evaluation_start
            record = {
                "target_s": target,
                "training_s": elapsed,
                "steps": step,
                "gaussians": len(model.p["means"]),
                "gpu_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
                "gpu_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
                "psnr": metrics["psnr_mean"],
                "psnr_median": metrics["psnr_median"],
                "ssim": metrics["ssim_mean"],
                "evaluation_s": evaluation,
                "loss": loss,
            }
            record["render"] = model.benchmark()
            torch.cuda.reset_peak_memory_stats()
            if target in (5, 15, 30):
                checkpoint = out / f"checkpoint-{target:g}s"
                checkpoint.mkdir(exist_ok=True)
                model.save(checkpoint)
            records.append(record)
            (out / f"eval-{target:g}s.json").write_text(json.dumps(metrics, indent=2))
            (out / "timing.json").write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)
        if step % 500 == 0:
            print(
                "progress",
                step,
                round(elapsed, 2),
                round(loss, 4),
                len(model.p["means"]),
                flush=True,
            )
    model.save(out)
    sheets(model, out)
    benchmark = model.benchmark()
    report = {
        "recording": args.run,
        "pose": args.pose,
        "initialization": args.init,
        "estimated_only": args.estimated_only,
        "prep_s": prep,
        "warm_s": warm,
        "wall_s": time.perf_counter() - start,
        "training": records,
        "render": benchmark,
        "heldout_every": 8,
        "heldout_count": len(data.test),
        "train_count": len(data.train),
        "filter": "3D min-depth over up to 64 train views, sqrt(0.2)*z/f; 2D gsplat antialiased",
        "seed": 42,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print("DONE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
