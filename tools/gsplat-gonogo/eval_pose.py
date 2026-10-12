"""Split a pose-induced PSNR gap into "the model got blurrier" and "the test pose is off".
Renders every held-out frame of a trained model at (a) the pose set it was trained with,
(b) the true pose, (c) its own pose refined per frame by photometric test-time optimisation
(rotation + translation only; the model is frozen). (c) measures the reconstruction itself."""

import argparse
import json

import numpy as np
import torch
from train import Dataset, Model, ssim


def axis_angle(w):
    theta = w.norm() + 1e-12
    k = w / theta
    K = torch.zeros(3, 3, device=w.device)
    K[0, 1], K[0, 2], K[1, 2] = -k[2], k[1], -k[0]
    K = K - K.T
    return torch.eye(3, device=w.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K


def score(r, target):
    psnr = float(-10 * torch.log10((r - target).square().mean()))
    return psnr, float(ssim(r, target))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument(
        "model", help="folder under out/<run>, e.g. fused or true-30/checkpoint-15s"
    )
    parser.add_argument("--pose", default="fused")
    parser.add_argument("--init", default="mast3r")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--estimated-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    data = Dataset(args.run, args.pose, args.init, args.estimated_only)
    model = Model(data, maxpoints=100)
    folder = data.out / args.model
    ckpt = torch.load(folder / "model.pt", map_location="cuda", weights_only=True)
    model.p = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v.cuda(), requires_grad=False) for k, v in ckpt["params"].items()}
    )
    model.filter = ckpt["filter"].cuda()
    model.degree = ckpt["degree"]
    truth_views = torch.tensor(
        np.linalg.inv(data.truth @ np.diag([1, -1, -1, 1])), dtype=torch.float32, device="cuda"
    )
    rows = []
    for i in data.test:
        i = int(i)
        target = data.image(i)
        with torch.no_grad():
            own = score(model.render(i)[0][0].clamp(0, 1), target)
            true = score(model.render(views=truth_views[i : i + 1])[0][0].clamp(0, 1), target)
        w = torch.zeros(3, device="cuda", requires_grad=True)
        t = torch.zeros(3, device="cuda", requires_grad=True)
        opt = torch.optim.Adam([{"params": [w], "lr": 2e-3}, {"params": [t], "lr": 2e-3}])
        base = data.views[i]
        for _ in range(args.steps):
            delta = torch.eye(4, device="cuda")
            delta[:3, :3] = axis_angle(w)
            delta[:3, 3] = t
            r = model.render(views=(delta @ base)[None])[0][0]
            loss = (r - target).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            delta = torch.eye(4, device="cuda")
            delta[:3, :3] = axis_angle(w)
            delta[:3, 3] = t
            refined = score(model.render(views=(delta @ base)[None])[0][0].clamp(0, 1), target)
        rows.append(
            {
                "frame": i,
                "own_psnr": own[0],
                "true_psnr": true[0],
                "refined_psnr": refined[0],
                "own_ssim": own[1],
                "refined_ssim": refined[1],
                "refine_rot_deg": float(np.degrees(w.norm().item())),
                "refine_trans_m": float(t.norm().item()),
            }
        )
    report = {
        "run": args.run,
        "model": args.model,
        "pose": args.pose,
        "frames": len(rows),
        **{
            k + "_mean": float(np.mean([r[k] for r in rows]))
            for k in ["own_psnr", "true_psnr", "refined_psnr", "own_ssim", "refined_ssim"]
        },
        "refine_rot_deg_p50_p95": np.percentile(
            [r["refine_rot_deg"] for r in rows], [50, 95]
        ).tolist(),
        "frames_detail": rows,
    }
    name = args.model.replace("/", "_")
    (data.out / f"eval-pose-{name}.json").write_text(json.dumps(report, indent=2))
    print("DONE", json.dumps({k: v for k, v in report.items() if k != "frames_detail"}), flush=True)


if __name__ == "__main__":
    main()
