# ruff: noqa: E501
"""Markdown tables of every training run, causal stream, pose split and renderer check."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


print("| recording | poses | init | s | PSNR | SSIM | Gaussians | stereo pairs/s | GPU MB |")
print("|---|---|---|---|---|---|---|---|---|")
for run in ["sweep180", "tour"]:
    for summary in sorted((OUT / run).glob("*/summary.json")):
        d = load(summary)
        if "training" not in d:
            continue
        name = summary.parent.name
        for r in d["training"]:
            fps = r.get("render", {}).get("stereo_fps_median")
            if fps is None and r is d["training"][-1]:
                fps = d["render"]["stereo_fps_median"]
            print(
                f"| {run} | {d['pose']} | {d.get('initialization', 'mast3r')} ({name}) "
                f"| {r['target_s']:g} | {r['psnr']:.2f} | {r['ssim']:.3f} | {r['gaussians']} "
                f"| {'' if fps is None else f'{fps:.0f}'} | {r['gpu_reserved_mb']:.0f} |"
            )
print()
print(
    "| recording | poses | init | targets | causal 3DGS PSNR median | Ito L1 same targets | SSIM | "
    "end Gaussians | end stereo pairs/s |"
)
print("|---|---|---|---|---|---|---|---|---|")
for run in ["sweep180", "tour"]:
    for summary in sorted((OUT / run).glob("*-stream*/summary.json")):
        d = load(summary)
        print(
            f"| {run} | {d['pose']} | {d.get('initialization', 'mast3r')} | {d['predictions']} "
            f"| {d['prediction_psnr_median']:.2f} | {d['baseline_same_targets_median']:.2f} "
            f"| {d['ssim_mean']:.3f} | {json.loads((summary.parent / 'predictions.json').read_text())[-1]['gaussians']} "
            f"| {d['render']['stereo_fps_median']:.0f} |"
        )
print()
print(
    "| recording | model | own-pose PSNR | true-pose PSNR | refined-pose PSNR | refine rot p50/p95 deg |"
)
print("|---|---|---|---|---|---|")
for run in ["sweep180", "tour"]:
    for f in sorted((OUT / run).glob("eval-pose-*.json")):
        d = load(f)
        p50, p95 = d["refine_rot_deg_p50_p95"]
        print(
            f"| {run} | {d['model']} ({d['pose']} poses) | {d['own_psnr_mean']:.2f} "
            f"| {d['true_psnr_mean']:.2f} | {d['refined_psnr_mean']:.2f} | {p50:.2f}/{p95:.2f} |"
        )
print()
for f in sorted(OUT.glob("*/**/ito-match.json")):
    d = load(f)
    print(
        f"- {f.parent.relative_to(OUT)}: {d['count']} Gaussians, SH {d['sh_degree']}, Ito vs gsplat "
        f"antialiased {d['antialiased_psnr']:.1f} dB, vs classic {d['classic_psnr']:.1f} dB, "
        f"Ito GL {d.get('ito_gl_stereo_pairs_per_s', float('nan')):.0f} stereo pairs/s"
    )
for run in ["sweep180", "tour"]:
    for name in ["pose-gap.json", "scale-check.json"]:
        d = load(OUT / run / name)
        if d:
            print(f"- {run} {name}: {json.dumps(d)}")
