# ruff: noqa: E402
"""Standalone experiment plots, rebuilt as runs finish."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

styles = [
    ("true", "True / MASt3R", "tab:blue", "-"),
    ("fused", "Fused / MASt3R", "tab:orange", "-"),
    ("true-sparse", "True / sparse", "tab:green", "-"),
    ("true-calibrated", "True / calibrated depth", "tab:purple", "-"),
    ("fused-calibrated", "Fused / calibrated depth", "tab:red", "--"),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for row, run in enumerate(["sweep180", "tour"]):
    for name, label, color, style in styles:
        p = ROOT / "out" / run / name / "timing.json"
        if not p.exists():
            continue
        values = json.loads(p.read_text())
        t = [x["training_s"] for x in values]
        q = [x["psnr"] for x in values]
        axes[row, 0].plot(t, q, marker="o", label=label, color=color, linestyle=style)
        good = [x for x in values if "render" in x]
        if good:
            axes[row, 1].plot(
                [x["psnr"] for x in good],
                [x["render"]["stereo_fps_median"] for x in good],
                marker="o",
                label=label,
                color=color,
                linestyle=style,
            )
    axes[row, 0].set(
        title=run + " held-out quality",
        xlabel="Optimization time (s); preprocessing excluded",
        ylabel="PSNR (dB)",
        xscale="log",
    )
    axes[row, 0].axhline(27, color="gray", linestyle=":")
    axes[row, 0].axvline(30, color="gray", linestyle=":")
    axes[row, 0].grid(alpha=0.2)
    axes[row, 0].legend(fontsize=8)
    axes[row, 1].set(
        title=run + " quality / rendering",
        xlabel="Held-out PSNR (dB)",
        ylabel="Stereo pairs/s, 2 × 1832 × 1920",
    )
    axes[row, 1].axhline(90, color="gray", linestyle=":")
    axes[row, 1].axvline(27, color="gray", linestyle=":")
    axes[row, 1].grid(alpha=0.2)
fig.savefig(ROOT / "out/quality-time.png", dpi=160)
