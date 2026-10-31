"""Numbers of tools/tapes replay runs (RUN/timeline.jsonl with ITO_TIMELINE, RUN/vram.csv,
RUN/ticks.jsonl), side by side, and a memory plot (OUT.png) over each run:

    uv run --no-project --with numpy --with opencv-python-headless python stats.py OUT.png RUN...

Per run: frames the worker took per second, pose latency (capture to pose out: the frame's age
when the worker took it plus its work), the worker's CUDA memory (Torch reserved + CuPy pool)
and private memory, the card's memory (nvidia-smi), map trainer steps per second."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def load(run):
    run = Path(run)
    rows = [json.loads(x) for x in (run / "timeline.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("event") == "backend_frame"]
    card = np.array(
        [[float(v) for v in x.split(",")] for x in (run / "vram.csv").read_text().split()]
    )
    return run.name, rows, card


def describe(name, rows, card):
    t = np.array([r["began"] for r in rows])
    span = t[-1] - t[0]
    latency = np.array([r["age_ms"] + r["work_ms"] for r in rows])
    work = np.array([r["work_ms"] for r in rows])
    cuda = np.array([r["cuda_memory_bytes"] for r in rows]) / 2**30
    rss = np.array([r["working_set_bytes"] for r in rows]) / 2**30
    steps = [r.get("map_trainer_steps") for r in rows]
    half = len(rows) // 2
    out = dict(
        run=name,
        frames=len(rows),
        fps=round(len(rows) / span, 2),
        latency_ms_p50=round(float(np.median(latency)), 1),
        latency_ms_p95=round(float(np.percentile(latency, 95)), 1),
        work_ms_p50=round(float(np.median(work)), 1),
        work_ms_p95=round(float(np.percentile(work, 95)), 1),
        cuda_gib_first_half_max=round(float(cuda[:half].max()), 2),
        cuda_gib_second_half_max=round(float(cuda[half:].max()), 2),
        cuda_gib_last=round(float(cuda[-1]), 2),
        rss_gib_first_half_max=round(float(rss[:half].max()), 2),
        rss_gib_second_half_max=round(float(rss[half:].max()), 2),
        card_gib_max=round(float(card[:, 1].max()) / 1024, 2),
    )
    if steps[-1] is not None:
        out["trainer_steps"] = steps[-1]
        out["trainer_steps_per_s"] = round((steps[-1] - (steps[0] or 0)) / span, 1)
    return out, t - t[0], cuda, rss, card


def main():
    out = Path(sys.argv[1])
    runs = [load(r) for r in sys.argv[2:]]
    W, H = 1200, 300 * len(runs)
    plot = np.full((H, W, 3), 255, np.uint8)
    for i, run in enumerate(runs):
        info, t, cuda, rss, card = describe(*run)
        print(json.dumps(info))
        y0 = i * 300
        top = 12.0
        cv2.putText(plot, run[0], (10, y0 + 20), 0, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        series = [
            (t, cuda, (200, 80, 0), "worker CUDA GiB"),
            (t, rss, (0, 140, 0), "worker RAM GiB"),
            (card[:, 0] - card[0, 0], card[:, 1] / 1024, (0, 0, 200), "card GiB (nvidia-smi)"),
        ]
        span = max(t[-1], 1)
        for k, (x, y, colour, label) in enumerate(series):
            px = (60 + x / span * (W - 80)).astype(int)
            py = (y0 + 280 - y / top * 240).astype(int)
            cv2.polylines(plot, [np.stack([px, py], 1).reshape(-1, 1, 2)], False, colour, 2)
            cv2.putText(plot, label, (300 + 260 * k, y0 + 20), 0, 0.5, colour, 1, cv2.LINE_AA)
        for g in range(0, 13, 4):
            yy = int(y0 + 280 - g / top * 240)
            cv2.line(plot, (55, yy), (W - 20, yy), (220, 220, 220), 1)
            cv2.putText(plot, f"{g}", (20, yy + 5), 0, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), plot)


if __name__ == "__main__":
    main()
