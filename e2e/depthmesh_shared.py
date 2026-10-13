"""One locked job: Linux CUDA inference competes with Windows' offscreen GL draws."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from depthmesh_infer import load_model, predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("../p47-out").resolve())
    p.add_argument("--frames", type=int, default=1000)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.58)
    model, _, _ = load_model(args.root, "metric-small")
    sample = np.load(args.root / "prepared/tour/000000.npz")
    rgb = sample["rgb"]
    predict(model, rgb, None, "metric-small", 392)
    torch.cuda.synchronize()

    def windows(path):
        return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()

    env = os.environ.copy()
    env.update(
        PYTHONPATH=windows(Path.cwd()),
        PYTHONUTF8="1",
        PYTHONDONTWRITEBYTECODE="1",
        WSLENV="PYTHONPATH:PYTHONUTF8:PYTHONDONTWRITEBYTECODE",
    )
    out = args.root / "display-shared.json"
    log = (args.root / "display-shared.log").open("w")
    child = subprocess.Popen(
        [
            str(args.root / "winvenv/Scripts/python.exe"),
            "-u",
            "e2e/depthmesh_display.py",
            windows(args.root / "rendered/metric-small-robot-cap64/tour-world.npz"),
            "--frames",
            str(args.frames),
            "--pace",
            "--out",
            windows(out),
        ],
        env=env,
        stdout=log,
        stderr=log,
    )
    durations = []
    try:
        while child.poll() is None:
            began = time.perf_counter()
            predict(model, rgb, None, "metric-small", 392)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            durations.append(elapsed * 1000)
            time.sleep(max(0, 0.2 - elapsed))
        if child.returncode:
            raise RuntimeError(f"Own GL child failed with {child.returncode}; see {log.name}")
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        log.close()
    report = json.loads(out.read_text())
    report.update(
        scope="Stereo GL fence + concurrent 5Hz Metric DA Small; excludes live uploads/compositor",
        depth_inferences=len(durations),
        depth_inference_ms=np.percentile(durations, [50, 95, 99]).tolist(),
        peak_torch_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
    )
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
