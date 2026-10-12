# ruff: noqa: E402
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repo/e2e"))
from truth import score

for run in ["sweep180", "tour"]:
    p = ROOT / "out" / run
    d = np.load(p / "data.npz")
    e = np.load(p / "fused-extracted.npz")
    report = score(e["estimates"], d["poses"][e["ids"]])
    (p / "fused-pose-score.json").write_text(json.dumps(report, indent=2))
    print(run, json.dumps(report))
