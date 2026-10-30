"""strip.py TAPE OUT.jpg [--start S] [--end E] [-n N]: N raw camera frames of a tape, evenly in time,
in one contact sheet with their tape times (agents only: to check what a tape shows)."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("tape")
p.add_argument("out")
p.add_argument("--start", type=float, default=0)
p.add_argument("--end", type=float)
p.add_argument("-n", type=int, default=24)
a = p.parse_args()
tape = Path(a.tape)
rows = [json.loads(line) for line in (tape / "frames.jsonl").read_text().splitlines()]
t0 = rows[0]["received"]
end = a.end if a.end is not None else rows[-1]["received"] - t0
times = np.linspace(a.start, end, a.n)
cells = []
for t in times:
    f = min(rows, key=lambda r: abs(r["received"] - t0 - t))
    img = cv2.resize(cv2.imread(str(tape / "frames" / f["file"])), (320, 180))
    cv2.putText(img, f"{t:.1f}s", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cells.append(img)
cells += [np.zeros_like(cells[0])] * (-len(cells) % 6)
sheet = np.concatenate([np.concatenate(cells[i : i + 6], 1) for i in range(0, len(cells), 6)], 0)
cv2.imwrite(a.out, sheet)
