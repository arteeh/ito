# ruff: noqa: E501, B905  (analysis script for the PoC, not product code)
"""Loss episodes of replay runs from their timeline: when tracking was lost (tape time) and for
how long (until the next tracked frame). Usage: episodes.py START RUN [RUN...]"""

import json
import sys

start = float(sys.argv[1])
for run in sys.argv[2:]:
    rows = [json.loads(x) for x in open(f"{run}/timeline.jsonl") if "backend_frame" in x]
    c0 = rows[0]["captured"]
    eps, lost_at, total = [], None, 0.0
    for r in rows:
        t = r["captured"] - c0 + start
        if r["outcome"] == "lost" and lost_at is None:
            lost_at = t
        elif r["outcome"] != "lost" and lost_at is not None:
            eps.append((lost_at, t - lost_at))
            total += t - lost_at
            lost_at = None
    if lost_at is not None:
        end = rows[-1]["captured"] - c0 + start
        eps.append((lost_at, None))
        total += end - lost_at
    lost = sum(r["outcome"] == "lost" for r in rows)
    print(
        f"{run.split('/')[-1]:18s} frames {len(rows)} lost {lost} | time lost {total:5.1f} s | "
        + ", ".join(f"{a:.0f}s:{'%.1f' % d if d is not None else 'end'}" for a, d in eps)
    )
