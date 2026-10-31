# ruff: noqa: E501, B905, UP031  (analysis script for the PoC, not product code)
"""Summarise a run's head-IMU log (timeline.jsonl.imu.jsonl): lost frames, the turns around the
losses, and the gyro's prediction error on tracked frames. Usage: imulog.py RUN [RUN...]"""

import json
import sys

import numpy as np

for run in sys.argv[1:]:
    rows = [json.loads(x) for x in open(f"{run}/timeline.jsonl.imu.jsonl")]
    t0 = rows[0]["t"]
    lost = [r for r in rows if r["outcome"] == "lost"]
    tracked = [r for r in rows if r["outcome"] == "tracked"]
    print(
        f"== {run.split('/')[-1]}: frames {len(rows)}, tracked {len(tracked)}, lost {len(lost)}, "
        f"seeded {sum(bool(r.get('seeded')) for r in rows)}"
    )
    err = [r["gyro_vs_tracked_deg"] for r in tracked if "gyro_vs_tracked_deg" in r]
    turn = [r["turn_since_last_deg"] for r in tracked if "gyro_vs_tracked_deg" in r]
    if err:
        err, turn = np.array(err), np.array(turn)
        big = turn > 5
        print(
            f"  gyro vs tracked rotation: med {np.median(err):.2f} deg, on turns >5 deg "
            f"({big.sum()}): med {np.median(err[big]) if big.any() else 0:.2f} p90 "
            f"{np.percentile(err[big], 90) if big.any() else 0:.2f}"
        )
    # each loss: the frames just before, and how long until tracked again
    previous = None
    for i, r in enumerate(rows):
        if r["outcome"] == "lost" and (previous is None or previous["outcome"] != "lost"):
            back = next((x for x in rows[i:] if x["outcome"] == "tracked"), None)
            ctx = rows[max(0, i - 2) : i + 1]
            print(
                f"  lost at {r['t'] - t0:6.1f} s, back {'%.1f s later' % (back['t'] - r['t']) if back else 'never'}: "
                + " | ".join(
                    f"turn {x.get('turn_since_last_deg', '-')} kf {x.get('turn_from_keyframe_deg', '-')} "
                    f"m {x.get('matched', '-')}/{x.get('confident', '-')} {x['outcome'][0]}"
                    + (" S" if x.get("seeded") else "")
                    for x in ctx
                )
            )
        previous = r
