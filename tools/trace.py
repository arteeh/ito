"""Ito's performance trace (ito/trace.py) as a table: where the time goes, stage by stage.

    python tools/trace.py TRACE_DIR [--from S] [--to S] [--chrome out.json]

TRACE_DIR is the trace folder beside diagnostics.jsonl (%APPDATA%\\ito\\trace). Prints, per
stage, its process, count, rate, busy share and percentiles; then a camera frame's way to the
screen: when each stage of the frames that became a depth map (or a MASt3R-SLAM frame) ended,
counted from the frame's capture. --from/--to keep spans that far into the trace (seconds);
--chrome writes the spans for chrome://tracing or ui.perfetto.dev.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

CHAIN = (
    "camera.joined",
    "frame.straighten",
    "camera.submit",
    "worker.frame",
    "fusion.posed",
    "fusion.integrate",
    "depth.request",
    "depth.wait",
    "depth.predict",
    "depth.anchor",
    "depth.normals",
    "depth.dense",
    "depth.send",
    "depth.receive",
    "fusion.depth_keyframe",
    "worker.publish",
    "mesh.receive",
    "frame.upload",
    "slam.frame",
)


def load(folder):
    spans = []
    for path in sorted(Path(folder).glob("*.jsonl*")):
        with open(path, encoding="utf-8") as lines:
            for line in lines:
                try:
                    spans.append(json.loads(line))
                except ValueError:
                    pass  # a line cut short by a killed process
    return sorted(spans, key=lambda s: s["t"])


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


def table(spans):
    if not spans:
        return
    first, last = spans[0]["t"], max(s["t"] + s["d"] for s in spans)
    seconds = max(1e-9, (last - first) / 1e9)
    groups = defaultdict(list)
    for s in spans:
        groups[(s["p"], s["n"])].append(s["d"] / 1e6)
    print(f"{seconds:.1f} s of trace, {len(spans):,} spans\n")
    print(f"{'process':9} {'stage':28} {'count':>7} {'/s':>6} {'busy%':>6} "
          f"{'p50 ms':>8} {'p95 ms':>8} {'max ms':>8}")  # fmt: skip
    for (process, name), values in sorted(groups.items(), key=lambda g: -sum(g[1])):
        busy = 100 * sum(values) / 1000 / seconds
        print(
            f"{process:9} {name:28} {len(values):7d} {len(values) / seconds:6.1f} {busy:6.1f} "
            f"{percentile(values, 50):8.2f} {percentile(values, 95):8.2f} {max(values):8.2f}"
        )
    fields = defaultdict(lambda: defaultdict(list))
    for s in spans:
        for k, v in s.items():
            if k not in ("n", "t", "d", "p", "th", "k", "gpu") and isinstance(v, (int, float)):
                fields[s["n"]][k].append(v)
    print("\nfields (p50 / p95):")
    for name, values in sorted(fields.items()):
        text = ", ".join(
            f"{k} {percentile(v, 50):.1f} / {percentile(v, 95):.1f}" for k, v in values.items()
        )
        print(f"  {name}: {text}")


def chain(spans):
    """Stage end times of each traced frame, ms after its capture (its key)."""
    ends = defaultdict(dict)
    for s in spans:
        if "k" in s and s["n"] in CHAIN and s["n"] not in ends[s["k"]]:
            ends[s["k"]][s["n"]] = (s["t"] + s["d"]) / 1e6 - s["k"] / 1e3
            if s["n"] in ("depth.predict", "slam.frame"):
                ends[s["k"]][s["n"] + " start"] = s["t"] / 1e6 - s["k"] / 1e3
    for start in ("depth.predict", "slam.frame", "fusion.integrate"):
        keys = sorted(k for k, e in ends.items() if start in e)
        if keys:
            break
    else:
        return
    followed = [ends[k] for k in keys]
    print(f"\n{len(followed)} frames ({start}) followed from capture (ms after it, p50 / p95):")
    names = [n for n in CHAIN for n in (n + " start", n)]
    for name in names:
        values = [e[name] for e in followed if name in e]
        if values:
            print(f"  {name:28} {percentile(values, 50):8.1f} {percentile(values, 95):8.1f}"
                  f"   ({len(values)})")  # fmt: skip
    if len(keys) > 1:
        gaps = [(b - a) / 1e3 for a, b in zip(keys, keys[1:], strict=False)]
        print(f"  between followed frames: p50 {statistics.median(gaps):.1f} ms "
              f"({1000 / statistics.median(gaps):.1f} a second)")  # fmt: skip


def chrome(spans, path):
    events = []
    threads, processes = {}, {}
    for s in spans:
        pid = processes.setdefault(s["p"], len(processes) + 1)
        tid = threads.setdefault((pid, s["th"], s.get("gpu", False)), len(threads) + 1)
        args = {k: v for k, v in s.items() if k not in ("n", "t", "d", "p", "th")}
        events.append(
            dict(name=s["n"], ph="X", ts=s["t"] / 1e3, dur=s["d"] / 1e3, pid=pid, tid=tid,
                 args=args)
        )  # fmt: skip
    for name, pid in processes.items():
        events.append(dict(name="process_name", ph="M", pid=pid, args={"name": name}))
    for (pid, thread, gpu), tid in threads.items():
        label = "GPU" if gpu else f"thread {thread}"
        events.append(dict(name="thread_name", ph="M", pid=pid, tid=tid, args={"name": label}))
    Path(path).write_text(json.dumps({"traceEvents": events}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--from", dest="start", type=float, default=0)
    parser.add_argument("--to", type=float, default=None)
    parser.add_argument("--chrome")
    args = parser.parse_args()
    spans = load(args.folder)
    if spans:
        zero = spans[0]["t"]
        end = None if args.to is None else zero + args.to * 1e9
        spans = [
            s for s in spans if s["t"] >= zero + args.start * 1e9 and (end is None or s["t"] < end)
        ]
    table(spans)
    chain(spans)
    if args.chrome:
        chrome(spans, args.chrome)


if __name__ == "__main__":
    main()
