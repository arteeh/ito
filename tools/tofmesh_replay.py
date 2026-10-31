"""Replay a Microduck tape (a driver recording: frames, motion, sensors) through the ToF mesh
pipeline in tape time, as the pilot would feed it, and optionally film it.

    uv run --extra microduck python tools/tofmesh_replay.py TAPE OUT [--start S] [--end S]
        [--video] [--size 960x540] [--fps 15] [--voxel 0.04]

The pipeline only sees what a pilot sees: the driver's camera pose stream, the ToF grids and the
camera frames (rectified), with the exposure estimated as received - frame_delay. The tape's
simulator truth is read only for the --truth reference path drawn in the overview, never fed in.

OUT gets stats.jsonl (one line per second of tape: step ms, mesh size, RSS) and, with --video,
pilot.mp4 (the mesh from the robot's camera) and overview.mp4 (a fixed view over the room).
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ito.tofmesh.fusion import Fusion  # noqa: E402
from ito.tofmesh.geometry import rotation  # noqa: E402


def pose_matrix(position, xyzw):
    m = np.eye(4)
    m[:3, :3] = rotation(xyzw)
    m[:3, 3] = position
    return m


class Tape:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.w, self.h = self.meta["width"], self.meta["height"]
        matrix = np.array(self.meta["matrix"], np.float64)
        distortion = np.array(self.meta["distortion"], np.float64)
        self.K = np.array(self.meta["rectified"], np.float64)
        self.maps = cv2.initUndistortRectifyMap(
            matrix, distortion, None, self.K, (self.w, self.h), cv2.CV_32FC1
        )
        self.frames = [json.loads(x) for x in (self.path / "frames.jsonl").open()]
        self.frames = [f for f in self.frames if f.get("file")]
        self.motion = [json.loads(x) for x in (self.path / "motion.jsonl").open()]
        self.sensors = [json.loads(x) for x in (self.path / "sensors.jsonl").open()]
        self.t0 = self.frames[0]["received"]

    def rgb(self, frame, size):
        bgr = cv2.imread(str(self.path / "frames" / frame["file"]), cv2.IMREAD_COLOR)
        rect = cv2.remap(bgr[..., ::-1], *self.maps, cv2.INTER_LINEAR)
        return np.ascontiguousarray(cv2.resize(rect, size, interpolation=cv2.INTER_AREA))

    def events(self, start, end):
        """(arrival, kind, payload) in tape order; arrivals model the link's delays."""
        out = []
        lo, hi = self.t0 + start, self.t0 + end
        for m in self.motion:
            p = m["camera_pose"]
            if lo - 1 <= m["time"] <= hi:
                out.append(
                    (
                        m["time"] + 0.02,
                        "motion",
                        (m["time"], pose_matrix(p["position"], p["orientation"])),
                    )
                )
        for s in self.sensors:
            for g in s.get("ranges", []):
                if lo <= g["time"] <= hi:
                    mount = g["mount"]
                    out.append(
                        (
                            g["time"] + 0.03,
                            "tof",
                            dict(
                                time=g["time"],
                                mount=pose_matrix(mount["position"], mount["orientation"]),
                                shape=(g["rows"], g["cols"]),
                                field_of_view=g["field_of_view"],
                                distances=[np.nan if d is None else d for d in g["distances"]],
                            ),
                        )
                    )
        for f in self.frames:
            if lo <= f["received"] <= hi:
                out.append((f["received"], "frame", f))
        out.sort(key=lambda e: e[0])
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tape")
    parser.add_argument("out")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, default=60)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--size", default="960x540")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--voxel", type=float, default=0.04)
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--overview", default="")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tape = Tape(args.tape)
    fusion = Fusion(voxel=args.voxel)
    fw = args.frame_width
    fh = round(fw * tape.h / tape.w)
    s = fw / tape.w
    K = (
        tape.K[0, 0] * s,
        tape.K[1, 1] * s,
        (tape.K[0, 2] + 0.5) * s - 0.5,
        (tape.K[1, 2] + 0.5) * s - 0.5,
    )
    delay = tape.meta.get("frame_delay", 0.09)
    events = tape.events(args.start, args.end)
    film = None
    if args.video:
        from tofmesh_film import Film

        width, height = map(int, args.size.split("x"))
        film = Film(out, (width, height), args.fps, K, (fw, fh), args.overview)
    stats = open(out / "stats.jsonl", "w")
    tick = 1 / 30
    now = tape.t0 + args.start
    i = 0
    steps = []
    next_stat = now + 1
    next_film = now
    while i < len(events) or now < tape.t0 + args.end:
        now += tick
        while i < len(events) and events[i][0] <= now:
            _, kind, payload = events[i]
            i += 1
            if kind == "motion":
                fusion.motion([payload])
            elif kind == "tof":
                fusion.ranges([payload])
            else:
                image = tape.rgb(payload, (fw, fh))
                fusion.frame(image, payload["received"] - delay, K)
        started = time.perf_counter()
        news = fusion.step(now)
        steps.append((time.perf_counter() - started) * 1000)
        if film is not None:
            if news is not None:
                film.apply(news)
            if now >= next_film:
                next_film += 1 / args.fps
                film.draw(now - tape.t0, fusion)
        if now >= next_stat:
            next_stat += 1
            rss = int(open("/proc/self/statm").read().split()[1]) * resource.getpagesize() / 2**20
            row = dict(
                t=round(now - tape.t0, 1),
                step_ms_p50=float(np.median(steps)),
                step_ms_max=float(np.max(steps)),
                rss_mb=round(rss),
                **fusion.stats,
            )
            stats.write(json.dumps(row) + "\n")
            stats.flush()
            steps = []
        if i >= len(events) and now >= tape.t0 + args.end:
            break
    if film is not None:
        film.close()
    print(json.dumps(row))


if __name__ == "__main__":
    main()
