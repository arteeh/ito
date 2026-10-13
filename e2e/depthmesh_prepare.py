"""Freeze mesh inputs from the unchanged baseline; sim depth never enters mono variants."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from measure_images import Room
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from truth import align, score

from ito.protocol import Intrinsics


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(work, out):
    from measure import prepare as freeze_inputs

    reference = out.parent / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    freeze_inputs(work, reference)
    manifest = {
        "sequences": {},
        "arrival": "legacy capture timestamps, as in frozen baseline",
        "robot_pose": "Only motion samples available at admission; fixed driver 90 ms delay",
        "truth": "Legacy thumbnail-matched trajectory, centred evaluation K; no new live capture",
    }
    for run in ("sweep180", "tour"):
        rec = work / f"rf-run/out/{run}/rec"
        rows = [json.loads(s) for s in (rec / "frames.jsonl").read_text().splitlines()]
        motion = [json.loads(s) for s in (rec / "motion.jsonl").read_text().splitlines()]
        plan = json.loads((work / f"m1-out/verified/{run}-schedule.json").read_text())
        last = max(s["frame"] for s in plan["snapshots"])
        events = [a for a in plan["admissions"] if a["frame"] <= last]
        i = Intrinsics.model_validate_json((rec / "camera.json").read_text())
        i = i.model_copy(update={"cx": i.width / 2, "cy": i.height / 2})
        delay = json.loads((rec / "robot.json").read_text())["cameras"][0]["capture_delay"]
        truth = np.load(work / f"gs/out/{run}/data.npz")["poses"]
        mt = np.array([m["local"] for m in motion])
        mp = np.array([m["camera"] for m in motion]).reshape(-1, 4, 4)
        rp = []
        for event in events:
            frame = event["frame"]
            available = int(np.searchsorted(mt, rows[frame]["captured"], side="right"))
            if available < 2:
                raise ValueError("Motion unavailable at first admitted frame")
            t = np.clip(rows[frame]["captured"] - delay, mt[0], mt[available - 1])
            b = min(max(1, int(np.searchsorted(mt[:available], t))), available - 1)
            pose = np.eye(4)
            pose[:3, :3] = Slerp(
                mt[b - 1 : b + 1], Rotation.from_matrix(mp[b - 1 : b + 1, :3, :3])
            )([t])[0].as_matrix()
            f = (t - mt[b - 1]) / (mt[b] - mt[b - 1])
            pose[:3, 3] = mp[b - 1, :3, 3] * (1 - f) + mp[b, :3, 3] * f
            rp.append(pose)
        ids = np.array([e["frame"] for e in events])
        robot = np.linalg.inv(align(rp[0], truth[ids[0]])) @ np.array(rp)
        folder = out / run
        folder.mkdir(parents=True, exist_ok=True)
        room = Room(work / "truth/data/scan/scene.mjb", i)
        for k, frame in enumerate(ids):
            path = folder / f"{frame:06d}.npz"
            if not path.exists():
                rgb = np.asarray(Image.open(rec / rows[frame]["name"]).convert("RGB"))
                depth = room.draw(truth[frame], depth=True)
                # Ideal registered zone medians, capped at 4m. No dense truth is passed to PromptDA.
                tof = np.array(
                    [
                        [np.median(tile) for tile in np.array_split(band, 8, axis=1)]
                        for band in np.array_split(depth, 8, axis=0)
                    ],
                    "f4",
                )
                tof[(tof < 0.05) | (tof > 4)] = 0
                np.savez_compressed(path, rgb=rgb, truth_depth=depth, tof=tof)
            if k % 50 == 0:
                print(run, k, "/", len(ids), flush=True)
        room.renderer.close()
        np.savez(
            folder / "trajectory.npz",
            frames=ids,
            truth=truth[ids],
            robot=robot,
            captured=[rows[f]["captured"] - rows[0]["captured"] for f in ids],
            arrivals=[e["arrival"] for e in events],
        )
        (folder / "camera.json").write_text(i.model_dump_json())
        (folder / "schedule.json").write_text(json.dumps(plan, indent=2) + "\n")
        manifest["sequences"][run] = dict(
            frames=len(ids),
            last_frame=int(ids[-1]),
            robot_pose_error=score(robot, truth[ids]),
            camera=i.model_dump(),
            hashes={
                str(p.relative_to(work)): digest(p)
                for p in [
                    rec / "frames.jsonl",
                    rec / "motion.jsonl",
                    rec / "robot.json",
                    work / f"gs/out/{run}/data.npz",
                    work / "truth/data/scan/scene.mjb",
                    work / f"m1-out/verified/{run}-schedule.json",
                ]
            },
        )
    (out / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=Path("..").resolve())
    p.add_argument("--out", type=Path, default=Path("../p47-out/prepared").resolve())
    args = p.parse_args()
    prepare(args.work, args.out)
