# ruff: noqa: B023
"""Truth/main/mesh eye checks with the frozen observations and inference availability."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import moderngl
import numpy as np
from measure import MOMENTS
from measure_gl import projection
from measure_images import Metrics, percentiles
from PIL import Image, ImageDraw

from ito.proto.depthmesh import MeshWorld
from ito.protocol import Intrinsics

VIEWS = [
    ("camera", 0),
    ("left", -0.032),
    ("right", 0.032),
    ("head-minus3cm", -0.03),
    ("head-plus3cm", 0.03),
]


def sheet(images, captions, path):
    h, w = images[0].shape[:2]
    result = Image.new("RGB", (w, (h + 36) * len(images)), (24, 24, 24))
    draw = ImageDraw.Draw(result)
    for k, (image, text) in enumerate(zip(images, captions, strict=True)):
        draw.text((8, k * (h + 36) + 10), text, fill="white")
        result.paste(Image.fromarray(image), (0, k * (h + 36) + 36))
    for quality in (90, 80, 70, 60):
        result.save(path, quality=quality)
        if path.stat().st_size < 1_000_000:
            break


def depth_diagnostics(old, current, old_pose, new_pose, i):
    """Same static truth point across frames; diagnostic truth never changes predicted depth."""
    y, x = np.mgrid[2 : i.height : 4, 2 : i.width : 4]
    z = old["truth_depth"][y, x]
    p = (
        np.stack(((x + 0.5 - i.cx) / i.fx, (i.cy - y - 0.5) / i.fy, -np.ones_like(z)), -1)
        * z[..., None]
    )
    transform = np.linalg.inv(new_pose) @ old_pose
    p = p.reshape(-1, 3) @ transform[:3, :3].T + transform[:3, 3]
    nz = -p[:, 2]
    u = np.floor(i.fx * p[:, 0] / np.maximum(nz, 1e-6) + i.cx).astype(int)
    v = np.floor(i.cy - i.fy * p[:, 1] / np.maximum(nz, 1e-6)).astype(int)
    valid = (nz > 0.05) & (u >= 0) & (u < i.width) & (v >= 0) & (v < i.height)
    ids = np.flatnonzero(valid)
    valid[ids] &= np.abs(current["truth_depth"][v[ids], u[ids]] - nz[ids]) < 0.02 + 0.01 * nz[ids]
    ids = np.flatnonzero(valid)
    if not len(ids):
        return np.empty(0)
    old_d = old["prediction"][y, x].ravel()[ids]
    new_d = current["prediction"][v[ids], u[ids]]
    ratio = new_d / np.maximum(old_d, 0.05) * z.ravel()[ids] / nz[ids]
    return np.abs(np.log(np.maximum(ratio, 1e-6)))


def evaluate(args):
    baseline = args.work / "m1-out/verified/repeat-1"
    name = f"{args.variant}-{args.pose}-cap{args.cap}"
    out = args.root / "rendered" / name
    out.mkdir(parents=True, exist_ok=True)
    metrics = Metrics()
    report = dict(
        variant=args.variant,
        pose=args.pose,
        cap=args.cap,
        moments=[],
        publications=[],
        input_size=392,
        grid=[90, 160],
        jump=[0.02, 0.08],
        retirement=[0.02, 0.03],
        availability="Queued measured CUDA inference + declared mesh-update budget",
        mesh_update_budget_ms=args.update_ms,
        gpu_deadlines_certified=False,
        tof="Only prompt-small uses ideal simulated registered 8x8 zones, 4m cap",
        future_observations=0,
        future_publications=0,
        depth_diagnostics={},
    )
    for run, moments in MOMENTS.items():
        source = args.root / "prepared" / run
        tr = np.load(source / "trajectory.npz")
        poses = tr[args.pose if args.pose == "robot" else "truth"]
        true_poses = np.load(args.work / f"gs/out/{run}/data.npz")["poses"]
        i = Intrinsics.model_validate_json((source / "camera.json").read_text())
        schedule = json.loads((source / "schedule.json").read_text())
        snapshots = [s for s in schedule["snapshots"] if s["frame"] in moments]
        inference_folder = args.root / "depth" / args.variant / run
        if args.variant == "truth":
            timings = {int(f): 0 for f in tr["frames"]}
        else:
            inference = json.loads((inference_folder / "inference.json").read_text())
            timings = {r["frame"]: r["seconds"] for r in inference["timings"]}
        ctx = moderngl.create_standalone_context(require=430, backend="egl")
        report["gl_renderer"] = ctx.info["GL_RENDERER"]
        if "llvmpipe" not in ctx.info["GL_RENDERER"].lower():
            raise RuntimeError("Offline evaluation must use Mesa, not the shared GPU")
        world = MeshWorld(ctx, i, cap=args.cap, carve=not args.no_carve)
        target = ctx.simple_framebuffer((i.width, i.height), components=4)
        panels, scores = {}, []
        diagnostic_scale, diagnostic_absrel, temporal = [], [], []
        pending = iter(sorted(snapshots, key=lambda s: s["deadline"]))
        next_snapshot = next(pending, None)
        finish, last_frame, last_capture = 0.0, -1, None
        previous = None
        update_ms = []

        def capture(snapshot):
            index, deadline = snapshot["frame"], snapshot["deadline"]
            if last_frame > index or finish > deadline:
                raise AssertionError("Future patch publication")
            for label, offset in VIEWS:
                camera = true_poses[index].copy()
                camera[:3, 3] += camera[:3, 0] * offset
                world.draw(camera, projection(i), target)
                pixels = (
                    np.frombuffer(target.read(components=4), "u1")
                    .reshape(i.height, i.width, 4)[::-1]
                    .copy()
                )
                image = pixels[..., :3].astype("f4") / 255
                covered = pixels[..., 3] > 127
                filename = f"{run}-{index + 1:04d}-{label}"
                frozen = np.asarray(Image.open(baseline / f"{filename}.png").convert("RGB"))
                truth = frozen[:, : i.width].astype("f4") / 255
                masks = np.load(baseline / f"{filename}-mask.npz")
                values = metrics.score(truth, image, covered, masks["seen"])
                record = dict(
                    run=run,
                    frame=index + 1,
                    view=label,
                    prefix_frame=last_frame + 1,
                    publication=finish,
                    deadline=deadline,
                    newest_capture_age_s=deadline - last_capture
                    if last_capture is not None
                    else None,
                    **values,
                )
                report["moments"].append(record)
                triplet = np.concatenate((frozen, pixels[..., :3]), axis=1)
                Image.fromarray(triplet).save(out / f"{filename}.png")
                np.savez_compressed(
                    out / f"{filename}-mask.npz", covered=covered, seen=masks["seen"]
                )
                panels[(index + 1, label)] = triplet
                scores.append(record)
                print(
                    name,
                    run,
                    index + 1,
                    label,
                    f"{values['psnr_db']:.2f}dB",
                    f"coverage {values['coverage'] * 100:.1f}%",
                    flush=True,
                )
            if args.save_world and index == max(moments):
                world.save(out / f"{run}-world.npz")
                np.save(out / f"{run}-camera.npy", true_poses[index])

        for k, frame in enumerate(tr["frames"]):
            elapsed = timings[int(frame)] + args.update_ms / 1000
            publication = max(finish, float(tr["arrivals"][k])) + elapsed
            while next_snapshot is not None and next_snapshot["deadline"] < publication:
                capture(next_snapshot)
                next_snapshot = next(pending, None)
            sample = np.load(source / f"{frame:06d}.npz")
            depth = (
                sample["truth_depth"]
                if args.variant == "truth"
                else np.load(inference_folder / f"{frame:06d}.npy")
            )
            current = dict(truth_depth=sample["truth_depth"], prediction=depth)
            valid = (current["truth_depth"] > 0.05) & (current["truth_depth"] < 20)
            diagnostic_scale.append(float(np.median(depth[valid] / current["truth_depth"][valid])))
            diagnostic_absrel.append(
                float(
                    np.median(
                        np.abs(depth[valid] - current["truth_depth"][valid])
                        / current["truth_depth"][valid]
                    )
                )
            )
            if previous is not None:
                temporal.append(
                    depth_diagnostics(previous, current, tr["truth"][k - 1], tr["truth"][k], i)
                )
            previous = current
            started = time.perf_counter()
            world.add(
                sample["rgb"], depth, poses[k], frame=int(frame), captured=float(tr["captured"][k])
            )
            ctx.finish()
            update_ms.append((time.perf_counter() - started) * 1000)
            finish, last_frame, last_capture = publication, int(frame), float(tr["captured"][k])
            report["publications"].append(
                dict(
                    run=run,
                    frame=int(frame),
                    capture=last_capture,
                    arrival=float(tr["arrivals"][k]),
                    published=finish,
                )
            )
            while next_snapshot is not None and next_snapshot["deadline"] <= finish:
                capture(next_snapshot)
                next_snapshot = next(pending, None)
        while next_snapshot is not None:
            capture(next_snapshot)
            next_snapshot = next(pending, None)
        report["depth_diagnostics"][run] = dict(
            frame_median_predicted_over_true=percentiles(diagnostic_scale),
            frame_median_absrel=percentiles(diagnostic_absrel),
            same_surface_abs_log_depth_change=percentiles(np.concatenate(temporal)),
            mesa_update_ms=percentiles(update_ms),
            note="Diagnostic truth only; log change removes actual camera depth change.",
        )
        reference = [f + 1 for f in sorted(moments) if f != 525]
        for label, ids in (("camera", reference), ("eyes", [max(reference)])):
            keys = (
                [(f, "camera") for f in ids]
                if label == "camera"
                else [(ids[0], e) for e in ("left", "right")]
            )
            captions = [f"{run} #{f} {v} | sim truth / frozen main / {name}" for f, v in keys]
            sheet([panels[key] for key in keys], captions, out / f"{run}-{label}.jpg")
        if run == "sweep180":
            sheet(
                [panels[(526, "camera")]],
                [f"sweep #526 hard turn | sim truth / frozen main / {name}"],
                out / "hard-526.jpg",
            )
        world.close()
        target.release()
        ctx.release()
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("../p47-out").resolve())
    p.add_argument("--work", type=Path, default=Path("..").resolve())
    p.add_argument("--variant", default="truth")
    p.add_argument("--pose", choices=["oracle", "robot"], default="oracle")
    p.add_argument("--cap", type=int, default=64)
    p.add_argument("--update-ms", type=float, default=10)
    p.add_argument("--no-carve", action="store_true")
    p.add_argument("--save-world", action="store_true")
    evaluate(p.parse_args())
