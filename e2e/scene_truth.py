"""Score the real reconstruction worker against MuJoCo, including off-surface ghosts.

env -u WAYLAND_DISPLAY xvfb-run -a uv run --with scipy python e2e/scene_truth.py --label main
Use --backend slam with the installed MASt3R environment. --dump RUN/drive.npz
rescores a saved scene without reconstruction. PYTHONPATH selects another checkout.
Geometry uses a 5 mm voxel-deduplicated union of dense truth depth samples, not
image coverage; distances are upper bounds to the continuous surface. Images are
scored over every pixel, including holes. The scripted kinematic rig shares the
bundled driver's camera and joint limits; this measures reconstruction, not locomotion.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

if sys.argv[1:] == ["--serve"]:
    os.environ["MUJOCO_GL"] = "osmesa" if sys.platform == "linux" else "glfw"
    if sys.platform == "linux":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, laplace
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from splat_quality import Pipe, drive

from ito import clock
from ito.app.frames import camera_matrix
from ito.reconstruction import Reconstruction
from ito.render import GaussianBuffer, GaussianRenderer, current_context, perspective, pose

SIZE = (320, 180)
WIDE = math.degrees(2 * math.atan(SIZE[1] / SIZE[0]))  # 90 degree horizontal FOV.
THRESHOLD = 0.05


def serve():
    import mujoco

    from drivers.mujoco.adapter import MujocoAdapter

    adapter = MujocoAdapter(width=SIZE[0], height=SIZE[1], gl=os.environ["MUJOCO_GL"])
    model, data = adapter.model, adapter.data
    renderer = mujoco.Renderer(model, SIZE[1], SIZE[0])
    free = model.jnt_qposadr[model.body_jntadr[adapter.base_id]]
    joints = [model.joint(f"head_{name}_joint").id for name in ("pan", "tilt")]
    # A moving robot is not part of the static room being reconstructed.
    for geom in range(model.ngeom):
        body = model.geom_bodyid[geom]
        while body and body != adapter.base_id:
            body = model.body_parentid[body]
        if body == adapter.base_id:
            model.geom_group[geom] = 5
    option = mujoco.MjvOption()
    option.geomgroup[5] = 0
    far = model.vis.map.zfar * model.stat.extent
    pipe = Pipe()
    pipe.send((adapter.description.cameras[0].intrinsics, model.jnt_range[joints].copy()))
    try:
        while (request := pipe.recv()) is not None:
            place, fovy = request
            x, y, yaw, height, pan, tilt = place
            data.qpos[free : free + 3] = x, y, height
            data.qpos[free + 3 : free + 7] = math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)
            for joint, angle in zip(joints, (pan, tilt), strict=True):
                data.qpos[model.jnt_qposadr[joint]] = angle
            model.cam_fovy[adapter.camera_id] = fovy
            mujoco.mj_forward(model, data)
            camera = camera_matrix(adapter.camera_pose(data))
            renderer.update_scene(data, camera=adapter.camera_id, scene_option=option)
            renderer.disable_depth_rendering()
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            depth = renderer.render().astype(np.float32)
            depth[~(np.isfinite(depth) & (depth > 0) & (depth < far * 0.999))] = 0
            pipe.send((rgb, depth, camera))
    finally:
        renderer.close()


class Room:
    def __init__(self):
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self.intrinsics, self.limits = pickle.load(self.process.stdout)

    def view(self, place, fovy=65):
        pickle.dump((place, fovy), self.process.stdin)
        self.process.stdin.flush()
        return pickle.load(self.process.stdout)

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def points(depth, camera, fovy):
    y, x = np.indices(depth.shape)
    focal = SIZE[1] / (2 * math.tan(math.radians(fovy) / 2))
    local = np.stack(
        ((x + 0.5 - SIZE[0] / 2) * depth / focal, (SIZE[1] / 2 - y - 0.5) * depth / focal, -depth),
        axis=-1,
    )
    return local[depth > 0] @ camera[:3, :3].T + camera[:3, 3]


def unique(points_):
    _, indices = np.unique(np.floor(points_ / 0.005).astype(np.int32), axis=0, return_index=True)
    return points_[indices].astype(np.float32)


def alignment(estimated, truth):
    """Umeyama on timestamp-matched camera centres; no surface fitting hides ghosts."""
    if len(estimated) < 3:
        raise ValueError("Sim3 requires at least three tracked, matched camera poses")
    a, b = estimated[:, :3, 3].astype(float), truth[:, :3, 3].astype(float)
    ac, bc = a - a.mean(0), b - b.mean(0)
    if np.linalg.matrix_rank(ac, tol=1e-5) < 2 or np.linalg.matrix_rank(bc, tol=1e-5) < 2:
        raise ValueError(
            "Sim3 scale is unobservable: matched camera centres are collinear/stationary"
        )
    u, singular, vt = np.linalg.svd(bc.T @ ac / len(a))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    scale = float(singular @ sign / np.mean(np.sum(ac * ac, axis=1)))
    translation = b.mean(0) - scale * rotation @ a.mean(0)
    errors = np.linalg.norm(scale * a @ rotation.T + translation - b, axis=1)
    return (
        scale,
        rotation,
        translation,
        dict(
            method="Umeyama camera centres",
            matches=len(a),
            scale=scale,
            rotation=rotation.tolist(),
            translation_m=translation.tolist(),
            position_rmse_m=float(np.sqrt(np.mean(errors**2))),
            position_p95_m=float(np.percentile(errors, 95)),
        ),
    )


def aligned(records, estimated, truth, backend):
    if backend == "rgbd":
        return records.copy(), dict(
            method="identity: driver metric world",
            scale=1.0,
            rotation=np.eye(3).tolist(),
            translation_m=[0, 0, 0],
        )
    scale, rotation, translation, report = alignment(estimated, truth)
    result = records.copy()
    result[:, 0, :3] = scale * result[:, 0, :3] @ rotation.T + translation
    result[:, 1, :3] *= scale
    quats = Rotation.from_quat(result[:, 2, [1, 2, 3, 0]])
    result[:, 2] = (Rotation.from_matrix(rotation) * quats).as_quat()[:, [3, 0, 1, 2]]
    return result, report


def image_metrics(image, truth, alpha):
    a, b = image.astype(float) / 255, truth.astype(float) / 255
    mse = float(np.mean((a - b) ** 2))
    # Wang SSIM: Gaussian 11x11 window, sigma 1.5, population covariance, RGB mean.
    try:
        from skimage.metrics import structural_similarity

        ssim = structural_similarity(
            a,
            b,
            data_range=1,
            channel_axis=2,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    except ImportError:

        def blur(value):
            return gaussian_filter(value, (1.5, 1.5, 0), truncate=3.5)

        ma, mb = blur(a), blur(b)
        va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
        field = (
            (2 * ma * mb + 0.01**2)
            * (2 * cov + 0.03**2)
            / ((ma * ma + mb * mb + 0.01**2) * (va + vb + 0.03**2))
        )
        ssim = field[5:-5, 5:-5].mean()
    energy = [float(np.mean(laplace(v.mean(2))[1:-1, 1:-1] ** 2)) for v in (a, b)]
    return dict(
        psnr_db=float(-10 * np.log10(max(mse, 1e-15))),
        ssim=float(ssim),
        coverage=float(np.mean(alpha > 0.5)),
        sharpness_ratio=energy[0] / energy[1] if energy[1] else None,
    )


def geometry(centres, surface, tree, splat_tree):
    distance = tree.query(centres, workers=-1)[0] if len(centres) else np.array([])
    complete = (
        splat_tree.query(surface, workers=-1)[0] if splat_tree.n else np.full(len(surface), np.inf)
    )
    return dict(
        splats=len(centres),
        truth_points=len(surface),
        accuracy_median_m=float(np.median(distance)) if len(distance) else None,
        accuracy_p95_m=float(np.percentile(distance, 95)) if len(distance) else None,
        ghost_fraction=float(np.mean(distance > THRESHOLD)) if len(distance) else None,
        completeness=float(np.mean(complete <= THRESHOLD)),
    )


def score(snapshot, out, name, renderer, target, views):
    records, transform = aligned(
        snapshot["records"],
        snapshot["estimated"],
        snapshot["matched_truth"],
        str(snapshot["backend"]),
    )
    renderer.upload(GaussianBuffer(np.ascontiguousarray(records)))
    surface = snapshot["surface"]
    tree = cKDTree(surface)
    centres = records[:, 0, :3]
    splat_tree = cKDTree(centres)
    results, tiles = {}, []
    for index in range(len(snapshot["view_cameras"])):
        view_name = str(snapshot["view_names"][index])
        if views != "all" and view_name not in views.split(","):
            continue
        camera, fovy = snapshot["view_cameras"][index], snapshot["view_fovy"][index]
        truth, depth = snapshot["view_rgb"][index], snapshot["view_depth"][index]
        renderer.draw(
            camera,
            pose(),
            perspective(math.radians(fovy), SIZE[0] / SIZE[1]),
            target,
            clear=(0, 0, 0, 0),
        )
        pixels = np.frombuffer(target.read(components=4), np.uint8).reshape(SIZE[1], SIZE[0], 4)[
            ::-1
        ]
        alpha = pixels[..., 3] / 255
        image = np.uint8(
            np.clip(
                pixels[..., :3] + (1 - alpha[..., None]) * np.array((0.025, 0.035, 0.055)) * 255,
                0,
                255,
            )
        )
        local = (centres - camera[:3, 3]) @ camera[:3, :3]
        tangent = math.tan(math.radians(fovy) / 2)
        visible = (
            (local[:, 2] < 0)
            & (np.abs(local[:, 1]) < -local[:, 2] * tangent)
            & (np.abs(local[:, 0]) < -local[:, 2] * tangent * SIZE[0] / SIZE[1])
        )
        metrics = image_metrics(image, truth, alpha)
        metrics.update(geometry(centres[visible], points(depth, camera, fovy), tree, splat_tree))
        results[view_name] = metrics
        tile = Image.new("RGB", (SIZE[0] * 2, SIZE[1] + 24))
        tile.paste(Image.fromarray(truth), (0, 24))
        tile.paste(Image.fromarray(image), (SIZE[0], 24))
        ImageDraw.Draw(tile).text((5, 4), f"{view_name}: MuJoCo | Ito", fill="white")
        tiles.append(tile)
    if not results:
        raise ValueError("--views selected no views")
    sheet = Image.new("RGB", (SIZE[0] * 2, len(tiles) * (SIZE[1] + 24)))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (0, i * (SIZE[1] + 24)))
    sheet.save(out / f"{name}.png")
    overall = geometry(centres, surface, tree, splat_tree)
    for key in ("psnr_db", "ssim", "coverage", "sharpness_ratio"):
        values = [r[key] for r in results.values() if r[key] is not None]
        overall[key] = float(np.mean(values)) if values else None
    return dict(
        alignment=transform,
        overall=overall,
        views=results,
        integrated_frames=int(snapshot.get("integrated_frames", 0)),
        elapsed_seconds=float(snapshot.get("elapsed_seconds", 0)),
        restarts=json.loads(str(snapshot.get("restarts", "null"))),
    )


def scan(limits, rate, seconds):
    # Stationary base: full servo sweeps, followed by four quarter body turns.
    pan, tilt = limits
    x, y, heading = drive(0, seconds)
    knots = [(0, 0, 0), (0, pan[0], 0), (0, pan[1], 0), (0, 0, tilt[0]), (0, 0, tilt[1]), (0, 0, 0)]
    knots += [
        (yaw, p, t)
        for yaw in np.linspace(math.pi / 2, math.tau, 4)
        for p, t in ((0, 0), (pan[0], tilt[0]), (pan[1], tilt[1]), (0, 0))
    ]
    for t in np.linspace(0, len(knots) - 1, round(seconds * rate)):
        i = min(int(t), len(knots) - 2)
        yaw, p, tilt_ = np.array(knots[i]) * (1 - (t - i)) + np.array(knots[i + 1]) * (t - i)
        yield (x, y, heading + yaw, 0.19, p, tilt_)


def capture(args, room):
    records = np.zeros((args.budget, 4, 4), np.float32)
    surfaces, estimated, matched, submitted = [], [], [], {}
    last_tracked = 0
    matched_times = []
    with Reconstruction(room.intrinsics, backend=args.backend, max_splats=args.budget) as worker:

        def poll():
            nonlocal last_tracked
            changed = False
            while (packet := worker.poll()) is not None:
                records[packet.indices] = packet.records
                if packet.acknowledge:
                    packet.acknowledge()
                changed = True
            if args.backend == "slam" and (state := worker.pose()) is not None:
                camera, tracked, valid, captured = state
                if valid and tracked > last_tracked and captured in submitted:
                    estimated.append(camera)
                    matched.append(submitted[captured])
                    matched_times.append(captured - worker.epoch)
                last_tracked = tracked
            return changed

        phases = []
        if not args.no_scan:
            phases.append(("scan", scan(room.limits, args.rate, args.scan_seconds)))
        phases.append(
            (
                "drive",
                (
                    (*drive(t, args.seconds), 0.19, 0, 0)
                    for t in np.linspace(0, args.seconds, round(args.seconds * args.rate))
                ),
            )
        )
        for phase, route in phases:
            count = 0
            for place in route:
                rgb, depth, camera = room.view(place)
                surfaces.append(points(depth, camera, 65))
                if len(surfaces) >= 20:
                    surfaces = [unique(np.concatenate(surfaces))]
                before = worker.integrated.value
                stamp = clock.now()
                submitted[stamp] = camera
                deadline = time.monotonic() + 180
                while not worker.submit(
                    rgb,
                    depth if args.backend == "rgbd" else None,
                    camera if args.backend == "rgbd" else pose(),
                    captured_at=stamp,
                ):
                    poll()
                    if time.monotonic() > deadline:
                        raise RuntimeError("Worker input remained busy")
                    time.sleep(0.005)
                while worker.integrated.value == before or clock.now() - stamp < 1 / args.rate:
                    poll()
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"Worker timed out: {worker.status()}")
                    time.sleep(0.005)
                poll()
                count += 1
                if count % 40 == 0:
                    print(f"{phase}: {count} frames; {worker.status()}", flush=True)
            # Drain publication and retiring splats before freezing this scene.
            until = time.monotonic() + 1
            while time.monotonic() < until:
                poll()
                time.sleep(0.01)
            frozen = records.copy()
            frozen[:, 0, 3] *= np.clip((frozen[:, 1, 3] - (clock.now() - worker.epoch)) / 0.5, 0, 1)
            frozen = frozen[frozen[:, 0, 3] >= 1 / 255]
            names, cameras, colors, depths, fovys = [], [], [], [], []
            views = [
                (f"room-{i}", (x, y, yaw, 0.65, 0, 0), WIDE)
                for i, (x, y, yaw) in enumerate(
                    ((-2, -1, 0), (2, -1, math.pi / 2), (2, 1, math.pi), (-2, 1, -math.pi / 2))
                )
            ]
            views.append(("head", place, 65))
            for name, place_, fovy in views:
                rgb, depth, camera = room.view(place_, fovy)
                names.append(name)
                colors.append(rgb)
                depths.append(depth)
                cameras.append(camera)
                fovys.append(fovy)
                surfaces.append(points(depth, camera, fovy))
            surface = unique(np.concatenate(surfaces))
            surfaces = [surface]
            yield (
                phase,
                dict(
                    records=frozen,
                    backend=np.array(args.backend),
                    estimated=np.array(estimated),
                    matched_seconds=np.array(matched_times),
                    integrated_frames=np.array(worker.integrated.value),
                    elapsed_seconds=np.array(clock.now() - worker.epoch),
                    restarts=np.array(json.dumps(worker.restarts())),
                    matched_truth=np.array(matched),
                    surface=surface,
                    view_names=np.array(names),
                    view_cameras=np.array(cameras),
                    view_rgb=np.array(colors),
                    view_depth=np.array(depths),
                    view_fovy=np.array(fovys),
                ),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--backend", choices=("rgbd", "slam"), default="rgbd")
    parser.add_argument("--views", default="all", help="all or comma-separated room-0..room-3,head")
    parser.add_argument("--no-scan", action="store_true")
    parser.add_argument("--dump", type=Path, help="rescore the NPZ produced by this benchmark")
    parser.add_argument("--seconds", type=float, default=40)
    parser.add_argument("--scan-seconds", type=float, default=40)
    parser.add_argument("--rate", type=float, default=8)
    parser.add_argument("--budget", type=int, default=131_072)
    args = parser.parse_args()
    if Path(args.label).name != args.label or args.label in (".", ".."):
        parser.error("label must be a single directory name")
    timing = (args.seconds, args.scan_seconds, args.rate)
    if not np.isfinite(timing).all() or min(timing) <= 0:
        parser.error("durations and rate must be finite and positive")
    if min(args.seconds, args.scan_seconds) * args.rate < 2:
        parser.error("each route must contain at least two frames")
    if args.views != "all" and not set(args.views.split(",")) <= {
        "head",
        "room-0",
        "room-1",
        "room-2",
        "room-3",
    }:
        parser.error("unknown view name")
    out = Path("e2e/out/scene_truth") / args.label
    out.mkdir(parents=True, exist_ok=True)
    pygame.display.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((64, 64), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
    context = current_context()
    renderer = GaussianRenderer(context)
    target = context.simple_framebuffer(SIZE)
    import ito

    source = Path(ito.__file__).resolve().parent.parent
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    report = dict(
        label=args.label,
        revision=revision,
        source=str(source),
        settings=vars(args) | {"dump": str(args.dump) if args.dump else None},
        geometry_threshold_m=THRESHOLD,
        truth_voxel_m=0.005,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        geometry_support="input + evaluation depth union; per-view splats filtered by frustum",
        image_size=list(SIZE),
        wide_horizontal_fov_deg=90,
        aggregation="image metrics: view mean; geometry: pooled unique surface/centres",
        timing="every frame through real worker; paced at rate, slowed if worker is busy",
        phases={},
    )
    room = None
    snapshots = None
    try:
        if args.dump:
            with np.load(args.dump, allow_pickle=False) as dump:
                snapshots = [(args.dump.stem, dict(dump))]
        else:
            room = Room()
            snapshots = capture(args, room)
        for phase, snapshot in snapshots:
            if not args.dump:
                snapshot["source_revision"] = np.array(revision)
            np.savez_compressed(out / f"{phase}.npz", **snapshot)
            try:
                result = score(snapshot, out, phase, renderer, target, args.views)
            except ValueError as exc:
                if not str(exc).startswith("Sim3"):
                    raise
                # A stationary SLAM scan may not determine scale. Keep driving so
                # the translated trajectory can still be scored; never invent scale.
                result = dict(alignment={"error": str(exc)}, overall=None, views={})
                warning = Image.new("RGB", (SIZE[0] * 2, 80))
                ImageDraw.Draw(warning).text(
                    (8, 8), f"{phase}: alignment unavailable", fill="white"
                )
                ImageDraw.Draw(warning).text((8, 32), str(exc), fill="white")
                warning.save(out / f"{phase}.png")
            report["phases"][phase] = result
            report["phases"][phase]["reconstruction_revision"] = str(
                snapshot.get("source_revision", "unknown")
            )
            (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(phase, json.dumps(report["phases"][phase]["overall"]), flush=True)
        assert context.error == "GL_NO_ERROR", context.error
        if any(row["overall"] is None for row in report["phases"].values()):
            raise SystemExit("Some phases lack observable Sim3 alignment; see summary.json")
    finally:
        if snapshots is not None and hasattr(snapshots, "close"):
            snapshots.close()
        if room:
            room.close()
        renderer.close()
        target.release()
        pygame.quit()


if __name__ == "__main__":
    serve() if sys.argv[1:] == ["--serve"] else main()
