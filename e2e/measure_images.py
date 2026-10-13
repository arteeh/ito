"""Matched truth/GL eye sheets, common visibility and temporal inspection clips."""

import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from measure import MOMENTS
from measure_gl import View
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, sobel
from skimage.metrics import structural_similarity

WORLD = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])


def percentiles(values):
    return (
        {f"p{p:g}": float(np.percentile(values, p)) for p in (50, 95, 99, 99.9)}
        if len(values)
        else None
    )


def timing(values, hz=90):
    misses = np.asarray(values) > 1000 / hz
    longest = streak = 0
    for miss in misses:
        streak = streak + 1 if miss else 0
        longest = max(streak, longest)
    return dict(
        ms=percentiles(values),
        missed_fraction=float(misses.mean()),
        longest_miss_run=longest,
        samples=len(values),
        deadline_ms=1000 / hz,
    )


def edges(image):
    grey = image.mean(axis=2)
    return np.hypot(sobel(grey, axis=0), sobel(grey, axis=1)) > 0.3


class Metrics:
    def __init__(self):
        import lpips
        import torch

        torch.set_num_threads(4)
        self.torch = torch
        self.lpips = lpips.LPIPS(net="alex").eval()

    def score(self, truth, candidate, covered, seen):
        mse = float(np.mean((truth - candidate) ** 2))

        def tensor(image):
            return self.torch.from_numpy(image.copy()).permute(2, 0, 1)[None] * 2 - 1

        with self.torch.inference_mode():
            perceptual = float(self.lpips(tensor(truth), tensor(candidate)).item())
        reference_edges = edges(truth)
        edge_distance = distance_transform_edt(~edges(candidate))[reference_edges]
        return dict(
            psnr_db=float(-10 * np.log10(max(mse, 1e-15))),
            ssim=float(structural_similarity(truth, candidate, channel_axis=2, data_range=1)),
            lpips=perceptual,
            coverage=float(covered.mean()),
            common_visible_fraction=float(seen.mean()),
            previously_seen_holes=float((seen & ~covered).sum() / max(1, seen.sum())),
            never_seen_fraction=float((~seen).mean()),
            never_seen_covered_fraction=float((~seen & covered).mean()),
            edge_distance_px=percentiles(edge_distance),
        )


class Room:
    def __init__(self, source, intrinsics):
        import mujoco

        self.mj, self.i = mujoco, intrinsics
        self.model = m = mujoco.MjModel.from_binary_path(str(source))
        m.cam_bodyid[0] = 0
        m.cam_mode[0] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        m.cam_intrinsic[0] = 0
        m.cam_sensorsize[0] = 0
        m.cam_fovy[0] = np.degrees(2 * np.arctan(intrinsics.height / (2 * intrinsics.fy)))
        m.vis.global_.offwidth = m.vis.global_.offheight = max(intrinsics.width, intrinsics.height)
        self.data = mujoco.MjData(m)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[2], self.option.geomgroup[5] = 0, 1
        self.renderer = mujoco.Renderer(m, height=intrinsics.height, width=intrinsics.width)

    def draw(self, camera, depth=False):
        m, mj = self.model, self.mj
        m.cam_pos[0] = WORLD.T @ camera[:3, 3]
        mj.mju_mat2Quat(m.cam_quat[0], (WORLD.T @ camera[:3, :3]).ravel())
        mj.mj_forward(m, self.data)
        self.renderer.update_scene(self.data, camera=0, scene_option=self.option)
        self.renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = False
        if depth:
            self.renderer.enable_depth_rendering()
        result = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return result if depth else result.astype(np.float32) / 255

    def points(self, camera, depth):
        i = self.i
        y, x = np.mgrid[: i.height, : i.width]
        # MuJoCo uses pixel centres at n + 0.5 in the centred frustum.
        local = (
            np.stack(((x + 0.5 - i.cx) / i.fx, -(y + 0.5 - i.cy) / i.fy, -np.ones_like(x)), -1)
            * depth[..., None]
        )
        return local.reshape(-1, 3) @ camera[:3, :3].T + camera[:3, 3]

    def seen(self, camera, depth, history):
        points = self.points(camera, depth)
        visible = np.zeros(len(points), bool)
        i = self.i
        for pose, old_depth in reversed(history):
            remaining = np.flatnonzero(~visible)
            if not len(remaining):
                break
            local = (points[remaining] - pose[:3, 3]) @ pose[:3, :3]
            z = -local[:, 2]
            u = np.floor(i.fx * local[:, 0] / np.maximum(z, 1e-6) + i.cx).astype(int)
            v = np.floor(i.cy - i.fy * local[:, 1] / np.maximum(z, 1e-6)).astype(int)
            valid = (z > 0) & (u >= 0) & (u < i.width) & (v >= 0) & (v < i.height)
            ids = np.flatnonzero(valid)
            valid[ids] &= np.abs(old_depth[v[ids], u[ids]] - z[ids]) < 0.01 + z[ids] * 0.005
            visible[remaining[valid]] = True
        return visible.reshape(i.height, i.width)


def pair(truth, candidate, caption):
    pixels = (np.concatenate((truth, candidate), axis=1) * 255).clip(0, 255).astype(np.uint8)
    result = Image.new("RGB", (pixels.shape[1], pixels.shape[0] + 32), (24, 24, 24))
    result.paste(Image.fromarray(pixels), (0, 32))
    ImageDraw.Draw(result).text((5, 5), caption, fill="white")
    return result


def evaluate(work, out, ipd):
    from truth import align

    from ito.protocol import Intrinsics

    metrics = Metrics()
    report = dict(
        ipd_m=ipd,
        ipd_status="configured, not measured from a headset",
        baseline="316a556",
        moments=[],
        clips=[],
        timing_scope="offline GL draw completion, excludes readback; not scanout",
        freshness={
            "exposure_to_first_visible": None,
            "exposure_to_stable_sharp": None,
            "head_motion_to_display": None,
            "reason": "legacy exposure unknown; no compositor/photodiode timestamps",
        },
    )
    for run, moments in MOMENTS.items():
        recording = work / f"rf-run/out/{run}/rec"
        i = Intrinsics.model_validate_json((recording / "camera.json").read_text())
        # Compare both renderers with exactly the same centred frustum, retain original K
        # in the input manifest. Old camera cx differs by one pixel and is not exact truth.
        i = i.model_copy(update=dict(cx=i.width / 2, cy=i.height / 2))
        room = Room(work / "truth/data/scan/scene.mjb", i)
        view = View(i)
        report["gl_renderer"] = view.device
        poses = np.load(work / f"gs/out/{run}/data.npz")["poses"]
        plan = json.loads((out / f"{run}-schedule.json").read_text())
        history, history_frame = [], -1
        for index in sorted(moments):
            snap = np.load(out / run / f"snapshot-{index:06d}.npz")
            if int(snap["index"]) > index or float(snap["published"]) > float(snap["deadline"]):
                raise AssertionError("Noncausal prefix snapshot")
            # The mask depends only on the frozen input schedule and truth, not on how many
            # observations a candidate integrated or how many pixels it elected to render.
            for event in plan["admissions"]:
                f = event["frame"]
                if history_frame < f <= index and event["arrival"] <= float(snap["deadline"]):
                    history.append((poses[f], room.draw(poses[f], depth=True)))
                    history_frame = f
            world = align(snap["first_pose"], poses[int(snap["first_index"])])
            view.upload(snap["records"])
            eye_images = {}
            for label, offset in (
                ("camera", 0),
                ("left", -ipd / 2),
                ("right", ipd / 2),
                ("head-minus3cm", -0.03),
                ("head-plus3cm", 0.03),
            ):
                camera = poses[index].copy()
                camera[:3, 3] += camera[:3, 0] * offset
                truth, depth = room.draw(camera), room.draw(camera, depth=True)
                seen = room.seen(camera, depth, history)
                candidate, covered = view.draw(world @ camera, sort=label not in {"left", "right"})
                values = metrics.score(truth, candidate, covered, seen)
                row = dict(
                    run=run,
                    frame=index + 1,
                    view=label,
                    prefix_frame=int(snap["index"]) + 1,
                    **values,
                )
                report["moments"].append(row)
                caption = f"{run} #{index + 1} {label} | truth / main | {values['psnr_db']:.2f} dB"
                name = f"{run}-{index + 1:04d}-{label}"
                pair(truth, candidate, caption).save(out / f"{name}.jpg", quality=90)
                np.savez_compressed(out / f"{name}-mask.npz", seen=seen, covered=covered)
                eye_images[label] = (truth, candidate, depth)
                print(caption, flush=True)
            stereo = stereo_error(eye_images["left"], eye_images["right"], i, ipd)
            report["moments"][-1]["stereo"] = stereo
        if run == "sweep180":
            report["clips"] = clips(out, run, poses, room, view)
        report[f"{run}_draw"] = timing(view.draw_ms)
        report[f"{run}_upload_ms"] = percentiles(view.upload_ms)
        view.close()
        room.renderer.close()
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


def stereo_error(left, right, i, ipd):
    """Stereo block disparity against true geometric disparity; unmatched stays explicit."""
    import cv2

    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8 * 25,
        P2=32 * 25,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
    )
    images = [(im * 255).astype(np.uint8) for im in (left[1], right[1])]
    disparity = matcher.compute(*[cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in images]) / 16
    expected = i.fx * ipd / left[2]
    valid = (disparity > 0) & (expected < 128)
    return dict(
        disparity_error_px=percentiles(np.abs(disparity[valid] - expected[valid])),
        matched_fraction=float(valid.mean()),
        method="SGBM 5px, 128 disparities",
    )


def clips(out, run, poses, room, view):
    import imageio.v2 as imageio
    from truth import align

    frames = list(range(495, 556, 6))
    snaps = [np.load(out / run / f"snapshot-{f:06d}.npz") for f in frames]
    world = align(snaps[0]["first_pose"], poses[int(snaps[0]["first_index"])])
    results = []
    for kind in ("frozen-map-moving-head", "fixed-head-map-updates", "both"):
        path = out / f"{kind}.mp4"
        writer = imageio.get_writer(path, fps=30, macro_block_size=1)
        slow = imageio.get_writer(out / f"{kind}-slow.mp4", fps=10, macro_block_size=1)
        residuals, edge_changes, previous = [], [], None
        head_completion = []
        for n in range(61):
            k = min(n // 6, len(snaps) - 1)
            snap = snaps[0] if kind == "frozen-map-moving-head" else snaps[k]
            if n == 0 or (n % 6 == 0 and kind != "frozen-map-moving-head"):
                view.upload(snap["records"])
            camera = poses[525].copy()
            if kind != "fixed-head-map-updates":
                camera[:3, 3] += camera[:3, 0] * (0.03 * np.sin(n / 60 * 2 * np.pi))
            truth = room.draw(camera)
            began = time.perf_counter()
            candidate, _ = view.draw(world @ camera)
            head_completion.append((time.perf_counter() - began) * 1000)
            residual = candidate - truth
            if previous is not None:
                residuals.append(float(np.abs(residual - previous[0]).mean()))
                edge_changes.append(float(np.mean(edges(candidate) != previous[1])))
            previous = residual, edges(candidate)
            caption = f"{kind} #{frames[k] + 1} deadline {float(snap['deadline']):.3f}s"
            image = np.asarray(pair(truth, candidate, caption))
            writer.append_data(image)
            slow.append_data(image)
        writer.close()
        slow.close()
        results.append(
            dict(
                file=path.name,
                residual_change=percentiles(residuals),
                edge_changed_fraction=percentiles(edge_changes),
                head_sample_to_readback_ms=percentiles(head_completion),
            )
        )
    return results
