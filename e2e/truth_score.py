"""Score O1-O4 with source truth, actual Ito fusion/renderer and novel MuJoCo views.

python e2e/truth_score.py DATASET --replay REPLAY --out OUTPUT
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pygame
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from truth_metrics import alignment, image_metrics, transform_records, unproject
from truth_replay import snapshot
from truth_surface import Surface, wall_histogram

from ito.protocol import Intrinsics
from ito.reconstruction.rgbd import RGBDBackend
from ito.render import GaussianBuffer, GaussianRenderer, current_context, perspective


class Truth:
    def __init__(self, dataset):
        self.model = mujoco.MjModel.from_binary_path(str(dataset / "scene.mjb"))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.surface = Surface(self.model, self.data)
        self.camera = json.loads((dataset / "model.json").read_text())["camera_id"]
        self.model.cam_bodyid[self.camera] = 0
        self.model.cam_mode[self.camera] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = [1, 1, 0, 0, 1, 1]
        self.renderer = mujoco.Renderer(self.model, height=180, width=320)

    def render(self, camera, fovy):
        self.model.cam_pos[self.camera] = camera[:3, 3]
        mujoco.mju_mat2Quat(self.model.cam_quat[self.camera], camera[:3, :3].ravel())
        self.model.cam_fovy[self.camera] = fovy
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        return self.renderer.render().copy()


def views(rows):
    result = []
    roll = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    for index in np.linspace(0, len(rows) - 1, 5, dtype=int):
        index = int(index)
        raw = np.array(rows[index]["camera"])
        upright = raw.copy()
        upright[:3, :3] = raw[:3, :3] @ roll
        yield f"recorded-{index:06d}", raw, "source", index
        for label, offset, yaw, hfov in (
            ("desktop", 0, 0, 62),
            ("vr100", 0, 0, 100),
            ("offset10-yaw+30", 0.1, 30, 100),
            ("offset30-yaw-30", 0.3, -30, 100),
            ("offset50-yaw+30", 0.5, 30, 100),
            ("offset-30-yaw-30", -0.3, -30, 100),
        ):
            camera = upright.copy()
            right = upright[:3, 0].copy()
            right[2] = 0
            right /= max(np.linalg.norm(right), 1e-6)
            camera[:3, 3] += right * offset
            camera[:3, :3] = (
                Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ camera[:3, :3]
            )
            vfov = math.degrees(2 * math.atan(math.tan(math.radians(hfov) / 2) * 180 / 320))
            result.append((f"{label}-{index:06d}", camera, vfov, index))
    yield from result


def draw(renderer, target, records, camera, fovy):
    renderer.upload(GaussianBuffer(np.ascontiguousarray(records)))
    renderer.draw(
        camera, np.eye(4), perspective(math.radians(fovy), 320 / 180), target, clear=(0, 0, 0, 0)
    )
    pixels = np.frombuffer(target.read(components=4), np.uint8).reshape(180, 320, 4)[::-1]
    alpha = pixels[..., 3] / 255
    image = np.uint8(
        np.clip(
            pixels[..., :3] + (1 - alpha[..., None]) * np.array([0.025, 0.035, 0.055]) * 255, 0, 255
        )
    )
    return image, alpha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=args.score_only)
    rows = [json.loads(s) for s in (args.dataset / "frames.jsonl").read_text().splitlines()]
    estimates = [json.loads(s) for s in (args.replay / "frames.jsonl").read_text().splitlines()]
    ids = [r["sequence"] for r in estimates if r["tracked"]]
    estimated = np.array([r["camera"] for r in estimates if r["tracked"]])
    cameras = np.array([rows[i]["camera"] for i in ids])
    scale, rotation, translation, aligned, fit = alignment(estimated, cameras)
    camera_by_id = dict(zip(ids, aligned, strict=True))
    (args.out / "alignment.json").write_text(json.dumps(fit, indent=2))
    intrinsics = Intrinsics.model_validate_json((args.dataset / "camera.json").read_text())
    k = np.array([[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]])
    source_fovy = math.degrees(2 * math.atan(intrinsics.height / (2 * intrinsics.fy)))
    targets = {int(p.stem) for p in (args.replay / "snapshots").glob("*.npz")}
    truth = Truth(args.dataset)
    import torch  # Load shipped CUDA DLLs before CuPy compiles kernels.

    if not args.score_only:
        backends = {
            label: RGBDBackend(1_048_576, intrinsics, device="cuda") for label in ("O1", "O2", "O3")
        }
        for label in backends:
            (args.out / label).mkdir()
        observed = []
        for i, row in enumerate(rows[: len(estimates)]):
            now = row["sim_time"] - rows[0]["sim_time"]
            if i in camera_by_id:
                rgb = np.asarray(Image.open(args.dataset / row["rgb"]))[::2, ::2]
                depth = np.load(args.dataset / row["depth"])[::2, ::2]
                small_k = k.copy()
                small_k[:2] /= 2
                valid = (depth > 0) & (
                    depth < truth.model.vis.map.zfar * truth.model.stat.extent * 0.99
                )
                data = np.load(args.replay / "frames" / f"{i:06d}.npz")
                predicted = data["depth"] * scale
                pred_k = data["K"]
                colors = cv2.resize(
                    np.asarray(Image.open(args.dataset / row["rgb"])),
                    (predicted.shape[1], predicted.shape[0]),
                )
                # Recover exact cropped image colours from exported valid point order.
                colors[predicted > 0] = data["colors"]
                for label, d, calibration, camera, color, mask in (
                    ("O1", depth, small_k, np.array(row["camera"]), rgb, valid),
                    ("O2", predicted, pred_k, np.array(row["camera"]), colors, predicted > 0),
                    ("O3", depth, small_k, camera_by_id[i], rgb, valid),
                ):
                    backend = backends[label]
                    cloud = unproject(d, calibration, camera)[mask]
                    if label == "O1" and i % 5 == 0:
                        observed.append(cloud[::20])
                    xp = backend.xp
                    backend.integrate_points(
                        xp.asarray(cloud),
                        xp.asarray(color[mask]),
                        now,
                        xp.asarray(d[mask] * 3 / calibration[0, 0]),
                    )
                    backend.carve(
                        camera,
                        *[calibration[a, b] for a, b in ((0, 0), (1, 1), (0, 2), (1, 2))],
                        xp.asarray(np.where(mask, d, 0).astype(np.float32)),
                        now,
                    )
            for label, backend in backends.items():
                backend.expire(now, np.inf)
                if i in targets:
                    np.savez_compressed(
                        args.out / label / f"{i:06d}.npz", records=snapshot(backend, now)
                    )
            if i % 100 == 0:
                print(f"oracle fusion {i}/{len(estimates)}", flush=True)
        surface_points = np.concatenate(observed)
        np.savez_compressed(args.out / "observed-truth.npz", points=surface_points)
    else:
        surface_points = np.load(args.out / "observed-truth.npz")["points"]
    all_views = list(views(rows[: len(estimates)]))
    truths = {}
    for name, camera, fovy, _index in all_views:
        fovy = source_fovy if fovy == "source" else fovy
        truths[name] = truth.render(camera, fovy)
    truth.renderer.close()
    pygame.display.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((320, 180), pygame.OPENGL | pygame.DOUBLEBUF)
    context = current_context()
    renderer = GaussianRenderer(context)
    target = context.simple_framebuffer((320, 180), components=4)
    lpips_model, lpips_error = None, None
    try:
        import lpips
        import torch

        lpips_model = lpips.LPIPS(net="alex").eval().to("cuda")
    except Exception as exc:
        lpips_error = repr(exc)
    output = dict(
        alignment=fit,
        description="Matched tracked frames; O1/O3 half-resolution true depth; "
        "O2 exported calibrated MASt3R depth; O4 shipped online fusion snapshots. "
        "Final-map views, not causal historical views. Completeness on observed source surfaces.",
        lpips_error=lpips_error,
        renderer=context.info["GL_RENDERER"],
        ladder={},
    )
    for label in ("O1", "O2", "O3", "O4"):
        paths = sorted(
            (args.replay / "snapshots" if label == "O4" else args.out / label).glob("*.npz")
        )

        def load(path, label=label):
            r = np.load(path)["records"]
            return transform_records(r, scale, rotation, translation) if label == "O4" else r

        records = load(paths[-1])
        centres = records[:, 0, :3]
        geometry = truth.surface.metrics(centres)
        geometry["completeness"] = float(
            np.mean(
                cKDTree(centres).query(surface_points, distance_upper_bound=0.050001)[0] <= 0.05
            )
        )
        geometry["old_sample_union_ghost_fraction"] = float(
            np.mean(cKDTree(surface_points).query(centres, distance_upper_bound=0.050001)[0] > 0.05)
        )
        geometry["walls"] = {
            name: wall_histogram(centres, axis=axis, face=face, bounds=bounds, inward=inward)
            for name, axis, face, bounds, inward in (
                ("east", 0, 2.7, ((-1, 1), (1.65, 2.1)), -1),
                ("west", 0, -1.3, ((-1, 1), (1.65, 2.1)), 1),
                ("north", 1, 1.8, ((-0.9, 2.3), (1.9, 2.2)), -1),
                ("south", 1, -1.8, ((-0.9, 2.3), (1.9, 2.2)), 1),
            )
        }
        metrics = []
        for name, camera, fovy, index in all_views:
            fovy = source_fovy if fovy == "source" else fovy
            image, alpha = draw(renderer, target, records, camera, fovy)
            reference = truths[name]
            m = image_metrics(image, reference, alpha)
            if lpips_model is not None:
                with torch.inference_mode():
                    tensors = [
                        torch.from_numpy(a.copy()).permute(2, 0, 1)[None].float().to("cuda") / 127.5
                        - 1
                        for a in (image, reference)
                    ]
                    m["lpips"] = float(lpips_model(*tensors).item())
            m.update(view=name, source_sequence=index)
            metrics.append(m)
            error = np.uint8(np.clip(np.abs(image.astype(float) - reference) * 3, 0, 255))
            sheet = Image.new("RGB", (960, 204))
            for column, tile in enumerate((reference, image, error)):
                sheet.paste(Image.fromarray(tile), (column * 320, 24))
            ImageDraw.Draw(sheet).text((5, 5), f"{label} {name}: truth | Ito | absolute error x3")
            sheet.save(args.out / f"{label}-{name}.png")
        temporal = []
        fixed = all_views[0][1]
        previous = None
        for path in paths:
            image, _ = draw(renderer, target, load(path), fixed, source_fovy)
            if previous is not None:
                temporal.append(float(np.mean(np.abs(image.astype(float) - previous)) / 255))
            previous = image.astype(float)
        means = {
            key: float(np.mean([m[key] for m in metrics]))
            for key in ("psnr_db", "ssim", "coverage", "sharpness_ratio")
        }
        if lpips_model is not None:
            means["lpips"] = float(np.mean([m["lpips"] for m in metrics]))
        output["ladder"][label] = dict(
            geometry=geometry,
            appearance=means,
            views=metrics,
            temporal=dict(
                fixed_view_mean_abs_change=float(np.mean(temporal)),
                changes=temporal,
                source_sequences=[int(p.stem) for p in paths],
                description="Static truth, fixed first camera, consecutive map snapshots; "
                "includes newly reconstructed coverage, not motion-compensated video.",
            ),
        )
        (args.out / "summary.json").write_text(json.dumps(output, indent=2))
        print(label, geometry["ghost_fraction"], means, flush=True)
    target.release()
    renderer.close()
    context.release()
    pygame.quit()


if __name__ == "__main__":
    main()
