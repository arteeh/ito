"""Headless static-room capture through H.264, exact PTS identity, and stereo calibration.

python e2e/measure_source.py --work .. --out ../m1-out/source
No simulator daemon, shared environment or desktop input is used.
"""

import argparse
import json
import os
import sys
import time
from fractions import Fraction

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import av
import numpy as np
from measure import MOMENTS
from measure_gl import View
from measure_images import Room, pair, percentiles
from PIL import Image


def calibration(out):
    import mujoco

    from ito.protocol import Intrinsics

    i = Intrinsics(width=360, height=640, fx=532.5694, fy=532.5694, cx=180.0, cy=320.0)
    points = np.array(
        [(x, y, -z) for z in (1.0, 2.0, 4.0) for x, y in ((-0.2, -0.3), (0.0, 0.1), (0.18, 0.35))]
    )
    geoms = "".join(
        f'<geom type="sphere" size="0.003" pos="{x} {y} {z}" rgba="1 1 1 1"/>' for x, y, z in points
    )
    xml = f'''<mujoco><visual><global offwidth="360" offheight="640"/>
    <headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>
    <worldbody><camera name="calibration" pos="0 0 0"
    fovy="{np.degrees(2 * np.arctan(i.height / (2 * i.fy)))}"/>{geoms}</worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=i.height, width=i.width)
    view = View(i)
    records = np.zeros((len(points), 4, 4), np.float32)
    records[:, 0, :3], records[:, 0, 3] = points, 1
    records[:, 1, :3], records[:, 2, 0] = 0.003, 1
    records[:, 3, :3] = 0.5 / 0.2820947918
    view.upload(records)
    residuals = {"truth": [], "GL": []}
    for label, offset in (
        ("camera", 0),
        ("left", -0.032),
        ("right", 0.032),
        ("minus3cm", -0.03),
        ("plus3cm", 0.03),
    ):
        camera = np.eye(4)
        camera[0, 3] = offset
        model.cam_pos[0, 0] = offset
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=0)
        truth = renderer.render().astype(np.float32) / 255
        candidate, _ = view.draw(camera)
        for name, image in (("truth", truth), ("GL", candidate)):
            for x, y, z in points:
                u, v = i.fx * (x - offset) / -z + i.cx - 0.5, i.cy - i.fy * y / -z - 0.5
                x0, y0 = int(round(u)), int(round(v))
                patch = image[y0 - 4 : y0 + 5, x0 - 4 : x0 + 5].mean(axis=2)
                yy, xx = np.mgrid[y0 - 4 : y0 + 5, x0 - 4 : x0 + 5]
                if patch.sum() <= 0:
                    raise AssertionError(f"Missing calibration marker: {name} {label}")
                centre = ((xx * patch).sum() / patch.sum(), (yy * patch).sum() / patch.sum())
                residuals[name].append(float(np.linalg.norm(np.array(centre) - (u, v))))
        pair(truth, candidate, f"calibration {label}: MuJoCo / production GL").save(
            out / f"calibration-{label}.png"
        )
    view.close()
    renderer.close()
    report = {name: percentiles(values) for name, values in residuals.items()}
    report["pass"] = all(v["p50"] <= 0.5 and v["p95"] <= 1 for v in report.values())
    return report


def capture(work, out):
    import cv2
    from skimage.metrics import structural_similarity

    from ito.protocol import Intrinsics
    from ito.reconstruction.rgbd import RGBDBackend

    k = Intrinsics(
        width=360, height=640, fx=532.5694343521658, fy=532.5694343521658, cx=180.0, cy=320.0
    )
    room = Room(work / "truth/data/scan/scene.mjb", k)
    (out / "camera.json").write_text(k.model_dump_json())
    encoder = av.open(str(out / "video.mp4"), "w")
    stream = encoder.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = k.width, k.height, "yuv420p"
    stream.options = {"crf": "18", "preset": "fast", "tune": "zerolatency"}
    rows = []
    for run, moments in MOMENTS.items():
        poses = np.load(work / f"gs/out/{run}/data.npz")["poses"]
        for centre in moments:
            for index in range(centre - 2, centre + 3):
                source = len(rows)
                exposed = time.perf_counter()
                rgb = (room.draw(poses[index]) * 255).round().astype(np.uint8)
                Image.fromarray(rgb).save(out / f"truth-{source:06d}.png")
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts, frame.time_base = source * 3000, Fraction(1, 90000)
                for packet in stream.encode(frame):
                    encoder.mux(packet)
                rows.append(
                    dict(
                        source_frame_id=source,
                        video_pts=source * 3000,
                        exposure_time=exposed,
                        exposure_status="simulated instantaneous",
                        exposure_duration=0.0,
                        readout_status="global",
                        readout_duration=0.0,
                        arrival=None,
                        publication=None,
                        display=None,
                        publication_status="not reconstructed",
                        display_status="not displayed",
                        camera_model={"model": "pinhole", **k.model_dump()},
                        coordinate_revision=0,
                        calibration_revision=0,
                        camera=poses[index].ravel().tolist(),
                        original_run=run,
                        original_frame=index + 1,
                    )
                )
    for packet in stream.encode():
        encoder.mux(packet)
    encoder.close()
    by_pts = {r["video_pts"]: r for r in rows}
    errors, matches, visible_ms, sharp_ms, head_ms = [], [], [], [], []
    alignment_errors = []
    view = View(k)
    backend = RGBDBackend(131072, k, device="cpu")
    with av.open(str(out / "video.mp4")) as video:
        for decoded in video.decode(video=0):
            pts = round(decoded.pts * decoded.time_base * 90000)
            if pts not in by_pts:
                raise AssertionError(f"Unknown source PTS {pts}")
            row = by_pts[pts]
            source = row["source_frame_id"]
            if source in matches:
                raise AssertionError("Duplicate decoded source identity")
            row["arrival"] = time.perf_counter()
            row["arrival_status"] = "offline decoder completion on source monotonic clock"
            rgb = decoded.to_ndarray(format="rgb24")
            row["name"] = f"{source:06d}.png"
            Image.fromarray(rgb).save(out / row["name"])
            truth = np.array(Image.open(out / f"truth-{source:06d}.png"))
            errors.append(float(-10 * np.log10(np.mean(((rgb.astype(float) - truth) / 255) ** 2))))
            grey = cv2.cvtColor(truth, cv2.COLOR_RGB2GRAY)
            corners = cv2.goodFeaturesToTrack(grey, 100, 0.02, 8)
            if corners is not None:
                moved, status, _ = cv2.calcOpticalFlowPyrLK(
                    grey,
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                    corners,
                    None,
                    winSize=(15, 15),
                    maxLevel=2,
                )
                alignment_errors.extend(
                    np.linalg.norm((moved - corners)[status[:, 0] == 1, 0], axis=1).tolist()
                )
            camera = np.array(row["camera"]).reshape(4, 4)
            depth = room.draw(camera, depth=True)
            observed_at = time.perf_counter()
            backend.integrate(rgb, depth, camera, observed_at)
            row["publication"] = time.perf_counter()
            row["publication_status"] = "offline RGB-D oracle completed, source monotonic clock"
            live = backend.keys >= 0
            records = backend.records[live].copy()
            view.upload(records)
            stable = 0
            for _ in range(3):
                image, _ = view.draw(camera)
                complete = time.perf_counter()
                if row["display"] is None:
                    row["display"] = complete
                reference = truth.astype(np.float32) / 255
                psnr = -10 * np.log10(max(float(np.mean((image - reference) ** 2)), 1e-15))
                ssim = structural_similarity(reference, image, channel_axis=2, data_range=1)
                stable = stable + 1 if psnr >= 30 and ssim >= 0.9 else 0
            row["display_status"] = "offscreen GL completion including readback; not scanout"
            row["stable_sharp"] = complete if stable == 3 else None
            row["stable_sharp_status"] = (
                "three draws at PSNR>=30, SSIM>=0.9" if stable == 3 else "censored"
            )
            if stable == 3:
                sharp_ms.append((complete - row["exposure_time"]) * 1000)
            row["head_motion"] = time.perf_counter()
            head_camera = camera.copy()
            head_camera[:3, 3] += camera[:3, 0] * (0.03 if source % 2 else -0.03)
            view.draw(head_camera)
            row["head_motion_display"] = time.perf_counter()
            head_ms.append((row["head_motion_display"] - row["head_motion"]) * 1000)
            # An ID-colour pass tests whether this observation contributed after compositing
            # against older surfaces. This diagnostic pass is not a displayed candidate.
            current = backend.seen[live] == observed_at
            records[:, 3, :3] = (current[:, None].astype(np.float32) - 0.5) / 0.2820947918
            view.upload(records)
            contribution, _ = view.draw(camera)
            row["visible_source_pixels"] = int((contribution[..., 0] > 0.1).sum())
            row["first_visible"] = row["display"] if row["visible_source_pixels"] else None
            if row["first_visible"] is not None:
                visible_ms.append((row["first_visible"] - row["exposure_time"]) * 1000)
            matches.append(source)
    view.close()
    room.renderer.close()
    if set(matches) != set(range(len(rows))):
        raise AssertionError("Some source frames never decoded")
    (out / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return dict(
        frames=len(rows),
        exact_id_matches=len(matches),
        exact_fraction=1.0,
        codec_psnr_db=percentiles(errors),
        decoded_alignment_px=percentiles(alignment_errors),
        transport="H.264 MP4 PTS; not mediad/WebRTC",
        freshness_scope="offline RGB-D oracle incl. recording/decoding delay, not live latency",
        exposure_to_first_visible_ms=percentiles(visible_ms),
        exposure_to_stable_sharp_ms=percentiles(sharp_ms),
        stable_sharp_censored=len(rows) - len(sharp_ms),
        head_sample_to_GL_complete_ms=percentiles(head_ms),
    )


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(capture=capture(args.work, args.out), calibration=calibration(args.out))
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["calibration"]["pass"]:
        raise SystemExit("Calibration gate failed")


if __name__ == "__main__":
    sys.exit(main())
