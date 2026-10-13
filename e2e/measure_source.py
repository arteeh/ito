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
    from ito.protocol import Intrinsics

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
    errors, matches = [], []
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
            matches.append(source)
    room.renderer.close()
    if set(matches) != set(range(len(rows))):
        raise AssertionError("Some source frames never decoded")
    (out / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return dict(
        frames=len(rows),
        exact_id_matches=len(matches),
        exact_fraction=1.0,
        codec_psnr_db=percentiles(errors),
        transport="H.264 MP4 PTS; not mediad/WebRTC",
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
