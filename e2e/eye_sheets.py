"""Revisions side by side at the same frames: ground truth, then each revision's scene.

python e2e/eye_sheets.py --work WORK --out OUT --column LABEL=PATTERN ... [--ito DIR]

PATTERN is a folder under WORK with {run} and {rung} in it, holding the snapshot-*.npz a
`slam_replay --eye LABEL --eye-at FRAMES` (or research/big-picture's big_picture_capture.py)
saved: e.g. d91e443=eye/d91e443-{run}-{rung}. Per recording and rung it writes one sheet
OUT/{run}-{rung}.jpg (under 1.5 MB) and one PNG per frame: the static room re-rendered by
MuJoCo at the frame's true camera pose (research/big-picture's method, saved scene
truth/data/scan/scene.mjb), then every column's scene drawn by Ito's renderer from that
pose, with its PSNR against the re-render. The k-th snapshot (by capture offset) of a
column is drawn at the k-th frame of FRAMES; truth goes into Ito's world by the first
tracked frame, as slam_replay's scores do. Retiring point splats are left out (the replay
never fades them). --ito: the checkout whose renderer draws (one with the antialiased
Gaussian path draws point maps unchanged and optimised maps as the pilot would).

Needs numpy, pillow, mujoco==3.3.7, moderngl (EGL), PyOpenGL, scipy, pydantic. CPU only
(llvmpipe) is fine.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# research/big-picture's frames: three moments per recording (1-based frame numbers).
FRAMES = {"sweep180": (215, 453, 693), "tour": (363, 1501, 2284)}
WORLD = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])  # Ito world to MuJoCo's
WIDTH, HEIGHT = 360, 640
SHEET_BYTES = 1_500_000


def psnr(a, b):
    return float(-10 * np.log10(max(np.mean(((a.astype(float) - b) / 255) ** 2), 1e-15)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--column", action="append", required=True, metavar="LABEL=PATTERN")
    parser.add_argument("--ito", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs", default="sweep180,tour")
    parser.add_argument("--rungs", default="L1,L4")
    args = parser.parse_args()
    sys.path.insert(0, str(args.ito))
    import moderngl
    import mujoco
    from truth import align

    from ito.render.renderer import GaussianRenderer
    from ito.render.scene import GaussianBuffer

    columns = [c.split("=", 1) for c in args.column]
    work, out = args.work, args.out
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_binary_path(str(work / "truth/data/scan/scene.mjb"))
    model.cam_bodyid[0] = 0
    model.cam_mode[0] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    model.cam_intrinsic[0] = 0
    model.cam_sensorsize[0] = 0
    model.vis.global_.offwidth = model.vis.global_.offheight = HEIGHT
    data = mujoco.MjData(model)
    option = mujoco.MjvOption()
    option.geomgroup[2], option.geomgroup[5] = 0, 1
    truth_renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    context = moderngl.create_standalone_context(require=430, backend="egl")
    with context:
        renderer = GaussianRenderer(context)
        target = context.simple_framebuffer((WIDTH, HEIGHT), components=4)
        # moderngl reads its limits once: here, where its own context is current.
        report = dict(ito=str(args.ito), gl_renderer=context.info["GL_RENDERER"], sheets=[])
    font = ImageFont.load_default(size=15)

    def draw(records, antialiased, camera, projection):
        options = {}
        if "antialiased" in GaussianBuffer.__dataclass_fields__:
            options["antialiased"] = antialiased
        # MuJoCo has a GL context of its own: make Ito's current for every draw.
        with context:
            renderer.upload(GaussianBuffer(np.ascontiguousarray(records, np.float32), **options))
            renderer.draw(
                camera.astype(np.float32),
                np.eye(4, dtype=np.float32),
                projection,
                target,
                clear=(0, 0, 0, 1),
            )
            context.finish()
            pixels = np.frombuffer(target.read(components=3), np.uint8)
        return pixels.reshape(HEIGHT, WIDTH, 3)[::-1].copy()

    for run in args.runs.split(","):
        frames = [f - 1 for f in FRAMES[run]]
        poses = np.load(work / f"gs/out/{run}/data.npz")["poses"]
        intrinsics = json.loads((work / f"rf-run/out/{run}/rec/camera.json").read_text())
        model.cam_fovy[0] = np.degrees(2 * np.arctan(HEIGHT / (2 * intrinsics["fy"])))
        # The re-render has a centred principal point; both panels use exactly that.
        projection = np.zeros((4, 4), np.float32)
        projection[0, 0] = 2 * intrinsics["fx"] / WIDTH
        projection[1, 1] = 2 * intrinsics["fy"] / HEIGHT
        projection[2, 2], projection[2, 3], projection[3, 2] = -1.00002, -0.0200002, -1
        truths = []
        for index in frames:
            camera = poses[index]
            model.cam_pos[0] = WORLD.T @ camera[:3, 3]
            mujoco.mju_mat2Quat(model.cam_quat[0], (WORLD.T @ camera[:3, :3]).ravel())
            mujoco.mj_forward(model, data)
            truth_renderer.update_scene(data, camera=0, scene_option=option)
            truth_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            truths.append(truth_renderer.render().copy())
        for rung in args.rungs.split(","):
            rows = []
            for k, index in enumerate(frames):
                panels = [(truths[k], "GROUND TRUTH (sim re-render)")]
                entry = dict(run=run, rung=rung, frame=index + 1, columns={})
                for label, pattern in columns:
                    folder = work / pattern.format(run=run, rung=rung)
                    snaps = sorted(
                        (np.load(p) for p in folder.glob("snapshot-*.npz")),
                        key=lambda s: float(s["offset"]) if "offset" in s else 0.0,
                    )
                    if len(snaps) != len(frames):
                        raise SystemExit(f"{folder}: {len(snaps)} snapshots, not {len(frames)}")
                    snap = snaps[k]
                    world = align(snap["first_pose"], poses[int(snap["first_index"])])
                    records = snap["records"]
                    records = records[records[:, 3, 3] != 1]
                    antialiased = bool(snap["antialiased"]) if "antialiased" in snap else False
                    seen = draw(records, antialiased, world @ poses[index], projection)
                    score = psnr(truths[k], seen)
                    entry["columns"][label] = dict(
                        snapshot_frame=int(snap["index"]) + 1, splats=len(records), psnr=score
                    )
                    panels.append((seen, f"{label}  {score:.1f} dB  {len(records):,}"))
                rows.append(panels)
                row = Image.fromarray(np.concatenate([p for p, _ in panels], 1))
                row.save(out / f"{run}-{rung}-frame{index + 1:04d}.png")
                report["sheets"].append(entry)
            sheet = Image.new(
                "RGB", (WIDTH * len(rows[0]), len(rows) * (HEIGHT + 24)), (25, 28, 33)
            )
            pen = ImageDraw.Draw(sheet)
            for r, panels in enumerate(rows):
                y = r * (HEIGHT + 24)
                for c, (pixels, title) in enumerate(panels):
                    pen.text((c * WIDTH + 6, y + 4), title, font=font, fill="white")
                    sheet.paste(Image.fromarray(pixels), (c * WIDTH, y + 24))
                pen.text(
                    (6, y + 30),
                    f"{run} {rung} frame {frames[r] + 1}",
                    font=font,
                    fill="yellow",
                    stroke_width=2,
                    stroke_fill="black",
                )
            for quality in (90, 80, 70, 60):
                path = out / f"{run}-{rung}.jpg"
                sheet.save(path, quality=quality)
                if path.stat().st_size < SHEET_BYTES:
                    break
    (out / "sheets.json").write_text(json.dumps(report, indent=2) + "\n")
    truth_renderer.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
