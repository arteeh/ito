"""Render truth-left / Ito-right review sheets without a live simulator or window.

Under the GPU lock, with numpy, pillow, mujoco==3.3.7, moderngl, PyOpenGL,
plyfile and scipy installed:
python e2e/big_picture_images.py --work /path/to/work --out docs/big-picture/img

Reads bigpic-out capture snapshots and existing gs checkpoints. MuJoCo's saved
static room is re-rendered at the recorded true camera pose. Snapshot publication
and headset timing are deliberately not simulated. Images are geometry/appearance
diagnostics; the JSON preserves inputs, times and reference disagreement.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import moderngl
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ito.render.renderer import GaussianRenderer
from ito.render.scene import GaussianBuffer, load_ply
from truth import align

WORLD = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def psnr(a, b):
    return float(-10 * np.log10(max(np.mean(((a.astype(float) - b) / 255) ** 2), 1e-15)))


def bake_color(records, eye):
    """Evaluate the renderer's SH at this one viewpoint to fit Mesa's SSBO limit."""
    if records.shape[1] == 4:
        return records.copy()
    direction = records[:, 0, :3] - eye
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    x, y, z = direction.T
    basis = [
        np.full(len(x), 0.2820947918), -0.4886025119 * y, 0.4886025119 * z,
        -0.4886025119 * x, 1.0925484306 * x * y, -1.0925484306 * y * z,
        0.3153915653 * (2*z*z-x*x-y*y), -1.0925484306 * x * z,
        0.5462742153 * (x*x-y*y), -0.5900435899 * y * (3*x*x-y*y),
        2.8906114426 * x * y * z, -0.4570457995 * y * (4*z*z-x*x-y*y),
        0.3731763326 * z * (2*z*z-3*x*x-3*y*y),
        -0.4570457995 * x * (4*z*z-x*x-y*y), 1.4453057213 * z * (x*x-y*y),
        -0.5900435899 * x * (x*x-3*y*y),
    ]
    color = np.full((len(x), 3), 0.5, np.float32)
    for k in range(records.shape[1] - 3):
        color += basis[k][:, None] * records[:, 3+k, :3]
    packed = np.ascontiguousarray(records[:, :4])
    packed[:, 3, :3] = (np.maximum(color, 0) - 0.5) / 0.2820947918
    return packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    work, out = args.work, args.out
    out.mkdir(parents=True, exist_ok=True)
    raw = work / "bigpic-out/rendered"
    raw.mkdir(exist_ok=True)
    source = work / "truth/data/scan/scene.mjb"
    model = mujoco.MjModel.from_binary_path(str(source))
    model.cam_bodyid[0] = 0
    model.cam_mode[0] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    model.cam_intrinsic[0] = 0
    model.cam_sensorsize[0] = 0
    model.vis.global_.offwidth = model.vis.global_.offheight = 640
    data = mujoco.MjData(model)
    option = mujoco.MjvOption()
    option.geomgroup[2], option.geomgroup[5] = 0, 1
    truth_renderer = mujoco.Renderer(model, height=640, width=360)
    ctx = moderngl.create_standalone_context(require=430, backend="egl")
    renderer = GaussianRenderer(ctx)
    fbo = ctx.simple_framebuffer((360, 640), components=4)
    font = ImageFont.load_default(size=16)
    report = {"code_revision": "fff04390daa945af16ddf7f97b7afc0ae8118b88",
              "method": "Static room, matched true camera, 360x640 centred projection; "
                        "active main slots, no publication timing; final offline GS checkpoints; "
                        "view-dependent SH evaluated before upload without dropping terms.",
              "scene": str(source.relative_to(work)), "scene_sha256": digest(source),
              "gl_renderer": ctx.info["GL_RENDERER"], "images": []}

    def render(records, camera, projection):
        with ctx:
            renderer.upload(GaussianBuffer(bake_color(records, camera[:3, 3])))
            renderer.draw(camera.astype(np.float32), np.eye(4, dtype=np.float32), projection,
                          fbo, clear=(0, 0, 0, 1))
            ctx.finish()
            return np.frombuffer(fbo.read(components=3), np.uint8).reshape(640, 360, 3)[::-1].copy()

    def sheet(name, pairs):
        image = Image.new("RGB", (720, len(pairs) * 688), (25, 28, 33))
        draw = ImageDraw.Draw(image)
        for n, (left, right, title) in enumerate(pairs):
            y = n * 688
            draw.text((8, y+3), title, font=font, fill="white")
            draw.text((8, y+25), "GROUND TRUTH", font=font, fill="white")
            draw.text((368, y+25), "ITO / SAME TRUE CAMERA", font=font, fill="white")
            image.paste(Image.fromarray(left), (0, y+48))
            image.paste(Image.fromarray(right), (360, y+48))
        image.save(out / f"{name}.jpg", quality=91, subsampling=0)
        assert (out / f"{name}.jpg").stat().st_size < 1_000_000

    for run, moments in [("sweep180", [6, 15, 23]), ("tour", [12, 50, 76])]:
        recording = work / f"rf-run/out/{run}/rec"
        rows = [json.loads(s) for s in (recording / "frames.jsonl").read_text().splitlines()]
        truth_data = np.load(work / f"gs/out/{run}/data.npz")
        poses = truth_data["poses"]
        intr = json.loads((recording / "camera.json").read_text())
        model.cam_fovy[0] = np.degrees(2*np.arctan(640/(2*intr["fy"])))
        projection = np.zeros((4, 4), np.float32)
        # MuJoCo re-render has a centred principal point; use exactly that for both panels.
        projection[0, 0], projection[1, 1] = 2*intr["fx"]/360, 2*intr["fy"]/640
        projection[2, 2], projection[2, 3], projection[3, 2] = -1.00002, -0.0200002, -1
        targets, truths = [], []
        snapshots = {rung: [np.load(work / f"bigpic-out/{run}-{rung}/snapshot-{t}.npz")
                            for t in moments] for rung in ["L1", "L4"]}
        for k in range(len(moments)):
            index = max(int(snapshots[r][k]["index"]) for r in snapshots)
            camera = poses[index]
            model.cam_pos[0] = WORLD.T @ camera[:3, 3]
            mujoco.mju_mat2Quat(model.cam_quat[0], (WORLD.T @ camera[:3, :3]).ravel())
            mujoco.mj_forward(model, data)
            truth_renderer.update_scene(data, camera=0, scene_option=option)
            truth_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            reference = truth_renderer.render().copy()
            targets.append(index)
            truths.append(reference)
            captured_rgb = np.array(Image.open(recording / rows[index]["name"]).convert("RGB"))
            report["images"].append(dict(run=run, frame=index+1, kind="reference",
                capture_offset=float(truth_data["times"][index]-truth_data["times"][0]),
                recorded_psnr=psnr(reference, captured_rgb),
                truth_match_error=float(truth_data["errors"][index])))
        for rung in snapshots:
            pairs = []
            for k, snap in enumerate(snapshots[rung]):
                index = targets[k]
                alignment = align(snap["first_pose"], poses[int(snap["first_index"])])
                camera = alignment @ poses[index]
                # Replay publishes retirement immediately. Its CPU preview ignores the fade;
                # use active slots only to inspect the persistent surface, state this in report.
                records = snap["records"]
                records = records[records[:, 3, 3] != 1]
                predicted = render(records, camera, projection)
                title = f"{run} frame {index+1} | main {rung} | active surface"
                pairs.append((truths[k], predicted, title))
                Image.fromarray(np.concatenate([truths[k], predicted], 1)).save(
                    raw / f"{run}-{rung}-{moments[k]}.png")
                report["images"].append(dict(run=run, frame=index+1, kind=f"main-{rung}",
                    snapshot_frame=int(snap["index"])+1, count=len(records),
                    psnr=psnr(truths[k], predicted),
                    snapshot_sha256=digest(work / f"bigpic-out/{run}-{rung}/snapshot-{moments[k]}.npz"),
                    snapshot=str((work / f"bigpic-out/{run}-{rung}/snapshot-{moments[k]}.npz").relative_to(work))))
            sheet(f"{run}-main-{rung}", pairs)
        pairs = []
        for label, folder in [("true", "true" if run == "sweep180" else "true-calibrated"),
                              ("old fused", "fused" if run == "sweep180" else "fused-calibrated")]:
            ply = work / f"gs/out/{run}/{folder}/scene.ply"
            scene = load_ply(ply)
            for k in [1, 2]:
                index = targets[k]
                predicted = render(scene.records, poses[index], projection)
                pairs.append((truths[k], predicted,
                              f"{run} frame {index+1} | offline 180s | {label} training"))
                Image.fromarray(np.concatenate([truths[k], predicted], 1)).save(
                    raw / f"{run}-gs-{folder}-{moments[k]}.png")
                report["images"].append(dict(run=run, frame=index+1, kind=f"offline-{folder}",
                    count=scene.count, psnr=psnr(truths[k], predicted),
                    model=str(ply.relative_to(work)), model_sha256=digest(ply)))
            del scene
        sheet(f"{run}-optimized", pairs)
    (out.parent / "images.json").write_text(json.dumps(report, indent=2)+"\n")
    truth_renderer.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
