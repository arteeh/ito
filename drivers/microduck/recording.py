"""Opt-in, synchronous source RGB-D recording; a completed index row commits a frame."""

import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from drivers.microduck import lens


class Recording:
    def __init__(self, directory, model, camera, width, height):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=False)
        self.model, self.camera = model, camera
        self.sequence = 0
        self.intrinsics = dict(
            width=width,
            height=height,
            fx=lens.focal_px(width),
            fy=lens.focal_px(width),
            cx=(width - 1) / 2,
            cy=(height - 1) / 2,
        )
        (self.path / "camera.json").write_text(json.dumps(self.intrinsics, indent=2))
        mujoco.mj_saveModel(model, str(self.path / "scene.mjb"))
        schema = dict(
            mujoco=mujoco.__version__,
            camera_id=camera,
            joints=[
                dict(
                    name=model.joint(i).name,
                    qpos=int(model.jnt_qposadr[i]),
                    dof=int(model.jnt_dofadr[i]),
                )
                for i in range(model.njnt)
            ],
            sensors=[
                dict(
                    name=model.sensor(i).name,
                    address=int(model.sensor_adr[i]),
                    dimension=int(model.sensor_dim[i]),
                )
                for i in range(model.nsensor)
            ],
        )
        (self.path / "model.json").write_text(json.dumps(schema, indent=2))
        (self.path / "README.md").write_text(
            "# Source camera truth\n\n"
            "Each frames.jsonl row commits RGB PNG, axial depth (float32 metres) and state NPZ. "
            "RGB is the exact unrotated MuJoCo image passed to to_uyvy, before chroma "
            "subsampling, mediad rotation or compression. SHA256 covers raw RGB bytes. "
            "No transport timestamps or receive-time poses are used.\n\n"
            "camera is world-from-camera (4x4, row-major), MuJoCo world Z-up, camera "
            "OpenGL right/up/back. camera.json uses integer pixel centres; cx=(width-1)/2. "
            "depth is positive camera -Z, not range. Invalid/far depth must be masked. "
            "RGB, depth and state share one scene copy under the physics lock. "
            "state includes qpos, qvel, qacc, ctrl and sensordata; joint/sensor offsets "
            "are in model.json. scene.mjb is the complete compiled scene including assets. "
            "sim_time is physics seconds; monotonic_ns and unix_ns are host capture clocks. "
            "Recording is synchronous and may lower source FPS; all rendered frames are saved.\n"
        )

    def snapshot(self, data):
        camera = np.eye(4)
        camera[:3, :3] = data.cam_xmat[self.camera].reshape(3, 3)
        camera[:3, 3] = data.cam_xpos[self.camera]
        return dict(
            camera=camera.tolist(),
            sim_time=float(data.time),
            monotonic_ns=time.monotonic_ns(),
            unix_ns=time.time_ns(),
        ), {
            name: getattr(data, name).copy()
            for name in ("qpos", "qvel", "qacc", "ctrl", "sensordata")
        }

    def save(self, rgb, depth, snapshot):
        row, state = snapshot
        name = f"{self.sequence:06d}"
        Image.fromarray(rgb).save(self.path / f"{name}.png")
        np.save(self.path / f"{name}.depth.npy", depth.astype(np.float32))
        np.savez_compressed(self.path / f"{name}.state.npz", **state)
        row.update(
            sequence=self.sequence,
            rgb=f"{name}.png",
            depth=f"{name}.depth.npy",
            state=f"{name}.state.npz",
            rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        )
        with (self.path / "frames.jsonl").open("a") as index:
            index.write(json.dumps(row) + "\n")
        self.sequence += 1
