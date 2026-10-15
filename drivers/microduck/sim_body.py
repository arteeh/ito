"""Keep the CAD body visible in Pollen's viewer without occluding its head camera."""

import atexit
import base64
import json
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
from mjlab_microduck.sim import body_server
from mjlab_microduck.sim.camera import Camera, to_uyvy

from drivers.microduck import lens

WORLD = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
UPRIGHT = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
THUMB = 24


def thumbnail(rgb):
    """A THUMB x THUMB grey image of a frame, base64: enough to tell frames apart."""
    height, width = rgb.shape[:2]
    rows = (np.arange(THUMB) + 0.5) * height / THUMB
    columns = (np.arange(THUMB) + 0.5) * width / THUMB
    grey = rgb[rows.astype(int)][:, columns.astype(int)].mean(axis=2)
    return base64.b64encode(grey.astype(np.uint8).tobytes()).decode("ascii")


class HeadCamera(Camera):
    def __init__(self, model, name):
        self.beta = os.environ.get("ITO_MICRODUCK_BOARD", "zero3") == "beta"
        height = int(os.environ.get("ITO_MICRODUCK_CAMERA_HEIGHT", 1080 if self.beta else 360))
        width = lens.width_for(height)
        # MuJoCo renders no larger than the offscreen buffer, which the scene sizes at 640x480.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        hfov = float(os.environ.get("ITO_MICRODUCK_CAMERA_HFOV", lens.HFOV_DEG))
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        model.cam_fovy[camera_id] = np.degrees(
            2 * np.arctan(height / width * np.tan(np.radians(hfov / 2)))
        )
        super().__init__(model, name, width, height)
        # Pollen first corrects the CAD camera by 180 degrees around local X.
        # Apply the upright beta roll afterwards; those rotations do not commute.
        if self.beta:
            q = model.cam_quat[camera_id].copy()
            mujoco.mju_mulQuat(model.cam_quat[camera_id], q, np.array([2**-0.5, 0, 0, 2**-0.5]))
        # ITO_MICRODUCK_TRUTH names a file for one line per rendered frame: when it was taken
        # (perf_counter, the driver's clock), the camera's true pose and a thumbnail to find
        # it again among the frames the pilot received (e2e/truth.py).
        path = os.environ.get("ITO_MICRODUCK_TRUTH")
        self.truth = open(path, "a", buffering=1) if path else None
        # ITO_MICRODUCK_BLUR=N averages N renders made back to back into each frame: motion
        # blur the simulator otherwise never shows (drive (f) of #36). The exposure is their
        # span; the truth pose is the middle one's.
        self.blur = max(1, int(os.environ.get("ITO_MICRODUCK_BLUR", 1)))
        recording = os.environ.get("ITO_MICRODUCK_RGBD_RECORD")
        self.rgbd = Path(recording) if recording else None
        self.source_frame_id = 0
        if self.rgbd:
            if self.blur != 1:
                raise ValueError("Exact RGB-D recording requires instantaneous sim exposure")
            self.rgbd.mkdir(parents=True, exist_ok=False)
            focal = width / 2 / np.tan(np.radians(hfov / 2))
            (self.rgbd / "camera.json").write_text(
                json.dumps(
                    dict(
                        width=width if self.beta else height,
                        height=height if self.beta else width,
                        fx=focal,
                        fy=focal,
                        cx=width / 2 if self.beta else height / 2 - 1,
                        cy=height / 2 if self.beta else width / 2,
                    )
                )
            )
            self.rgbd_index = (self.rgbd / "frames.jsonl").open("w", buffering=1)
            atexit.register(self.rgbd_index.close)

    def render(self, world):
        option = mujoco.MjvOption()
        option.geomgroup[2] = 0
        option.geomgroup[5] = 1  # Room cutaway walls are hidden only in the overview viewer.
        frames, poses = [], []
        for _ in range(self.blur):
            with world.lock:
                self.renderer.update_scene(world.data, camera=self.camera, scene_option=option)
                if self.truth or self.rgbd:
                    camera = np.eye(4)
                    # MuJoCo's world (z up) and camera to Ito's (y up), the image upright.
                    camera[:3, :3] = (
                        WORLD
                        @ world.data.cam_xmat[self.camera].reshape(3, 3)
                        @ (np.eye(3) if self.beta else UPRIGHT)
                    )
                    camera[:3, 3] = WORLD @ world.data.cam_xpos[self.camera]
                    poses.append((time.perf_counter(), camera))
            # MuJoCo's shadow pass produces dark floor triangles at this low camera height.
            # Keep the head image clean without changing the viewer or the robot's physics.
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            frames.append(self.renderer.render().copy())
        rgb = frames[0] if self.blur == 1 else np.mean(frames, axis=0).astype(np.uint8)
        if self.rgbd:
            # The copied MjvScene is the RGB scene, even if physics advanced meanwhile.
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render().copy()
            self.renderer.disable_depth_rendering()
            taken, camera = poses[0]
            name = f"{self.source_frame_id:06d}.npz"
            np.savez(
                self.rgbd / name,
                rgb=rgb.copy() if self.beta else np.rot90(rgb, -1).copy(),
                depth=depth.copy() if self.beta else np.rot90(depth, -1).copy(),
                camera=camera,
            )
            self.rgbd_index.write(
                json.dumps(
                    dict(
                        source_frame_id=self.source_frame_id,
                        file=name,
                        captured=taken,
                        available=time.perf_counter(),
                        exposure_duration=0,
                        readout_duration=0,
                        exposure_status="sim instantaneous",
                        readout_status="global",
                        coordinate_revision=0,
                    )
                )
                + "\n"
            )
            self.source_frame_id += 1
        if self.truth:
            taken, camera = poses[len(poses) // 2]
            self.truth.write(
                json.dumps(
                    dict(
                        t=taken,
                        exposure=poses[-1][0] - poses[0][0],
                        camera=camera.ravel().tolist(),
                        thumb=thumbnail(rgb),
                    ),
                )
                + "\n"
            )
        packed = to_uyvy(rgb)
        with self.lock:
            self.latest = packed


# The real head camera (IMX219 through mediad) runs every rung at 30 fps.
FPS = int(os.environ.get("ITO_MICRODUCK_CAMERA_FPS", 30))
CAMERAS = []  # Render threads to stop before the viewer closes.


class ThreadedHeadCamera(HeadCamera):
    """Render on a thread of its own, at the real camera's rate.

    Pollen's step loop renders inline every few physics passes, so each 12 ms render delays
    physics and the viewer, and the rate is fixed at 15 fps. Here the step loop only hands over
    the world; a dedicated thread renders at FPS, taking the world lock only to copy the scene.
    """

    def __init__(self, model, name):
        super().__init__(model, name)
        # A GL context is current on one thread at a time: the render thread makes its own.
        self.renderer.close()
        self.renderer = None
        self.model = model
        self.world = None
        self.seq = 0
        self.fresh = threading.Condition(self.lock)
        self.thread = None
        self.stopping = threading.Event()
        # Before GLFW goes: rendering on after the viewer closed and GLFW terminated
        # segfaulted the body in make_context_current (GLFW 3.4, Ubuntu 26.04).
        atexit.register(self.stop)
        CAMERAS.append(self)

    def render(self, world):
        if self.world is None:
            self.world = world
            self.thread = threading.Thread(target=self.loop, name="head-camera", daemon=True)
            self.thread.start()

    def stop(self):
        self.stopping.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)

    def loop(self):
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        try:
            self.frames()
        finally:
            self.renderer.close()

    def frames(self):
        period = 1.0 / max(1, FPS)
        next_frame = time.perf_counter()
        costs, late, window = [], 0, time.perf_counter()
        while not self.stopping.is_set():
            start = time.perf_counter()
            super().render(self.world)
            with self.lock:
                self.seq += 1
                self.fresh.notify_all()
            done = time.perf_counter()
            costs.append(done - start)
            if done - window >= 10:
                costs.sort()
                print(
                    f"== head camera: {len(costs) / (done - window):.1f} fps (target {FPS}), "
                    f"render p50 {costs[len(costs) // 2] * 1e3:.1f} ms "
                    f"p95 {costs[int(len(costs) * 0.95)] * 1e3:.1f} ms, late {late}",
                    flush=True,
                )
                costs, late, window = [], 0, done
            next_frame += period
            slack = next_frame - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                late += 1
                next_frame = time.perf_counter()

    def next_frame(self, seen, timeout=1.0):
        """The first frame newer than `seen`, or None after `timeout`."""
        with self.fresh:
            if not self.fresh.wait_for(lambda: self.seq != seen, timeout):
                return seen, None
            return self.seq, self.latest


class FrameHandler(body_server.FrameHandler):
    """Send each rendered frame once, as it is rendered, like a camera's own clock."""

    def handle(self):
        camera = self.server.camera
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"== camera: a reader connected from {self.client_address}", flush=True)
        seen = 0
        try:
            while True:
                seen, frame = camera.next_frame(seen)
                if frame is not None:
                    self.request.sendall(struct.pack("<I", len(frame)) + frame)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        print("== camera: the reader went away", flush=True)


if __name__ == "__main__":
    if "--headless" not in sys.argv:
        import mujoco.viewer

        launch = mujoco.viewer.launch_passive

        def visible_viewer(model, data, **kwargs):
            try:
                viewer = launch(model, data, **kwargs)
                with viewer.lock():
                    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
                close = viewer.close

                def stop_cameras_then_close():
                    for camera in CAMERAS:
                        camera.stop()
                    close()

                viewer.close = stop_cameras_then_close
                return viewer
            except Exception as exc:
                raise SystemExit(f"MuJoCo viewer unavailable: {exc}") from exc

        mujoco.viewer.launch_passive = visible_viewer
    body_server.Camera = ThreadedHeadCamera
    body_server.FrameHandler = FrameHandler
    body_server.main()
