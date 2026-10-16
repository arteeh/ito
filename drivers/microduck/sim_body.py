import atexit
import collections
import os
import socket
import struct
import sys
import threading
import time

import mujoco
import numpy as np
from mjlab_microduck.sim import body_server
from mjlab_microduck.sim.camera import Camera, to_uyvy

from drivers.microduck import lens


class HeadCamera(Camera):
    def __init__(self, model, name, width=None, height=None, mount=0):
        height = int(os.environ.get("ITO_MICRODUCK_CAMERA_HEIGHT", lens.DEFAULT_HEIGHT))
        width = lens.width_for(height)
        hfov = float(os.environ.get("ITO_MICRODUCK_CAMERA_HFOV", lens.HFOV_ESTIMATE_DEG))
        # MuJoCo renders no larger than the offscreen buffer, which the scene sizes at 640x480.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        # MuJoCo's fovy is across the frame's height.
        model.cam_fovy[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)] = (
            lens.vertical_fov_deg(width, height, hfov)
        )
        super().__init__(model, name, width, height, mount=mount)
        self.blur = max(1, int(os.environ.get("ITO_MICRODUCK_BLUR", 1)))
        self.option = mujoco.MjvOption()
        self.option.geomgroup[2] = 0
        self.option.geomgroup[5] = 1

    def render(self, world):
        frames = []
        for _ in range(self.blur):
            with world.lock:
                self.scene(world.data)
            frames.append(self.draw())
        self.publish(frames)

    def scene(self, data):
        self.renderer.update_scene(data, camera=self.camera, scene_option=self.option)

    def draw(self):
        # MuJoCo's shadow pass produces dark floor triangles at this low camera height.
        # Keep the head image clean without changing the viewer or the robot's physics.
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        return self.renderer.render().copy()

    def publish(self, frames):
        rgb = frames[0] if self.blur == 1 else np.mean(frames, axis=0).astype(np.uint8)
        packed = to_uyvy(rgb)
        with self.lock:
            self.latest = packed


class Stepping:
    def __init__(self):
        self.hooks = []

    def __getattr__(self, name):
        return getattr(mujoco, name)

    def mj_step(self, model, data, *args):
        for hook in self.hooks:
            hook(data)
        mujoco.mj_step(model, data, *args)


STEPPING = Stepping()


FPS = int(os.environ.get("ITO_MICRODUCK_CAMERA_FPS", 30))
CAMERAS = []


class ThreadedHeadCamera(HeadCamera):
    """Renders on its own thread at FPS. Each frame is a joint-state copy taken before the physics
    step it is due at (Stepping), not the live world: physics advances in 20 ms batches, so
    reading it on the camera's 33 ms clock spaces frames unevenly."""

    def __init__(self, model, name, width=None, height=None, mount=0):
        super().__init__(model, name, width, height, mount)
        self.renderer.close()
        self.renderer = None
        self.model = model
        self.world = None
        self.seq = 0
        self.fresh = threading.Condition(self.lock)
        self.thread = None
        self.stopping = threading.Event()
        self.shots = collections.deque()
        self.taking = []
        self.due = None
        # Before GLFW goes: rendering on after the viewer closed and GLFW terminated
        # segfaulted the body in make_context_current (GLFW 3.4, Ubuntu 26.04).
        atexit.register(self.stop)
        CAMERAS.append(self)

    def render(self, world):
        if self.world is None:
            self.world = world
            self.thread = threading.Thread(target=self.loop, name="head-camera", daemon=True)
            self.thread.start()

    def before_step(self, data):
        period = 1.0 / max(1, FPS)
        if self.due is None or data.time < self.due - 1.0:
            self.due = data.time
        if not self.taking and data.time < self.due - 1e-9:
            return
        self.taking.append(
            (data.time, data.qpos.copy(), data.mocap_pos.copy(), data.mocap_quat.copy())
        )
        if len(self.taking) < self.blur:
            return
        with self.fresh:
            self.shots.append(self.taking)
            self.fresh.notify_all()
        self.taking = []
        self.due = max(self.due + period, data.time)

    def stop(self):
        self.stopping.set()
        with self.fresh:
            self.fresh.notify_all()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)

    def loop(self):
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        body_server.mujoco = STEPPING
        STEPPING.hooks.append(self.before_step)
        try:
            self.frames()
        finally:
            self.renderer.close()

    def pose(self, kept, state):
        _, qpos, mocap_pos, mocap_quat = state
        kept.qpos[:] = qpos
        kept.mocap_pos[:] = mocap_pos
        kept.mocap_quat[:] = mocap_quat
        mujoco.mj_kinematics(self.model, kept)
        mujoco.mj_comPos(self.model, kept)
        mujoco.mj_camlight(self.model, kept)
        if self.model.ntendon:
            mujoco.mj_tendon(self.model, kept)

    def frames(self):
        period = 1.0 / max(1, FPS)
        kept = mujoco.MjData(self.model)
        next_frame = None
        costs, late, dropped, window = [], 0, 0, time.perf_counter()
        while not self.stopping.is_set():
            with self.fresh:
                self.fresh.wait_for(lambda: self.shots or self.stopping.is_set(), timeout=0.5)
                if self.stopping.is_set() or not self.shots:
                    continue
                while len(self.shots) > 2:
                    self.shots.popleft()
                    dropped += 1
                states = self.shots.popleft()
            now = time.perf_counter()
            if next_frame is None or now - next_frame > period:
                late += next_frame is not None
                next_frame = now
            elif next_frame > now:
                time.sleep(next_frame - now)
            start = time.perf_counter()
            frames = []
            for state in states:
                self.pose(kept, state)
                self.scene(kept)
                frames.append(self.draw())
            self.publish(frames)
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
                    f"p95 {costs[int(len(costs) * 0.95)] * 1e3:.1f} ms, late {late}, "
                    f"dropped {dropped}",
                    flush=True,
                )
                costs, late, dropped, window = [], 0, 0, done
            next_frame += period

    def next_frame(self, seen, timeout=1.0):
        with self.fresh:
            if not self.fresh.wait_for(lambda: self.seq != seen, timeout):
                return seen, None
            return self.seq, self.latest


class FrameHandler(body_server.FrameHandler):
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
