"""Keep the CAD body visible in Pollen's viewer without occluding its head camera."""

import os
import socket
import struct
import sys
import threading
import time

import mujoco
from mjlab_microduck.sim import body_server
from mjlab_microduck.sim.camera import Camera, to_uyvy

from drivers.microduck import lens


class HeadCamera(Camera):
    def __init__(self, model, name):
        height = int(os.environ.get("ITO_MICRODUCK_CAMERA_HEIGHT", 360))
        width = lens.width_for(height)
        # MuJoCo renders no larger than the offscreen buffer, which the scene sizes at 640x480.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        model.cam_fovy[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)] = (
            lens.vertical_fov_deg(width, height)
        )
        super().__init__(model, name, width, height)

    def render(self, world):
        option = mujoco.MjvOption()
        option.geomgroup[2] = 0
        option.geomgroup[5] = 1  # Room cutaway walls are hidden only in the overview viewer.
        with world.lock:
            self.renderer.update_scene(world.data, camera=self.camera, scene_option=option)
        # MuJoCo's shadow pass produces dark floor triangles at this low camera height.
        # Keep the head image clean without changing the viewer or the robot's physics.
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        packed = to_uyvy(self.renderer.render())
        with self.lock:
            self.latest = packed


# The real head camera (IMX219 through mediad) runs every rung at 30 fps.
FPS = int(os.environ.get("ITO_MICRODUCK_CAMERA_FPS", 30))


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

    def render(self, world):
        if self.world is None:
            self.world = world
            threading.Thread(target=self.loop, name="head-camera", daemon=True).start()

    def loop(self):
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        period = 1.0 / max(1, FPS)
        next_frame = time.perf_counter()
        costs, late, window = [], 0, time.perf_counter()
        while True:
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
                return viewer
            except Exception as exc:
                raise SystemExit(f"MuJoCo viewer unavailable: {exc}") from exc

        mujoco.viewer.launch_passive = visible_viewer
    body_server.Camera = ThreadedHeadCamera
    body_server.FrameHandler = FrameHandler
    body_server.main()
