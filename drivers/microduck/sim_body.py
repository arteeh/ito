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
from mjlab_microduck.sim.camera import Camera

from drivers.microduck import lens


class Fisheye:
    """The beta's lens (lens.py), drawn from pinhole views, which are all MuJoCo renders: views
    turned YAWS about the camera's up axis, side by side in one framebuffer, each as dense as the
    fisheye where it is used (at least one view pixel per fisheye pixel), then remapped. One
    pinhole over the whole frame would need 2.55 focal lengths either side of the axis at the
    corners: several times the pixels for the same sharpness."""

    YAWS = (-46.0, 0.0, 46.0)  # degrees to the right

    def __init__(self, width, height):
        try:
            import cv2
        except ImportError:
            raise SystemExit("the sim body needs OpenCV: uv run --extra sim") from None
        self.cv = cv2
        started = time.perf_counter()
        self.width, self.height = width, height
        u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        ray = lens.rays(u, v, width, height)
        # Half a pixel across and down: the fisheye pixel's footprint, which sizes each view.
        across, down = lens.rays(u + 0.5, v, width, height), lens.rays(u, v + 0.5, width, height)
        azimuth = np.degrees(np.arctan2(ray[..., 0], ray[..., 2]))
        owner = np.argmin(np.abs(azimuth[..., None] - np.array(self.YAWS)), -1)
        map_x = np.empty((height, width), np.float32)
        map_y = np.empty((height, width), np.float32)
        self.views, x0 = [], 0
        for i, yaw in enumerate(self.YAWS):
            a = np.radians(yaw)
            right = np.array([np.cos(a), 0.0, -np.sin(a)])
            axis = np.array([np.sin(a), 0.0, np.cos(a)])
            mine = owner == i

            def plane(r, mine=mine, right=right, axis=axis):
                z = r[mine] @ axis
                return (r[mine] @ right) / z, r[mine][:, 1] / z

            x, y = plane(ray)
            xa, ya = plane(across)
            xd, yd = plane(down)
            jacobian = 2 * np.stack(
                [np.stack([xa - x, xd - x], -1), np.stack([ya - y, yd - y], -1)], -2
            )
            focal = 1 / np.linalg.svd(jacobian, compute_uv=False)[:, -1].min()
            left, top = x.min() - 2 / focal, y.min() - 2 / focal
            w = int(np.ceil((x.max() - left) * focal + 2))
            h = int(np.ceil((y.max() - top) * focal + 2))
            map_x[mine] = x0 + (x - left) * focal - 0.5
            # OpenGL's rows count up from the bottom of the framebuffer.
            map_y[mine] = h - 1 - ((y - top) * focal - 0.5)
            self.views.append((a, left, top, w, h, focal, mujoco.MjrRect(x0, 0, w, h)))
            x0 += w
        self.size = (x0, max(view[4] for view in self.views))
        self.rect = mujoco.MjrRect(0, 0, *self.size)
        self.maps = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
        self.atlas = np.empty((self.size[1], self.size[0], 3), np.uint8)
        self.build_seconds = time.perf_counter() - started

    def views_drawn(self, renderer):
        """Draw every view into the framebuffer from the scene's camera."""
        scene, context = renderer.scene, renderer._mjr_context
        cameras = scene.camera
        kept = [
            (np.array(c.forward), np.array(c.up), c.frustum_center, c.frustum_bottom, c.frustum_top)
            for c in cameras
        ]
        forward, up = kept[0][0].astype(np.float64), kept[0][1].astype(np.float64)
        right = np.cross(forward, up)
        near = cameras[0].frustum_near
        # The views need no lighting of their own: the scene's headlight is ambient only
        # (scene.py), so all three views are lit alike and the fisheye has no seams.
        try:
            for yaw, left, top, w, h, focal, rect in self.views:
                look = np.cos(yaw) * forward + np.sin(yaw) * right
                # Both eyes: mono rendering draws their average. MuJoCo sets the frustum's left
                # and right from its centre and the viewport's aspect, which these match.
                for camera in cameras:
                    camera.forward[:] = look
                    camera.frustum_center = near * (left + w / focal / 2)
                    camera.frustum_top = -near * top
                    camera.frustum_bottom = -near * (top + h / focal)
                mujoco.mjr_render(rect, scene, context)
        finally:
            for camera, (f, _, center, bottom, top) in zip(cameras, kept, strict=True):
                camera.forward[:] = f
                camera.frustum_center, camera.frustum_bottom, camera.frustum_top = (
                    center,
                    bottom,
                    top,
                )

    def rgb(self, renderer):
        self.views_drawn(renderer)
        mujoco.mjr_readPixels(self.atlas, None, self.rect, renderer._mjr_context)
        return self.cv.remap(self.atlas, *self.maps, self.cv.INTER_LINEAR)


class HeadCamera(Camera):
    def __init__(self, model, name, width=None, height=None, mount=0):
        self.fisheye = Fisheye(lens.WIDTH, lens.HEIGHT)
        # MuJoCo renders no larger than the offscreen buffer, which the scene sizes at 640x480.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.fisheye.size[0])
        model.vis.global_.offheight = max(model.vis.global_.offheight, self.fisheye.size[1])
        super().__init__(model, name, *self.fisheye.size, mount=mount)
        self.width, self.height = width, height
        self.option = mujoco.MjvOption()
        self.option.geomgroup[5] = 1

    def render(self, world):
        with world.lock:
            self.scene(world.data)
        self.publish(self.draw())

    def uyvy(self, rgb):
        """mjlab's to_uyvy (BT.601 full range, chroma averaged across each pair) in OpenCV:
        the same frame to within 2 levels, a fifth of the time, which the fisheye needs."""
        ycc = self.fisheye.cv.cvtColor(rgb, self.fisheye.cv.COLOR_RGB2YCrCb)
        pairs = ycc.reshape(ycc.shape[0], -1, 2, 3).astype(np.uint16)
        packed = np.empty((ycc.shape[0], ycc.shape[1] // 2, 4), np.uint8)
        packed[..., 0] = (pairs[:, :, 0, 2] + pairs[:, :, 1, 2] + 1) >> 1
        packed[..., 1] = ycc[:, 0::2, 0]
        packed[..., 2] = (pairs[:, :, 0, 1] + pairs[:, :, 1, 1] + 1) >> 1
        packed[..., 3] = ycc[:, 1::2, 0]
        return packed.tobytes()

    def scene(self, data):
        self.renderer.update_scene(data, camera=self.camera, scene_option=self.option)

    def draw(self):
        return self.fisheye.rgb(self.renderer)

    def publish(self, rgb, exposed_ns=None):
        packed = self.uyvy(rgb)
        with self.lock:
            self.latest = packed
            self.latest_ns = exposed_ns


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


FPS = 30
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
        if data.time < self.due - 1e-9:
            return
        with self.fresh:
            self.shots.append(
                (data.time, data.qpos.copy(), data.mocap_pos.copy(), data.mocap_quat.copy())
            )
            self.fresh.notify_all()
        self.due = max(self.due + period, data.time)

    def stop(self):
        self.stopping.set()
        with self.fresh:
            self.fresh.notify_all()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)

    def loop(self):
        width, height = self.fisheye.size
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            from OpenGL import GL

            gl = GL.glGetString(GL.GL_RENDERER).decode()
        except Exception as exc:
            gl = f"unknown ({exc})"
        print(
            f"== head camera: rendering, MUJOCO_GL={os.environ.get('MUJOCO_GL')} on {gl}; "
            f"fisheye maps took {self.fisheye.build_seconds:.1f} s",
            flush=True,
        )
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
                state = self.shots.popleft()
            now = time.perf_counter()
            if next_frame is None or now - next_frame > period:
                late += next_frame is not None
                next_frame = now
            elif next_frame > now:
                time.sleep(next_frame - now)
            start = time.perf_counter()
            self.pose(kept, state)
            self.scene(kept)
            # The exposure on CLOCK_MONOTONIC, the clock robotd and tofd stamp with: the physics
            # time of the state drawn.
            self.publish(self.draw(), self.world.epoch_ns + round(state[0] * 1e9))
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
                return seen, None, None
            return self.seq, self.latest, getattr(self, "latest_ns", None)


# A reader (mediad) that sends these four bytes on connecting gets each frame's exposure:
# the same four bytes back, then [len u32][exposed u64, CLOCK_MONOTONIC ns, 0 unknown][frame].
# One that does not (an older mediad) gets [len u32][frame], as before.
STAMPS = b"ITS1"
STAMPS_WAIT = 0.5  # seconds a reader has to ask before frames go out unstamped


class FrameHandler(body_server.FrameHandler):
    def handle(self):
        camera = self.server.camera
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.request.settimeout(STAMPS_WAIT)
        asked = b""
        try:
            while len(asked) < len(STAMPS):
                chunk = self.request.recv(len(STAMPS) - len(asked))
                if not chunk:
                    break
                asked += chunk
        except (TimeoutError, OSError):
            pass
        self.request.settimeout(None)
        stamped = asked == STAMPS
        print(
            f"== camera: a reader connected from {self.client_address}, "
            f"{'stamped' if stamped else 'unstamped'} frames",
            flush=True,
        )
        seen = 0
        try:
            if stamped:
                self.request.sendall(STAMPS)
            while True:
                seen, frame, exposed = camera.next_frame(seen)
                if frame is not None:
                    header = (
                        struct.pack("<IQ", len(frame), exposed or 0)
                        if stamped
                        else struct.pack("<I", len(frame))
                    )
                    self.request.sendall(header + frame)
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
                    viewer.opt.geomgroup[4] = 1  # the eye ring (sim.py)
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
