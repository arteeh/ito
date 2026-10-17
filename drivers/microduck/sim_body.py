import atexit
import base64
import collections
import json
import os
import socket
import socketserver
import struct
import sys
import threading
import time
import zlib

import mujoco
import numpy as np
from mjlab_microduck.sim import body_server
from mjlab_microduck.sim.camera import Camera

from drivers.microduck import lens, truth


class Fisheye:
    """The beta's lens (lens.py), drawn from pinhole views, which are all MuJoCo renders: views
    turned YAWS about the camera's up axis, side by side in one framebuffer, each as dense as the
    fisheye where it is used (at least one view pixel per fisheye pixel), then remapped."""

    YAWS = (-46.0, 0.0, 46.0)  # degrees to the right

    def __init__(self, width, height, depth_step):
        try:
            import cv2
        except ImportError:
            raise SystemExit("the sim body needs OpenCV now: re-run setup-sim.sh") from None
        self.cv = cv2
        self.width, self.height = width, height
        u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        ray = lens.rays(u, v, width, height)
        # Half a pixel across and down: the fisheye pixel's footprint, which sizes each view.
        across, down = lens.rays(u + 0.5, v, width, height), lens.rays(u, v + 0.5, width, height)
        azimuth = np.degrees(np.arctan2(ray[..., 0], ray[..., 2]))
        owner = np.argmin(np.abs(azimuth[..., None] - np.array(self.YAWS)), -1)
        map_x = np.empty((height, width), np.float32)
        map_y = np.empty((height, width), np.float32)
        # View-axial depth to the camera's own axial depth, per fisheye pixel.
        self.axial = np.empty((height, width), np.float32)
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
            self.axial[mine] = ray[mine][:, 2] / (ray[mine] @ axis)
            self.views.append((a, left, top, w, h, focal, mujoco.MjrRect(x0, 0, w, h)))
            x0 += w
        self.size = (x0, max(view[4] for view in self.views))
        self.rect = mujoco.MjrRect(0, 0, *self.size)
        self.maps = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
        # Depth: every depth_step-th fisheye pixel, the nearest rendered sample, never averaged.
        rows = slice(depth_step // 2, None, depth_step)
        self.depth_at = (
            np.rint(map_y[rows, rows.start :: depth_step]).astype(np.intp),
            np.rint(map_x[rows, rows.start :: depth_step]).astype(np.intp),
        )
        self.depth_axial = self.axial[rows, rows.start :: depth_step]
        self.atlas = np.empty((self.size[1], self.size[0], 3), np.uint8)
        self.atlas_depth = np.empty((self.size[1], self.size[0]), np.float32)

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
        # MuJoCo lights a headlight along each drawn view's own axis, which would light every view
        # differently and seam the fisheye: fix it along the camera's axis for all of them.
        heads = [light for light in scene.lights[: scene.nlight] if light.headlight]
        for light in heads:
            light.headlight = 0
            light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            light.pos[:] = (np.array(cameras[0].pos) + np.array(cameras[1].pos)) / 2
            light.dir[:] = forward
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
            for light in heads:
                light.headlight = 1
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

    def depth(self, renderer, model):
        """The scene's axial depth along the camera's own axis, in metres, at every
        depth_step-th fisheye pixel (0: nothing there)."""
        flags = renderer.scene.flags
        original = flags.copy()
        # As mujoco.Renderer's depth: segmented drawing keeps far depth accurate.
        flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
        flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
        try:
            self.views_drawn(renderer)
            mujoco.mjr_readPixels(None, self.atlas_depth, self.rect, renderer._mjr_context)
        finally:
            np.copyto(flags, original)
        extent = model.stat.extent
        near = np.float32(model.vis.map.znear * extent)
        far = np.float32(model.vis.map.zfar * extent)
        # mujoco.Renderer's reversed-Z linearisation.
        c = np.float32(-0.5) * (-(far + near) / (far - near)) - np.float32(0.5)
        d = np.float32(-0.5) * (-(np.float32(2) * far * near) / (far - near))
        z = self.atlas_depth[self.depth_at].astype(np.float64)
        return (d / (z + c)) * self.depth_axial


class HeadCamera(Camera):
    def __init__(self, model, name, width=None, height=None, mount=0):
        height = int(os.environ.get("ITO_MICRODUCK_CAMERA_HEIGHT", lens.DEFAULT_HEIGHT))
        width = lens.width_for(height)
        self.fisheye = Fisheye(width, height, max(1, width // 640))
        # MuJoCo renders no larger than the offscreen buffer, which the scene sizes at 640x480.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.fisheye.size[0])
        model.vis.global_.offheight = max(model.vis.global_.offheight, self.fisheye.size[1])
        super().__init__(model, name, *self.fisheye.size, mount=mount)
        self.width, self.height = width, height
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
        # MuJoCo's shadow pass produces dark floor triangles at this low camera height.
        # Keep the head image clean without changing the viewer or the robot's physics.
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        return self.fisheye.rgb(self.renderer)

    def publish(self, frames, exposed=None):
        rgb = frames[0] if self.blur == 1 else np.mean(frames, axis=0).astype(np.uint8)
        packed = self.uyvy(rgb)
        with self.lock:
            self.latest = packed
            self.exposed = None if exposed is None else exposed | {"thumb": truth.encode(rgb)}


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
        self.exposed_at = collections.deque()
        self.exposed = None
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
        if len(self.taking) == (self.blur + 1) // 2:
            # The middle of a blurred frame's span, on CLOCK_MONOTONIC (the driver's clock).
            self.exposed_at.append(self.world.epoch_ns / 1e9 + data.time)
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
        width, height = self.fisheye.size
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
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
                    self.exposed_at.popleft()
                    dropped += 1
                states = self.shots.popleft()
                exposed_at = self.exposed_at.popleft()
            now = time.perf_counter()
            if next_frame is None or now - next_frame > period:
                late += next_frame is not None
                next_frame = now
            elif next_frame > now:
                time.sleep(next_frame - now)
            start = time.perf_counter()
            frames = []
            for i, state in enumerate(states):
                self.pose(kept, state)
                if i == (len(states) - 1) // 2:
                    # MuJoCo's camera frame is Ito's: +X right, +Y up, looking down -Z.
                    exposed = dict(
                        time=exposed_at,
                        position=kept.cam_xpos[self.camera].tolist(),
                        orientation=truth.quaternion(kept.cam_xmat[self.camera].reshape(3, 3)),
                    )
                self.scene(kept)
                frames.append(self.draw())
                if i == (len(states) - 1) // 2 and TRUTH_DEPTH and self.seq % TRUTH_DEPTH == 0:
                    mm = self.depth()
                    exposed |= dict(
                        depth_shape=list(mm.shape),
                        depth=base64.b64encode(zlib.compress(mm.tobytes(), 1)).decode(),
                    )
                    if TRUTH_DUMP:
                        np.savez(
                            f"{TRUTH_DUMP}/{self.seq:06d}.npz",
                            rgb=frames[-1],
                            depth_mm=mm,
                            **{
                                k: np.asarray(exposed[k])
                                for k in ("time", "position", "orientation")
                            },
                        )
            self.publish(frames, exposed)
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

    def depth(self):
        """The rendered scene's axial depth, every width // 640-th fisheye pixel, in millimetres
        (0: none)."""
        depth = self.fisheye.depth(self.renderer, self.model)
        return np.where(depth < 30, np.rint(depth * 1000), 0).astype("<u2")

    def next_frame(self, seen, timeout=1.0):
        with self.fresh:
            if not self.fresh.wait_for(lambda: self.seq != seen, timeout):
                return seen, None, None
            return self.seq, self.latest, self.exposed


class TruthServer(socketserver.ThreadingTCPServer):
    """ITO_MICRODUCK_TRUTH_PORT: for each frame handed to mediad, one JSON line to every
    reader: when it was sent and exposed (CLOCK_MONOTONIC seconds), the camera's pose in
    MuJoCo's world and a thumbnail (drivers/microduck/truth.py). Diagnostics only."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port):
        self.readers = []
        self.readers_lock = threading.Lock()
        super().__init__(("127.0.0.1", port), TruthHandler)
        threading.Thread(target=self.serve_forever, name="truth", daemon=True).start()

    def send(self, row):
        line = (json.dumps(row) + "\n").encode()
        with self.readers_lock:
            for reader in list(self.readers):
                try:
                    reader.sendall(line)
                except OSError:
                    self.readers.remove(reader)


class TruthHandler(socketserver.BaseRequestHandler):
    def handle(self):
        with self.server.readers_lock:
            self.server.readers.append(self.request)
        try:
            while self.request.recv(1):
                pass
        except OSError:
            pass


TRUTH = None
# ITO_MICRODUCK_TRUTH_DEPTH=N: every Nth frame's truth also carries its rendered depth.
TRUTH_DEPTH = int(os.environ.get("ITO_MICRODUCK_TRUTH_DEPTH", 0))
# ITO_MICRODUCK_TRUTH_DUMP=DIR also saves those frames, depths and poses there (npz), for
# judging a depth estimator offline against the exact scene.
TRUTH_DUMP = os.environ.get("ITO_MICRODUCK_TRUTH_DUMP")


class FrameHandler(body_server.FrameHandler):
    def handle(self):
        camera = self.server.camera
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"== camera: a reader connected from {self.client_address}", flush=True)
        seen = 0
        try:
            while True:
                seen, frame, exposed = camera.next_frame(seen)
                if frame is not None:
                    self.request.sendall(struct.pack("<I", len(frame)) + frame)
                    if TRUTH is not None and exposed is not None:
                        TRUTH.send(exposed | {"sent": time.perf_counter()})
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        print("== camera: the reader went away", flush=True)


# Seconds the hand that holds a new duck keeps holding it after robotd turns torque on: robotd
# ramps to the home pose over two, and a duck let go at once sometimes fell during the ramp and
# got up facing anywhere, so no two runs of one script walked the same path.
HOLD_AFTER_TORQUE = 3.0


def held_through_ramp(set_torque):
    def held(body, on):
        first = on and not getattr(body, "ever_torqued", False)
        set_torque(body, on)
        if first:
            body.ever_torqued = True
            body.released = False
            release = threading.Timer(HOLD_AFTER_TORQUE, setattr, (body, "released", True))
            release.daemon = True
            release.start()

    return held


if __name__ == "__main__":
    body_server.Body.set_torque = held_through_ramp(body_server.Body.set_torque)
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
    if os.environ.get("ITO_MICRODUCK_TRUTH_PORT"):
        TRUTH = TruthServer(int(os.environ["ITO_MICRODUCK_TRUTH_PORT"]))
    body_server.Camera = ThreadedHeadCamera
    body_server.FrameHandler = FrameHandler
    body_server.main()
