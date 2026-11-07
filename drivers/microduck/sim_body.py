import atexit
import collections
import ctypes
import os
import queue
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

    def views_drawn(self, renderer, between=None):
        """Draw every view into the framebuffer from the scene's camera; `between`, called after
        each view."""
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
                if between is not None:
                    between()
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
        return self.remap(self.atlas)

    def remap(self, atlas):
        """The fisheye frame from the views' pixels (rows as OpenGL reads them)."""
        return self.cv.remap(atlas, *self.maps, self.cv.INTER_LINEAR)


class Readback:
    """The views' pixels off the GPU without waiting for it. mjr_readPixels copies the
    framebuffer into memory at once, so the camera's thread first sat out the GPU's queue:
    9 ms of a frame with the GPU to itself, 12-13 ms with Ito's depth model or MASt3R-SLAM on
    the same GPU, where with the draw (17 ms then) a frame took more than its 33 ms and the next
    exposure was dropped (29.4-29.8 frames a second). Here each frame's pixels are copied on the
    GPU into a pixel buffer of its own (SLOTS of them), behind a fence; the thread collects
    them, in order, once the GPU is past the fence, and draws the next frame meanwhile."""

    SLOTS = 3

    def __init__(self, rect, context):
        from OpenGL import GL
        from OpenGL.raw.GL.VERSION import GL_1_0, GL_1_5

        self.gl, self.rect, self.context = GL, rect, context
        # Raw: with a pixel buffer bound, the pointer is an offset into it (or the destination).
        self.read_pixels, self.buffer_data = GL_1_0.glReadPixels, GL_1_5.glGetBufferSubData
        self.shape = (rect.height, rect.width, 3)
        self.size = rect.width * rect.height * 3
        self.free = collections.deque(int(b) for b in np.atleast_1d(GL.glGenBuffers(self.SLOTS)))
        for buffer in self.free:
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, buffer)
            GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, self.size, None, GL.GL_STREAM_READ)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        self.pending = collections.deque()  # (buffer, fence, item), oldest first
        self.changed = threading.Condition()  # pending or free, from the thread that collects

    def full(self):
        with self.changed:
            return not self.free

    def wait(self, ready, timeout):
        with self.changed:
            return self.changed.wait_for(ready, timeout)

    def start(self, item):
        """Copy the views just drawn, as mjr_readPixels reads the offscreen framebuffer: resolved
        from its multisampled buffer first, RGB rows bottom first."""
        GL, context, r = self.gl, self.context, self.rect
        box = (r.left, r.bottom, r.left + r.width, r.bottom + r.height)
        if context.offFBO_r:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, context.offFBO)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, context.offFBO_r)
            GL.glBlitFramebuffer(*box, *box, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, context.offFBO_r)
        else:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, context.offFBO)
        with self.changed:
            buffer = self.free.popleft()
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, buffer)
        self.read_pixels(
            r.left, r.bottom, r.width, r.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, ctypes.c_void_p(0)
        )
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        mujoco.mjr_restoreBuffer(context)
        fence = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        GL.glFlush()
        with self.changed:
            self.pending.append((buffer, fence, item))
            self.changed.notify_all()

    def collect(self, wait):
        """The frames the GPU has copied, oldest first, as (pixels, item, seconds taking the
        pixels); waiting up to `wait` seconds for the oldest. On one thread only: the drawing
        thread's or one whose context shares its objects."""
        GL, out = self.gl, []
        while True:
            with self.changed:
                if not self.pending:
                    break
                buffer, fence, item = self.pending[0]
            status = GL.glClientWaitSync(fence, 0, int(wait * 1e9))
            if status == GL.GL_WAIT_FAILED:
                raise RuntimeError("head camera: waiting on the GPU's copy failed")
            if status == GL.GL_TIMEOUT_EXPIRED:
                break
            with self.changed:
                self.pending.popleft()
            GL.glDeleteSync(fence)
            began = time.perf_counter()
            pixels = np.empty(self.shape, np.uint8)
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, buffer)
            self.buffer_data(
                GL.GL_PIXEL_PACK_BUFFER, 0, self.size, pixels.ctypes.data_as(ctypes.c_void_p)
            )
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
            with self.changed:
                self.free.append(buffer)
                self.changed.notify_all()
            out.append((pixels, item, time.perf_counter() - began))
            wait = 0
        return out


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
        self.out = queue.Queue()
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

    POLL = 0.002  # seconds the thread waits on the GPU's copy before it looks for a new shot

    def frames(self):
        """Each shot drawn as soon as physics takes it, the newest if drawing fell behind, as a
        camera's frame leaves once it is exposed. Drawn on a 30 Hz schedule of its own instead,
        a shot waited up to a frame for its turn, at a phase that moved only when a frame came
        late. The pixels come back from the GPU behind the draw (Readback) and are remapped and
        packed on a thread of their own (converting), so this thread only draws: it is free for
        the next shot while the GPU, shared with the pilot's depth model, finishes this one.
        `late`: frames begun more than a frame after their exposure."""
        period = 1.0 / max(1, FPS)
        kept = mujoco.MjData(self.model)
        readback = Readback(self.fisheye.rect, self.renderer._mjr_context)
        converter = threading.Thread(target=self.converting, name="head-camera-out", daemon=True)
        converter.start()
        self.costs, self.copies, self.late, self.dropped = [], [], 0, 0
        window = self.copier()
        copier = None
        if window is not None:
            copier = threading.Thread(
                target=self.copying, args=(readback, window), name="head-camera-copy", daemon=True
            )
            copier.start()
        print(
            "== head camera: pixels taken on "
            + ("a thread of their own" if copier else "the drawing thread, between views"),
            flush=True,
        )
        try:
            while not self.stopping.is_set():
                with self.fresh:
                    if copier is not None or not readback.pending:
                        self.fresh.wait_for(
                            lambda: self.shots or self.stopping.is_set(), timeout=0.5
                        )
                    state = self.shots[-1] if self.shots else None
                    self.dropped += max(0, len(self.shots) - 1)
                    self.shots.clear()
                if self.stopping.is_set():
                    break
                if state is not None:
                    start = time.perf_counter()
                    while readback.full():
                        if copier is not None:
                            readback.wait(lambda: readback.free, 1.0)
                        else:
                            self.converted(readback.collect(1.0))
                    # The exposure on CLOCK_MONOTONIC, the clock robotd and tofd stamp with: the
                    # physics time of the state drawn.
                    exposed = self.world.epoch_ns + round(state[0] * 1e9)
                    self.late += time.monotonic_ns() - exposed > period * 1e9
                    self.pose(kept, state)
                    self.scene(kept)
                    # Without a copier, a frame still on the GPU goes on as soon as it is copied,
                    # not a whole draw later: a shot can come while the last is being copied
                    # (physics runs in batches), and a draw takes 20 ms or more.
                    self.fisheye.views_drawn(
                        self.renderer,
                        None if copier else lambda: self.converted(readback.collect(0)),
                    )
                    readback.start(exposed)
                    self.costs.append(time.perf_counter() - start)
                if copier is None:
                    self.converted(readback.collect(0 if state is not None else self.POLL))
        finally:
            self.stopping.set()
            if copier is not None:
                copier.join(timeout=2)
                import glfw

                glfw.destroy_window(window)
            self.out.put(None)
            converter.join(timeout=2)

    def copier(self):
        """With GLFW, a hidden window whose context shares the renderer's objects (its pixel
        buffers and fences), for a thread that takes each frame's pixels the moment the GPU has
        them: the drawing thread, as a rule busy drawing the next frame, took them between its
        views, up to a view late. None with another GL (EGL, headless)."""
        context = getattr(self.renderer._gl_context, "_context", None)
        if os.environ.get("MUJOCO_GL") != "glfw" or context is None:
            return None
        import glfw

        glfw.window_hint(glfw.VISIBLE, 0)
        window = glfw.create_window(16, 16, "head camera copies", None, context)
        glfw.default_window_hints()
        return window or None

    def copying(self, readback, window):
        import glfw

        glfw.make_context_current(window)
        try:
            while not self.stopping.is_set():
                if readback.wait(lambda: readback.pending or self.stopping.is_set(), 0.5):
                    self.converted(readback.collect(1.0))
        finally:
            glfw.make_context_current(None)

    def converted(self, collected):
        for pixels, exposed, copy in collected:
            self.copies.append(copy)
            self.out.put((pixels, exposed))

    def converting(self):
        """Each frame the GPU has copied, in order: remapped to the fisheye, packed and
        published. Reported every 10 s: `render`, the drawing thread's time for a frame (pose,
        scene, views); `copy`, its time to take a frame's pixels once the GPU has them; `out`,
        exposure to published, the frame's age as it leaves for mediad."""

        def ms(values):
            v = sorted(values) or [0.0]
            return f"p50 {v[len(v) // 2]:.1f} ms p95 {v[int(len(v) * 0.95)]:.1f} ms"

        outs, window = [], time.perf_counter()
        while (job := self.out.get()) is not None:
            pixels, exposed = job
            self.publish(self.fisheye.remap(pixels), exposed)
            with self.lock:
                self.seq += 1
                self.fresh.notify_all()
            outs.append((time.monotonic_ns() - exposed) / 1e6)
            done = time.perf_counter()
            if done - window >= 10:
                costs, copies = self.costs, self.copies
                late, dropped = self.late, self.dropped
                self.costs, self.copies, self.late, self.dropped = [], [], 0, 0
                print(
                    f"== head camera: {len(outs) / (done - window):.1f} fps (target {FPS}), "
                    f"render {ms(c * 1e3 for c in costs)}, copy {ms(c * 1e3 for c in copies)}, "
                    f"out {ms(outs)}, late {late}, dropped {dropped}",
                    flush=True,
                )
                outs, window = [], done

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
