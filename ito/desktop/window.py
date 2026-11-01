import ctypes
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import pygame
from OpenGL import GL

from ito import clock, diagnostics
from ito.render import current_context, perspective, pose
from ito.render.anchor import AnchorGlide
from ito.render.camera import CameraFeed
from ito.render.gpu import GPU
from ito.render.performance import Performance
from ito.render.pose import FloatArray
from ito.render.video import VideoPanel
from ito.tofmesh.render import MeshRenderer

from .dispatch import DisplayDispatch
from .input import DesktopInput, PilotInput
from .overlay import Overlay, PilotStatus
from .record import Recorder, from_env

log = logging.getLogger(__name__)
MESHES_PER_FRAME = 2


@dataclass(frozen=True)
class DesktopState:
    robot_camera: FloatArray = field(default_factory=pose)
    status: PilotStatus = field(default_factory=PilotStatus)
    # The latest camera frame as decoded (an av.VideoFrame), the camera's pinhole and lens.
    video: object | None = field(default=None, repr=False, compare=False)
    video_time: float = 0
    video_intrinsics: object | None = None
    video_lens: object | None = None
    video_fov: float | None = None
    flat_video: bool = False
    # Yaw of the robot camera's view in the map's world, positive to the left: where
    # recentering turns the pilot's view, so the robot keeps looking where it looks.
    robot_heading: float = 0.0


class DesktopWindow:
    def __init__(
        self,
        size=(1280, 720),
        *,
        fps: int = 90,
        fov: float = 70,
        speed: float = 1.5,
        capture_dir: Path = Path("captures"),
    ):
        self.context = self.renderer = self.overlay = self.input = self.video_panel = None
        self.camera = None
        self.gpu = None
        self.readback = None  # (pixel buffer, its size, pending (time, size) or None)
        if (
            min(size) < 64
            or fps < 1
            or not 1 <= fov <= 175
            or not math.isfinite(speed)
            or speed <= 0
        ):
            raise ValueError("Invalid window size, frame rate, field of view or movement speed")
        self.fps, self.fov, self.capture_dir = fps, math.radians(fov), capture_dir
        self.capture_number = 0
        self.glide = AnchorGlide()
        pygame.display.init()
        try:
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(
                pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
            )
            pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
            try:
                pygame.display.set_mode(
                    size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE, vsync=0
                )
                self.context = current_context()
            except (pygame.error, OSError, ValueError) as exc:
                raise RuntimeError(
                    "Ito needs an OpenGL 4.3 display and driver; Mesa llvmpipe "
                    f"under xvfb-run is supported. {exc}"
                ) from exc
            pygame.display.set_caption("Ito — Desktop pilot")
            self.screen = self.context.screen
            self.size = size
            self.gpu = GPU() if "nvidia" in self.context.info["GL_VENDOR"].lower() else None
            self.renderer = MeshRenderer(self.context)
            self.video_panel = VideoPanel(self.context)
            self.camera = CameraFeed(self.context)
            if self.gpu is not None:
                self.gpu.start()
            self.overlay = Overlay(self.context)
            self.input = DesktopInput(speed)
            if self.input.script is not None:
                self.input.script.controls = self.overlay.layout
            log.info(
                "OpenGL %s | %s", self.context.info["GL_VERSION"], self.context.info["GL_RENDERER"]
            )
        except Exception:
            self.close()
            raise

    def _capture(self) -> Path:
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        while True:
            self.capture_number += 1
            path = self.capture_dir / f"capture-{self.capture_number:03d}.png"
            if not path.exists():
                break
        # SDL owns the default framebuffer; select its back buffer explicitly.
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
        GL.glReadBuffer(GL.GL_BACK)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        pixels = GL.glReadPixels(0, 0, *self.size, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        image = pygame.image.frombytes(pixels, self.size, "RGB")
        pygame.image.save(pygame.transform.flip(image, False, True), path)
        log.info("Capture: %s", path)
        return path

    def run(
        self,
        source,
        *,
        state: Callable[[], DesktopState],
        on_input: Callable[[PilotInput], None],
        on_sample: Callable[[PilotInput], None],
    ) -> None:
        dispatch = DisplayDispatch(self)
        self.input.heading = lambda: state().robot_heading
        dispatch.run(lambda: self._draw(source, state, on_input, dispatch), on_sample)

    def _draw(self, source, state, on_input, dispatch):
        recorder = from_env()
        self.performance = Performance()
        try:
            self._draw_loop(source, state, on_input, dispatch, recorder)
        finally:
            if self.readback is not None:  # the screen grab's pixel buffer (its last grab unread)
                GL.glDeleteBuffers(1, [self.readback[0]])
                self.readback = None
            if recorder is not None:
                recorder.close()

    def _record(self, recorder: Recorder, current):
        """The screen grab reads into a pixel buffer and is collected a frame later (the frame
        loop's finish has completed it by then): a direct glReadPixels waited for the GPU in the
        frame loop, 4-26 ms, and recorded sessions missed 2-3 times as many 90 Hz frames as the
        same session unrecorded."""
        recorder.camera(current.video, current.video_time)
        if self.readback is not None and self.readback[2] is not None:
            buffer, length, (taken, size) = self.readback
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, buffer)
            address = GL.glMapBufferRange(GL.GL_PIXEL_PACK_BUFFER, 0, length, GL.GL_MAP_READ_BIT)
            pixels = None
            if address:  # a failed mapping returns null: that grab is skipped
                pixels = ctypes.string_at(address, length)
                GL.glUnmapBuffer(GL.GL_PIXEL_PACK_BUFFER)
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
            self.readback = (buffer, length, None)
            if pixels is not None:
                recorder.screen(taken, pixels, size)
        now = clock.now()
        if not recorder.due(now):
            return
        size = tuple(self.size)
        length = size[0] * size[1] * 3
        if self.readback is None or self.readback[1] != length:
            if self.readback is not None:
                GL.glDeleteBuffers(1, [self.readback[0]])
            buffer = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, buffer)
            GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, length, None, GL.GL_STREAM_READ)
            self.readback = (buffer, length, None)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
        GL.glReadBuffer(GL.GL_BACK)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self.readback[0])
        GL.glReadPixels(0, 0, *size, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        self.readback = (self.readback[0], length, (now, size))

    def _draw_loop(self, source, state, on_input, dispatch, recorder):
        pacing = pygame.time.Clock()
        revision = None
        captured_at = None
        frames = 0
        request = None
        due = clock.now()
        while True:
            now = clock.now()
            events, pilot = dispatch.frame()
            io = self.overlay.begin(events, pygame.display.get_window_size(), self.input.captured)
            dispatch.ui(io, (), events)
            on_input(pilot)
            if pilot.quit or self.overlay.leave:
                self.overlay.draw(PilotStatus(), 0, False, request)
                break
            if pilot.commands:
                request = pilot.commands[-1].replace("_", "-").upper() + " requested"
                log.info("Command: %s", pilot.commands[-1])
            applied = []
            for _ in range(MESHES_PER_FRAME):
                frame = source.poll()
                if frame is None:
                    break
                self.renderer.upload(frame)
                applied.append(frame)
                revision, captured_at = frame.revision, frame.captured_at
            current = self.glided(state(), now)
            self.straighten(source, current)
            size = pygame.display.get_window_size()
            if min(size) > 0:
                self.size = size
            projection = perspective(self.fov, self.size[0] / self.size[1])
            self.draw_view(current, pilot.head, projection, self.screen, (0, 0, *self.size))
            visible_time = current.video_time if current.flat_video else captured_at
            age = None if visible_time is None else max(0, now - visible_time)
            self.overlay.draw(
                current.status,
                pacing.get_fps(),
                self.input.captured,
                request,
                performance=self.performance.last,
                driving=any(pilot.movement),
            )
            dispatch.ui(io, self.overlay.commands)
            self.overlay.commands.clear()
            if recorder is not None:
                self._record(recorder, current)
            if pilot.screenshot:
                try:
                    self._capture()
                except OSError as exc:
                    log.error("Could not save capture: %s", exc)
            self.context.finish()
            pygame.display.flip()
            statistics = source.statistics()
            statistics.update(
                camera_hz=current.status.rates[1],
                pilot_input_hz=current.status.rates[0],
                flat_video=current.flat_video,
                reconstruction_status=current.status.reconstruction,
                link=current.status.link,
                vram=self.gpu.used() if self.gpu is not None else None,
            )
            performance = self.performance.completed(
                now,
                due,
                1 / self.fps,
                applied,
                scene=not current.flat_video,
                statistics=statistics,
                gpu_busy=self.gpu.busy if self.gpu is not None else None,
            )
            frames += 1
            diagnostics.event(
                "display_frame",
                interval=1,
                frame=frames,
                frame_ms=(clock.now() - now) * 1000,
                revision=revision,
                capture_time=visible_time,
                capture_age_ms=None if age is None else age * 1000,
                performance=performance,
            )
            due += 1 / self.fps
            remaining = due - clock.now()
            if remaining > 0:
                time.sleep(remaining)
            else:
                skipped = int(-remaining * self.fps) + 1
                self.performance.skipped_deadlines(skipped)
                due += skipped / self.fps
                remaining = due - clock.now()
                if remaining > 0:
                    time.sleep(remaining)
            pacing.tick(0)

    def straighten(self, source, current):
        """The newest camera frame to RGB on the GPU, and on to the ToF mesh's colour."""
        image = self.camera.update(
            current.video, current.video_time, current.video_intrinsics, current.video_lens
        )
        if image is not None:
            source.camera_frame(image, current.video_time)

    def glided(self, current, now):
        return replace(current, robot_camera=self.glide(current.robot_camera, now))

    def draw_view(self, current, head, projection, target, viewport, center=None):
        if current.flat_video:
            self.video_panel.draw(
                self.camera.texture,
                current.video_time,
                head,
                projection,
                target,
                viewport,
                center=center,
                fov=current.video_fov,
            )
        else:
            self.renderer.draw(current.robot_camera, head, projection, target, viewport=viewport)

    def close(self) -> None:
        for resource in (
            self.input,
            self.overlay,
            self.video_panel,
            self.camera,
            self.renderer,
            self.gpu,
        ):
            if resource is not None:
                resource.close()
        if self.context is not None:
            self.context.release()
        pygame.quit()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
