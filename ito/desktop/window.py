"""Independently paced SDL input and desktop drawing."""

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TextIO

import pygame
from OpenGL import GL

from ito import clock, diagnostics
from ito.reconstruction import SplatUpdate, default_budget
from ito.reconstruction.admission import PUBLICATION_PACKETS
from ito.render import GaussianRenderer, SceneSource, current_context, perspective, pose
from ito.render.anchor import AnchorGlide, CorrectionEase
from ito.render.gpu import GPU, MAX_SLOT_COUNT, slot_budget
from ito.render.performance import Performance
from ito.render.scene import FloatArray
from ito.render.video import VideoPanel

from .dispatch import DisplayDispatch
from .input import DesktopInput, PilotInput
from .overlay import Overlay, PilotStatus
from .record import Recorder, from_env, heading
from .settings import load_budget, save_budget

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopState:
    robot_camera: FloatArray = field(default_factory=pose)
    status: PilotStatus = field(default_factory=PilotStatus)
    video: object | None = field(default=None, repr=False, compare=False)
    video_time: float = 0
    video_fov: float | None = None  # The camera's horizontal field of view in radians.
    flat_video: bool = False
    # Fusion's world-from-odometry: the eye is drawn at correction @ robot_camera, eased in.
    correction: FloatArray = field(default_factory=pose)
    # Yaw of the robot camera's view in its odometry world, positive to the left: where
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
        max_splats: int | None = None,
    ):
        self.context = self.renderer = self.overlay = self.input = self.video_panel = None
        self.gpu = None
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
        self.ease = CorrectionEase()
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
            software = any(
                name in self.context.info["GL_RENDERER"].lower()
                for name in ("llvmpipe", "softpipe", "software")
            )
            buffer_limit = self.context.info["GL_MAX_SHADER_STORAGE_BLOCK_SIZE"]
            self.gpu = GPU() if "nvidia" in self.context.info["GL_VENDOR"].lower() else None
            free_bytes = self.gpu.free if self.gpu and self.gpu.library else None
            if "GL_NVX_gpu_memory_info" in self.context.extensions:
                available = int(GL.glGetIntegerv(0x9049)) * 1024
                free_bytes = available if free_bytes is None else min(available, free_bytes)
            self.splat_limit = (
                16_384
                if software
                else max(1, slot_budget(free_bytes, buffer_limit))
                if free_bytes is not None
                else min(MAX_SLOT_COUNT, buffer_limit // 64, 262_144)
            )
            self.renderer = GaussianRenderer(self.context, live_capacity=self.splat_limit)
            self.video_panel = VideoPanel(self.context)
            self.max_splats = min(
                self.splat_limit,
                max_splats
                or load_budget(
                    default_budget(
                        self.context.info["GL_RENDERER"],
                        free_bytes=free_bytes,
                        buffer_limit=buffer_limit,
                    )
                ),
            )
            if self.gpu is not None:
                self.gpu.start()
            self.overlay = Overlay(self.context, self.max_splats)
            self.input = DesktopInput(speed)
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
        source: SceneSource,
        *,
        state: Callable[[], DesktopState] = DesktopState,
        on_input: Callable[[PilotInput], None] | None = None,
        on_sample: Callable[[PilotInput], None] | None = None,
        max_frames: int = 0,
        metrics: TextIO | None = None,
        save_settings: Callable[[int], None] = save_budget,
    ) -> None:
        """Poll input at 90 Hz while the display draws the latest snapshot."""
        dispatch = DisplayDispatch(self)
        dispatch.run(
            lambda: self._draw(
                source, state, on_input, max_frames, metrics, save_settings, dispatch
            ),
            on_sample,
        )

    def _draw(self, source, state, on_input, max_frames, metrics, save_settings, dispatch):
        recorder = from_env()
        self.performance = Performance()
        try:
            self._draw_loop(
                source, state, on_input, max_frames, metrics, save_settings, dispatch, recorder
            )
        finally:
            summary = self.performance.close(metrics)
            if recorder is not None:
                recorder.close(summary)

    def _record(self, recorder: Recorder, now, current, pilot, frame, age, revision):
        recorder.camera(current.video, current.video_time, now, current.robot_camera)
        if not recorder.due(now):
            return
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
        GL.glReadBuffer(GL.GL_BACK)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        pixels = GL.glReadPixels(0, 0, *self.size, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        recorder.pilot(
            now,
            pixels,
            self.size,
            frame=frame,
            revision=revision,
            gaussians=self.renderer.count,
            flat_video=current.flat_video,
            capture_age_ms=None if age is None else round(age * 1000, 1),
            yaw=heading(current.robot_camera),
            head_yaw=heading(pilot.head),
            move=[round(v, 2) for v in pilot.movement],
            look=[round(v, 2) for v in pilot.look],
            performance=self.performance.last,
        )

    def _draw_loop(
        self, source, state, on_input, max_frames, metrics, save_settings, dispatch, recorder
    ):
        live = hasattr(source, "set_max_splats")
        if live:
            if source.max_splats > self.splat_limit:
                raise ValueError(f"This GPU supports at most {self.splat_limit:,} splats")
            self.overlay.max_splats = source.max_splats
        pacing = pygame.time.Clock()
        revision = None
        pending = None
        captured_at = None
        frames = 0
        next_metric = 0.0
        request = None
        due = clock.now()
        while not max_frames or frames < max_frames:
            now = clock.now()
            events, pilot = dispatch.frame()
            io = self.overlay.begin(events, pygame.display.get_window_size(), self.input.captured)
            dispatch.ui(io, (), events)
            if on_input is not None:
                on_input(pilot)
            if pilot.quit or self.overlay.leave:
                self.overlay.draw(PilotStatus(), 0, 0, None, False, request)
                break
            if pilot.commands:
                command = pilot.commands[-1].replace("_", "-").upper()
                request = (
                    f"{command} requested"
                    if on_input is not None
                    else f"{command}: no robot connected"
                )
                log.info("Command: %s", pilot.commands[-1])
            applied = []
            # A bounded drain never lets a fast producer take over the display loop.
            for _ in range(PUBLICATION_PACKETS):
                frame = pending if pending is not None else source.poll()
                if frame is None:
                    break
                if isinstance(frame, SplatUpdate):
                    if not self.renderer.apply(frame):
                        pending = frame
                        break
                elif frame.revision != revision:
                    self.renderer.upload(frame.gaussians)
                if isinstance(frame, SplatUpdate):
                    applied.append(frame)
                pending = None
                revision, captured_at = frame.revision, frame.captured_at
            current = self.glided(state(), now, pilot.head)
            size = pygame.display.get_window_size()
            if min(size) > 0:
                self.size = size
            projection = perspective(self.fov, self.size[0] / self.size[1])
            self.draw_view(current, pilot.head, projection, self.screen, (0, 0, *self.size))
            visible_time = current.video_time if current.flat_video else captured_at
            age = None if visible_time is None else max(0, now - visible_time)
            if recorder is not None:
                self._record(recorder, now, current, pilot, frames, age, revision)
            budget = self.overlay.draw(
                current.status,
                pacing.get_fps(),
                self.renderer.count,
                age,
                self.input.captured,
                request,
                live=live,
                performance=self.performance.last,
                driving=any(pilot.movement),
            )
            dispatch.ui(io, self.overlay.commands)
            self.overlay.commands.clear()
            if budget is not None and budget > self.splat_limit:
                self.overlay.error = f"This GPU supports at most {self.splat_limit:,} splats"
                budget = None
            if budget is not None:
                source.set_max_splats(budget)
                try:
                    save_settings(budget)
                except OSError as exc:
                    self.overlay.error = f"Could not save setting: {exc}"
            capture = None
            if pilot.screenshot:
                try:
                    capture = str(self._capture())
                except OSError as exc:
                    log.error("Could not save capture: %s", exc)
            self.context.finish()
            pygame.display.flip()
            if hasattr(source, "display_completed"):
                source.display_completed(due, 1 / self.fps)
            statistics = self.performance.worker_statistics | (
                source.statistics() if hasattr(source, "statistics") else {}
            )
            statistics.update(
                camera_hz=current.status.rates[1],
                pilot_input_hz=current.status.rates[0],
                flat_video=current.flat_video,
                reconstruction_status=current.status.reconstruction,
                link=current.status.link,
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
            if metrics is not None and (now >= next_metric or pilot.commands or capture):
                metrics.write(
                    json.dumps(
                        {
                            "frame": frames,
                            "time": now,
                            "fps": pacing.get_fps(),
                            "frame_ms": (clock.now() - now) * 1000,
                            "gaussians": self.renderer.count,
                            "revision": revision,
                            "head": pilot.head.tolist(),
                            "movement": pilot.movement,
                            "look": pilot.look,
                            "active": pilot.active,
                            "commands": pilot.commands,
                            "capture": capture,
                            "size": self.size,
                            "renderer": self.context.info["GL_RENDERER"],
                            "link": current.status.link,
                            "robot": current.status.robot,
                            "status": current.status.detail,
                            "reconstruction": current.status.reconstruction,
                            "flat_video": current.flat_video,
                            "video_capture_time": current.video_time,
                            "e_stop": current.status.e_stop,
                            "rtt_ms": current.status.latency_ms,
                            "pilot_input_to_robot_ms": current.status.input_latency_ms,
                            "capture_to_splat_visible_ms": self.performance.latest_latency,
                            "performance": performance,
                            "scene_capture_time": captured_at,
                        }
                    )
                    + "\n"
                )
                metrics.flush()
                next_metric = now + 0.5
            due += 1 / self.fps
            remaining = due - clock.now()
            if remaining > 0:
                time.sleep(remaining)
            else:
                # Count the late frame, then skip obsolete starts; don't catch up
                # by submitting a burst of frames the pilot cannot see.
                skipped = int(-remaining * self.fps) + 1
                self.performance.skipped_deadlines(skipped)
                due += skipped / self.fps
                remaining = due - clock.now()
                if remaining > 0:
                    time.sleep(remaining)
            pacing.tick(0)

    def glided(self, current, now, head=None):
        """The eye to draw now: glided between updates, fusion's corrections eased in."""
        anchor = self.glide(current.robot_camera, now)
        return replace(current, robot_camera=self.ease(current.correction, anchor, head, now))

    def draw_view(self, current, head, projection, target, viewport, sort=True, center=None):
        """Draw one eye; center is the pose between both eyes, for a stereo panel."""
        if current.flat_video:
            self.video_panel.draw(
                current.video,
                current.video_time,
                head,
                projection,
                target,
                viewport,
                center=center,
                fov=current.video_fov,
            )
        else:
            self.renderer.draw(
                current.robot_camera, head, projection, target, viewport=viewport, sort=sort
            )

    def close(self) -> None:
        for resource in (self.input, self.overlay, self.video_panel, self.renderer, self.gpu):
            if resource is not None:
                resource.close()
        if self.context is not None:
            self.context.release()
        pygame.quit()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
