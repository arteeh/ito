"""Display-thread loop. Providers and the input sink must be nonblocking."""

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import moderngl
import pygame

from ito.render import GaussianRenderer, SceneSource, perspective, pose
from ito.render.scene import FloatArray

from .input import DesktopInput, PilotInput
from .overlay import Overlay, PilotStatus

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopState:
    robot_camera: FloatArray = field(default_factory=pose)
    status: PilotStatus = field(default_factory=PilotStatus)


class DesktopWindow:
    def __init__(self, size=(1280, 720), *, fps: int = 90, fov: float = 70,
                 speed: float = 1.5, capture_dir: Path = Path("captures")):
        self.context = self.renderer = self.overlay = self.input = None
        if (min(size) < 64 or fps < 1 or not 1 <= fov <= 175
                or not math.isfinite(speed) or speed <= 0):
            raise ValueError("Invalid window size, frame rate, field of view or movement speed")
        self.fps, self.fov, self.capture_dir = fps, math.radians(fov), capture_dir
        self.capture_number = 0
        pygame.display.init()
        pygame.font.init()
        try:
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
            pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
            try:
                pygame.display.set_mode(size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE, vsync=0)
                # glcontext resolves runtime GL libraries without requiring development symlinks.
                self.context = moderngl.create_context(require=430, libgl=None)
            except (pygame.error, OSError, ValueError) as exc:
                raise RuntimeError("Ito needs an OpenGL 4.3 display and driver; Mesa llvmpipe "
                                   f"under xvfb-run is supported. {exc}") from exc
            pygame.display.set_caption("Ito — Desktop pilot")
            self.screen = self.context.screen
            self.size = size
            self.renderer = GaussianRenderer(self.context)
            self.overlay = Overlay(self.context)
            self.input = DesktopInput(speed)
            log.info("OpenGL %s | %s", self.context.info["GL_VERSION"], self.context.info["GL_RENDERER"])
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
        pixels = self.screen.read(viewport=(0, 0, *self.size), components=3, alignment=1)
        image = pygame.image.frombytes(pixels, self.size, "RGB")
        pygame.image.save(pygame.transform.flip(image, False, True), path)
        log.info("Capture: %s", path)
        return path

    def run(self, source: SceneSource, *,
            state: Callable[[], DesktopState] = DesktopState,
            on_input: Callable[[PilotInput], None] | None = None,
            max_frames: int = 0, metrics: TextIO | None = None) -> None:
        """Keep drawing the last scene while source.poll() returns None.

        The app supplies fresh camera/status snapshots and queues PilotInput for
        its independent link loop. E-stop confirmation comes only from status.
        """
        clock = pygame.time.Clock()
        revision = None
        captured_at = None
        frames = 0
        next_metric = 0.0
        request = None
        previous = time.monotonic()
        while not max_frames or frames < max_frames:
            now = time.monotonic()
            pilot = self.input.poll(now - previous)
            previous = now
            if on_input is not None:
                on_input(pilot)
            if pilot.quit:
                break
            if pilot.commands:
                command = pilot.commands[-1].replace("_", "-").upper()
                request = (f"{command} requested" if on_input is not None
                           else f"{command}: no robot connected")
                log.info("Command: %s", pilot.commands[-1])
            frame = source.poll()
            if frame is not None and frame.revision != revision:
                self.renderer.upload(frame.gaussians)
                revision, captured_at = frame.revision, frame.captured_at
                log.info("Scene revision %s: %s Gaussians, SH degree %s", revision,
                         self.renderer.count, frame.gaussians.sh_degree)
            current = state()
            size = pygame.display.get_window_size()
            if min(size) > 0:
                self.size = size
            projection = perspective(self.fov, self.size[0] / self.size[1])
            self.renderer.draw(current.robot_camera, pilot.head, projection, self.screen,
                               viewport=(0, 0, *self.size))
            age = None if captured_at is None else max(0, now - captured_at)
            self.overlay.draw(current.status, clock.get_fps(), self.renderer.count, age,
                              self.input.captured, request)
            capture = None
            if pilot.screenshot:
                try:
                    capture = str(self._capture())
                except OSError as exc:
                    log.error("Could not save capture: %s", exc)
            pygame.display.flip()
            frames += 1
            if metrics is not None and (now >= next_metric or pilot.commands or capture):
                metrics.write(json.dumps({"frame": frames, "time": now, "fps": clock.get_fps(),
                                          "frame_ms": (time.monotonic() - now) * 1000,
                                          "gaussians": self.renderer.count, "revision": revision,
                                          "head": pilot.head.tolist(), "movement": pilot.movement,
                                          "look": pilot.look, "active": pilot.active,
                                          "commands": pilot.commands, "capture": capture,
                                          "size": self.size,
                                          "renderer": self.context.info["GL_RENDERER"]}) + "\n")
                metrics.flush()
                next_metric = now + 0.5
            clock.tick(self.fps)

    def close(self) -> None:
        for resource in (self.input, self.overlay, self.renderer):
            if resource is not None:
                resource.close()
        if self.context is not None:
            self.context.release()
        pygame.quit()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
