import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import pygame

from ito import clock, diagnostics, recon, trace
from ito.render import perspective, pose
from ito.render.anchor import AnchorGlide
from ito.render.camera import CameraFeed
from ito.render.gpu import GPU
from ito.render.performance import Performance
from ito.render.pose import FloatArray
from ito.render.video import VideoPanel
from ito.render.vk import Device, Image, Presenter, Swapchain, native_window
from ito.tofmesh.render import COLOUR

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


NVIDIA = 0x10DE


class WindowClosed(Exception):
    """The pilot closed the window (its X, Alt+F4) while it waited for something: Ito quits."""


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
        self.device = self.swapchain = self.present = self.scene = None
        self.renderer = self.overlay = self.input = self.video_panel = self.camera = None
        self.renderers = {}
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
        # Vulkan presents to the window: SDL must not put a renderer of its own on it.
        os.environ["SDL_FRAMEBUFFER_ACCELERATION"] = "0"
        pygame.display.init()
        try:
            pygame.display.set_mode(size, pygame.RESIZABLE)
            pygame.display.set_caption("Ito — Desktop pilot")
            self.size = size
            try:
                device = Device(native_window())
            except (RuntimeError, OSError) as exc:
                raise RuntimeError(f"Ito needs a GPU and driver with Vulkan 1.3. {exc}") from exc
            self._open(device)
            if device.properties.vendorID == NVIDIA:
                self.gpu = GPU()
                self.gpu.start()
            self.input = DesktopInput(speed)
            if self.input.script is not None:
                self.input.script.controls = self.overlay.layout
        except Exception:
            self.close()
            raise

    def _open(self, device):
        """Everything that draws, on this device."""
        self.device = device
        self.swapchain = Swapchain(device)
        self.present = Presenter(device)
        # The scenes' renderers, by ito.recon.Method.scene, made as sessions first need them.
        self.renderers = {"mesh": recon.renderer("mesh", device)}
        self.renderer = self.renderers["mesh"]
        self.video_panel = VideoPanel(device, COLOUR)
        self.camera = CameraFeed(device)
        self.live = None  # capture time of the newest frame, until its pose is known
        if self.overlay is None:
            self.overlay = Overlay(device)

    def _close_device(self):
        if self.device is not None:
            self.device.idle()
        for resource in (
            self.video_panel,
            self.camera,
            *self.renderers.values(),
            self.scene,
            self.present,
            self.swapchain,
        ):
            if resource is not None:
                resource.close()
        self.video_panel = self.camera = self.renderer = self.scene = None
        self.renderers = {}
        self.present = self.swapchain = None

    def reopen(self, device):
        """Draw on another device from now on (the one a headset's runtime made)."""
        self._close_device()
        self.overlay.use(device)  # its textures freed on the old device, still there
        old, self.device = self.device, None
        old.close()
        self._open(device)

    def _view_image(self, size):
        """The offscreen image the desktop's view is drawn into, at the window's size."""
        if self.scene is None or self.scene.size != tuple(size):
            if self.scene is not None:
                self.device.defer(self.scene)
            self.scene = Image(self.device, size, COLOUR, sampled=True, attachment=True)
        return self.scene

    def show(self, image):
        """Present image (acquired from the swapchain, drawn in the frame being recorded) and
        submit the frame; None (a minimised window): submit only."""
        if image is not None:
            self.swapchain.finish(self.device.frame)
            self.device.submit(self.swapchain.wait(), self.swapchain.signal())
            self.swapchain.present()
        else:
            self.device.submit()

    def show_menu(self, clear):
        """A frame of only ImGui (its frame ended here) over `clear`: the menus."""
        frame = self.device.begin()
        self.device.collect()
        image = self.swapchain.acquire(pygame.display.get_window_size())
        if image is not None:
            with frame.render([image], clear=clear):
                pass
        self.overlay.render(image)
        self.show(image)

    def _capture(self, frame, image):
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        while True:
            self.capture_number += 1
            path = self.capture_dir / f"capture-{self.capture_number:03d}.png"
            if not path.exists():
                break

        def save(pixels, size):
            try:
                rgb = self._rgb(pixels, size).tobytes()
                pygame.image.save(pygame.image.frombytes(rgb, size, "RGB"), path)
                log.info("Capture: %s", path)
            except (OSError, pygame.error) as exc:
                log.error("Could not save capture: %s", exc)

        frame.read(image, save)
        return path

    def _rgb(self, pixels, size):
        """A display image's texels as read back: RGB rows, top first (a view of them)."""
        texels = pixels.reshape(size[1], size[0], 4)
        return texels[..., 2::-1] if self.swapchain.bgr else texels[..., :3]

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

    def use_scene(self, source):
        """Draw the session's scene with its renderer (ito.recon.Method.scene), the renderer it
        replaces left empty. Each frame, before its updates: a session sets its scene as it
        connects, after the window began drawing."""
        kind = getattr(source, "scene", "mesh")
        if kind not in self.renderers:
            self.renderers[kind] = recon.renderer(kind, self.device)
        if self.renderers[kind] is not self.renderer:
            self.renderer.clear()
            self.renderer = self.renderers[kind]

    def _draw(self, source, state, on_input, dispatch):
        recorder = from_env()
        self.performance = Performance()
        try:
            self._draw_loop(source, state, on_input, dispatch, recorder)
        finally:
            self.device.idle()
            for frame in self.device.frames:
                frame.wait()  # the last screen grabs reach the recorder
            if recorder is not None:
                recorder.close()
            self.renderer.clear()  # the next session or a menu shows none of this one's mesh

    def _record(self, recorder: Recorder, current, frame, image):
        """The screen grab is copied out in the frame and collected when the frame is done (a
        frame or two later): the frame loop never waits for it."""
        recorder.camera(current.video, current.video_time)
        now = clock.now()
        if recorder.due(now):
            recorder.taken(now)
            frame.read(image, lambda pixels, size: recorder.screen(now, self._rgb(pixels, size)))

    def _draw_loop(self, source, state, on_input, dispatch, recorder):
        pacing = pygame.time.Clock()
        revision = None
        captured_at = None
        frames = 0
        request = None
        due = clock.now()
        while True:
            now = clock.now()
            began = clock.now_ns()
            events, pilot = dispatch.frame()
            io = self.overlay.begin(events, pygame.display.get_window_size(), self.input.captured)
            dispatch.ui(io, (), events)
            on_input(pilot)
            if pilot.quit or self.overlay.leave:
                self.overlay.render()
                break
            if pilot.commands:
                request = pilot.commands[-1].replace("_", "-").upper() + " requested"
                log.info("Command: %s", pilot.commands[-1])
            trace.add("frame.input", began, clock.now_ns())
            gpu = self.device.begin()
            self.device.collect()
            applied = []
            self.use_scene(source)
            for _ in range(MESHES_PER_FRAME):
                with trace.span("frame.poll"):
                    frame = source.poll()
                if frame is None:
                    break
                with trace.span("frame.upload", trace.key(frame.captured_at)):
                    self.renderer.upload(frame)
                applied.append(frame)
                revision, captured_at = frame.revision, frame.captured_at
            gpu.mark("upload")
            current = self.glided(state(), now)
            with trace.span("frame.straighten", trace.key(current.video_time)):
                self.straighten(source, current)
            size = pygame.display.get_window_size()
            if min(size) > 0:
                self.size = size
            projection = perspective(self.fov, self.size[0] / self.size[1])
            with trace.span("frame.draw"):
                self.prepare_view(current, pilot.head, projection)
                scene = self._view_image(self.size)
                self.draw_view(current, pilot.head, projection, scene)
            gpu.mark("scene")
            with trace.span("frame.acquire"):
                image = self.swapchain.acquire(self.size)
            if image is not None:
                self.present(gpu, scene, image, upright=True, srgb=self.swapchain.srgb)
            started = clock.now_ns()
            visible_time = current.video_time if current.flat_video else captured_at
            age = None if visible_time is None else max(0, now - visible_time)
            self.overlay.draw(
                current.status,
                pacing.get_fps(),
                self.input.captured,
                request,
                target=image,
                performance=self.performance.last,
                driving=any(pilot.movement),
            )
            dispatch.ui(io, self.overlay.commands)
            self.overlay.commands.clear()
            trace.add("frame.overlay", started, clock.now_ns())
            gpu.mark("overlay")
            if image is not None:
                if recorder is not None:
                    self._record(recorder, current, gpu, image)
                if pilot.screenshot:
                    try:
                        self._capture(gpu, image)
                    except OSError as exc:
                        log.error("Could not save capture: %s", exc)
            with trace.span("frame.submit"):
                self.show(image)
            if hasattr(source, "display_completed"):  # MASt3R-SLAM yields the GPU to late frames
                source.display_completed(due, 1 / self.fps)
            with trace.span("frame.camera_wait"):
                self.camera.collect()
            started = clock.now_ns()
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
            trace.add("frame.statistics", started, clock.now_ns())
            trace.add("frame", began, clock.now_ns())
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
                with trace.span("frame.sleep"):
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
        """The newest camera frame to RGB on the GPU, on to the ToF mesh's colour (once the GPU
        has it: camera.collect()) and into the mesh's live projector as soon as the pose it was
        captured at is known (usually at once; a newer frame takes the place of one waiting)."""
        if current.video is not None and self.camera.update(
            current.video, current.video_time, current.video_intrinsics, current.video_lens,
            source.camera_frame,
        ):  # fmt: skip
            self.live = current.video_time
        if self.live is not None:
            posed = source.live_camera(self.live)
            if posed is not None:
                self.renderer.live(self.camera.colour, *posed)
                self.live = None

    def glided(self, current, now):
        return replace(current, robot_camera=self.glide(current.robot_camera, now))

    def prepare_view(self, current, head, projection, views=None):
        """Once per displayed frame, before its views: head is the viewer between the eyes (the
        one view on a desktop), projection covers every view's field, views are each view's
        projection."""
        if current.flat_video:
            self.video_panel.prepare(views or [projection])
        else:
            self.renderer.prepare(current.robot_camera, head, projection)

    def draw_view(self, current, head, projection, target, center=None):
        """One view into target (an offscreen image of format COLOUR, OpenGL's row order)."""
        if current.flat_video:
            self.video_panel.draw(
                self.camera.texture,
                current.video_time,
                head,
                projection,
                target,
                center=center,
                fov=current.video_fov,
            )
        else:
            self.renderer.draw(current.robot_camera, head, projection, target)

    def close(self) -> None:
        try:
            self._close_device()
            for resource in (self.input, self.overlay, self.gpu, self.device):
                if resource is not None:
                    resource.close()
            self.device = None
        finally:
            pygame.quit()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
