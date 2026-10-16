import json
import logging
import math
import time

import numpy as np
import pygame
import xr
from imgui_bundle import imgui
from OpenGL import GL
from xr.utils import GraphicsAPI, Matrix4x4f

from ito import clock, diagnostics
from ito.app.menu_layout import MenuLayout
from ito.desktop import DesktopWindow
from ito.desktop.window import DesktopState
from ito.reconstruction import SplatUpdate
from ito.reconstruction.admission import PUBLICATION_PACKETS
from ito.render import pose
from ito.render.performance import Performance
from ito.render.pose import quaternion

from .dispatch import XRDispatch
from .input import VALID, Actions
from .session import HEADSET_POLL, HEADSET_WAIT, HUD_PANEL_PIXELS, Session

log = logging.getLogger(__name__)
PANEL_LINGER = 0.5
PANEL_DISTANCE = 1.2
PANEL_DROP = math.radians(25)
PANEL_SIZE = (1.05, 1.05 * HUD_PANEL_PIXELS[1] / HUD_PANEL_PIXELS[0])
PANEL_FOLLOW = 0.25
PANEL_SLACK = (math.radians(20), math.radians(32))  # yaw, pitch
PANEL_SETTLED = math.radians(2)
PANEL_PITCH = (math.radians(-45), math.radians(40))


class PanelFollow:
    def __init__(self):
        self.position = None
        self.angles = [0.0, 0.0]
        self.following = [False, False]

    @staticmethod
    def gaze(head):
        forward = -head[:3, 2]
        flat = math.hypot(forward[0], forward[2])
        yaw = math.atan2(-forward[0], -forward[2]) if flat > 0.1 else None
        return yaw, math.asin(max(-1.0, min(1.0, float(forward[1]))))

    def update(self, head, dt, *, snap=False, hold=False):
        yaw, pitch = self.gaze(head)
        target = (yaw, min(PANEL_PITCH[1], max(PANEL_PITCH[0], pitch)))
        if self.position is None or snap:
            self.position = head[:3, 3].astype(np.float64)
            self.angles = [yaw if yaw is not None else self.angles[0], target[1]]
            self.following = [False, False]
        elif not hold:
            ease = 1 - math.exp(-max(dt, 0.0) / PANEL_FOLLOW)
            self.position += ease * (head[:3, 3] - self.position)
            for axis in (0, 1):
                if target[axis] is None:
                    continue
                error = math.remainder(target[axis] - self.angles[axis], math.tau)
                if abs(error) > PANEL_SLACK[axis]:
                    self.following[axis] = True
                elif abs(error) < PANEL_SETTLED:
                    self.following[axis] = False
                if self.following[axis]:
                    self.angles[axis] += ease * error
        return self.matrix()

    def matrix(self):
        """The panel's pose in the reference space; its +Z faces the anchor's eye point."""
        if self.position is None:
            return pose((0, -0.15, -1.2))
        yaw, pitch = self.angles
        result = pose(yaw=yaw, pitch=pitch - PANEL_DROP)
        result[:3, 3] = self.position - PANEL_DISTANCE * result[:3, 2]
        return result


class XRWindow(DesktopWindow):
    def __init__(self, *args, reference_space="seated", **kwargs):
        self.xr = None
        super().__init__(*args, **kwargs)
        try:
            self.xr = Session(self.context, reference_space, waiting=self._waiting)
            if self.overlay.theme.name == "waiting":
                self._finish_waiting()
            self.actions = Actions(self.xr)
            self.input.translate = False
            self.centered = False
            self.recenters = 0
            self.panel_until = 0.0
            self.panel_visible = True
            self.panel_hovered = False
            self.panel = PanelFollow()
            self.panel_size = PANEL_SIZE
            self.panel_pose = self.panel.matrix()
            pygame.display.set_caption("Ito — XR pilot controls")
        except xr.XrException as exc:
            self.close()
            raise RuntimeError(
                f"OpenXR initialization failed: {type(exc).__name__}: {exc}"
            ) from exc
        except BaseException:
            self.close()
            raise

    def _waiting(self, runtime, waited):
        pygame.display.set_caption("Ito — waiting for the headset")
        theme = self.overlay.theme
        if theme.name != "waiting":
            theme.show("waiting", "menu", intro=True)
            self.waiting_layout = MenuLayout(lambda _: None, theme)
        self.waiting_status = runtime, waited
        until = time.monotonic() + HEADSET_POLL
        cancelled = False
        pacing = pygame.time.Clock()
        while time.monotonic() < until or cancelled:
            events = pygame.event.get()
            if any(
                e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE)
                for e in events
            ):
                cancelled = True
                theme.leave()
            if self._waiting_frame(events):
                cancelled = True
                theme.leave()
            if cancelled and theme.showing()[1] <= 0:
                return False
            pacing.tick(self.fps)
        return True

    def _waiting_frame(self, events):
        size = pygame.display.get_window_size()
        self.overlay.begin(events, size, False)
        theme = self.overlay.theme
        runtime, waited = self.waiting_status
        _, alpha, leaving = theme.showing()
        with theme.faded(alpha, leaving):
            with self.waiting_layout.window("Ito", size):
                imgui.text(f"Waiting for a headset on {runtime}")
                imgui.text_disabled("Connect it: Steam Link, Virtual Desktop or its cable.")
                with theme.font("numeric"):
                    imgui.text_disabled(f"{waited:.0f} of {HEADSET_WAIT:.0f} s")
                cancel = self.waiting_layout.button("Cancel", "cancel")
        self.context.screen.use()
        self.context.viewport = (0, 0, *size)
        self.context.clear(*theme.color("clear"))
        self.overlay.render()
        pygame.display.flip()
        return cancel

    def _finish_waiting(self):
        self.overlay.theme.leave()
        pacing = pygame.time.Clock()
        while self.overlay.theme.showing()[1] > 0:
            self._waiting_frame(pygame.event.get())
            pacing.tick(self.fps)

    def aiming_hand(self):
        return next((hand for hand in ("right", "left") if hand in self.actions.aims), None)

    def _pointer(self):
        hand = self.aiming_hand()
        if hand is None:
            return (-10000, -10000, False)
        aim = np.linalg.inv(self.panel_pose) @ self.actions.aims[hand]
        origin, direction = aim[:3, 3], -aim[:3, 2]
        if direction[2] >= -1e-5:
            return (-10000, -10000, False)
        distance = -origin[2] / direction[2]
        hit = origin + distance * direction
        width, height = self.xr.panel.size
        x, y = (
            (hit[0] / self.panel_size[0] + 0.5) * width,
            (0.5 - hit[1] / self.panel_size[1]) * height,
        )
        inside = distance > 0 and 0 <= x < width and 0 <= y < height
        return (
            (float(x), float(y), self.actions.triggers.get(hand, False))
            if inside
            else (-10000, -10000, False)
        )

    def _capture_target(self, target, name):
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        # ModernGL readback holds the GIL; wait for the GPU through ctypes first
        # so a slow stereo capture cannot starve tracking and the link sender.
        GL.glFinish()
        pixels = target.read(components=3, alignment=1)
        image = pygame.image.frombytes(pixels, target.size, "RGB")
        path = self.capture_dir / f"capture-{self.capture_number:03d}-{name}.png"
        pygame.image.save(pygame.transform.flip(image, False, True), path)
        return str(path)

    def choose(self, menu):
        from .session import Swapchain

        class EmptyScene:
            def poll(self):
                return None

        if not hasattr(self, "menu_panel"):
            self.menu_panel = Swapchain(self.xr, (768, 768))
        runtime_panel = self.xr.panel
        self.xr.panel, self.panel_size = self.menu_panel, (1.05, 1.05)
        self.overlay.leave = False
        from ito.render.scene import GaussianBuffer

        self.renderer.upload(GaussianBuffer(np.empty((0, 4, 4), np.float32)))
        choice = None

        def draw(size):
            nonlocal choice
            selected = menu.draw(size, xr_mode=True)
            if selected is not None:
                choice = selected
            self.overlay.render()
            if choice:
                self.overlay.leave = True

        try:
            self.run(EmptyScene(), state=DesktopState, menu=draw)
        finally:
            self.xr.panel, self.panel_size = runtime_panel, PANEL_SIZE
            self.overlay.leave = False
        return choice

    def run(
        self,
        source,
        *,
        state,
        on_input=None,
        on_sample=None,
        max_frames=0,
        metrics=None,
        save_settings=None,
        menu=None,
    ):
        self.performance = Performance()
        try:
            dispatch = XRDispatch(self, state)
            dispatch.run(
                lambda: self._run(source, state, on_input, max_frames, metrics, dispatch, menu),
                on_sample,
            )
        except xr.XrException as exc:
            raise RuntimeError(f"OpenXR session failed: {type(exc).__name__}: {exc}") from exc
        finally:
            self.performance.close(metrics)

    def _run(self, source, state, on_input, max_frames, metrics, dispatch, menu=None):
        revision = captured_at = None
        pending = None
        frames = 0
        request = None
        previous = clock.now()
        mouse_until = 0.0
        recenters = self.recenters
        while not max_frames or frames < max_frames:
            self.xr.poll()
            if self.xr.exiting:
                break
            events, value = dispatch.frame()
            if not self.xr.running:
                if value.quit:
                    break
                time.sleep(0.01)
                continue
            with self.xr.frame() as (frame, layers, space):
                now = clock.now()
                dt = now - previous
                previous = now
                at = frame.predicted_display_time
                commands = value.commands
                if on_input:
                    on_input(value)
                if commands:
                    request = f"{commands[-1].replace('_', '-').upper()} requested"
                    log.info("Command: %s", commands[-1])
                current = self.glided(state(), now, value.head)
                applied = []
                for _ in range(PUBLICATION_PACKETS):
                    update = pending if pending is not None else source.poll()
                    if update is None:
                        break
                    if isinstance(update, SplatUpdate):
                        if not self.renderer.apply(update):
                            pending = update
                            break
                        if (update.epoch, update.revision) != (
                            self.renderer.epoch,
                            self.renderer.revision,
                        ):
                            pending = None
                            continue
                    elif update.revision != revision:
                        self.renderer.upload(update.gaussians)
                    if isinstance(update, SplatUpdate):
                        applied.append(update)
                    pending = None
                    revision, captured_at = update.revision, update.captured_at
                captures = []
                rendered = False
                eye_poses = []
                if frame.should_render:
                    view_state, views = xr.locate_views(
                        self.xr.session,
                        xr.ViewLocateInfo(
                            view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO,
                            display_time=at,
                            space=space,
                        ),
                    )
                    flags = (
                        xr.ViewStateFlags.POSITION_VALID_BIT
                        | xr.ViewStateFlags.ORIENTATION_VALID_BIT
                    )
                    if view_state.view_state_flags & flags == flags:
                        from .input import matrix

                        projection_views = []
                        if value.screenshot:
                            self.capture_number += 1
                        first, second = (v.pose.orientation.as_numpy() for v in views)
                        turn = first + (second if first @ second >= 0 else -second)
                        turn = xr.Quaternionf(*turn / np.linalg.norm(turn))
                        middle = matrix(xr.Posef(orientation=turn))
                        middle[:3, 3] = sum(v.pose.position.as_numpy() for v in views) / 2
                        if not current.flat_video:
                            self.renderer.sort(current.robot_camera, middle)
                        for index, (view, swapchain) in enumerate(
                            zip(views, self.xr.eyes, strict=True)
                        ):
                            eye = matrix(view.pose)
                            eye_poses.append(eye.tolist())
                            projection = Matrix4x4f.create_projection_fov(
                                GraphicsAPI.OPENGL, view.fov, 0.02, 1000
                            ).as_numpy()
                            with swapchain.acquire() as target:
                                GL.glEnable(GL.GL_FRAMEBUFFER_SRGB)
                                self.draw_view(
                                    current,
                                    eye,
                                    projection,
                                    target,
                                    viewport=(0, 0, *swapchain.size),
                                    sort=False,
                                    center=middle,
                                )
                                if value.screenshot:
                                    captures.append(
                                        self._capture_target(
                                            target, "left" if index == 0 else "right"
                                        )
                                    )
                            projection_views.append(
                                xr.CompositionLayerProjectionView(
                                    pose=view.pose, fov=view.fov, sub_image=swapchain.sub_image
                                )
                            )
                        layers.append(
                            xr.CompositionLayerProjection(space=space, views=projection_views)
                        )
                        rendered = True
                    pointer = dispatch.pointer
                    panel_events = []
                    for event in events:
                        if event.type == pygame.MOUSEMOTION:
                            width, height = pygame.display.get_window_size()
                            event.pos = (
                                event.pos[0] * self.xr.panel.size[0] / width,
                                event.pos[1] * self.xr.panel.size[1] / height,
                            )
                        panel_events.append(event)
                    mouse = any(
                        e.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP)
                        for e in events
                    )
                    if mouse:
                        mouse_until = now + 2
                    aimed = now >= mouse_until
                    io = self.overlay.begin(
                        panel_events, self.xr.panel.size, False, pointer=pointer if aimed else None
                    )
                    self.panel_hovered = aimed and pointer[0] >= 0 and io.want_capture_mouse
                    status = current.status
                    if self.panel_hovered or status.robot_state != "active":
                        self.panel_until = now + PANEL_LINGER
                    shown = self.panel_visible
                    self.panel_visible = (
                        self.actions.panel_pinned
                        or now < self.panel_until
                        or status.link != "CONNECTED"
                        or status.e_stop
                    )
                    if self.actions.head_flags & VALID == VALID:
                        self.panel_pose = self.panel.update(
                            value.head,
                            dt,
                            snap=not shown or recenters != self.recenters,
                            hold=self.panel_hovered,
                        )
                        recenters = self.recenters
                    if pointer[0] >= 0:
                        imgui.get_foreground_draw_list().add_circle_filled(
                            (pointer[0], pointer[1]), 4, 0xFFFFFFFF
                        )
                    with self.xr.panel.acquire() as target:
                        target.use()
                        self.context.viewport = (0, 0, *self.xr.panel.size)
                        target.clear(0, 0, 0, 0)
                        if menu:
                            menu(self.xr.panel.size)
                        else:
                            budget = self.overlay.draw(
                                current.status,
                                1 / max(dt, 1e-6),
                                self.renderer.count,
                                None if captured_at is None else max(0, now - captured_at),
                                False,
                                request,
                                live=True,
                                target=target,
                                xr_mode=True,
                                performance=self.performance.last,
                                driving=any(value.movement),
                                panel_shown=self.panel_visible,
                            )
                            if budget is not None:
                                if budget <= self.splat_limit:
                                    source.set_max_splats(budget)
                                else:
                                    self.overlay.error = (
                                        f"This GPU supports at most {self.splat_limit:,} splats"
                                    )
                        dispatch.ui(io, self.overlay.commands)
                        self.overlay.commands.clear()
                        if value.screenshot:
                            captures.append(self._capture_target(target, "panel"))
                        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, target.glo)
                        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)
                        GL.glBlitFramebuffer(
                            0,
                            0,
                            *self.xr.panel.size,
                            0,
                            0,
                            *pygame.display.get_window_size(),
                            GL.GL_COLOR_BUFFER_BIT,
                            GL.GL_LINEAR,
                        )
                        pygame.display.flip()
                    if (
                        menu
                        or self.panel_visible
                        or self.overlay.theme.hud.alpha(self.overlay.theme.t) > 0
                    ):
                        layers.append(
                            xr.CompositionLayerQuad(
                                layer_flags=xr.CompositionLayerFlags.BLEND_TEXTURE_SOURCE_ALPHA_BIT,
                                space=space,
                                eye_visibility=xr.EyeVisibility.BOTH,
                                sub_image=self.xr.panel.sub_image,
                                pose=xr.Posef(
                                    orientation=xr.Quaternionf(*quaternion(self.panel_pose)),
                                    position=xr.Vector3f(*map(float, self.panel_pose[:3, 3])),
                                ),
                                size=xr.Extent2Df(*self.panel_size),
                            )
                        )
                performance = self.performance.last
                if rendered:
                    self.context.finish()
                    period = frame.predicted_display_period / 1e9
                    if hasattr(source, "display_completed"):
                        source.display_completed(now, period)
                    statistics = self.performance.worker_statistics | (
                        source.statistics() if hasattr(source, "statistics") else {}
                    )
                    statistics.update(
                        camera_hz=current.status.rates[1],
                        pilot_input_hz=current.status.rates[0],
                        flat_video=current.flat_video,
                        reconstruction_status=current.status.reconstruction,
                        link=current.status.link,
                        xr_frame_interval_ms=dt * 1000,
                        display_period_ms=period * 1000,
                        vram=self.gpu.used() if self.gpu is not None else None,
                    )
                    performance = self.performance.completed(
                        now,
                        now,
                        period,
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
                    frame_ms=dt * 1000,
                    revision=revision,
                    capture_time=captured_at,
                    predicted_display_time=at,
                    should_render=bool(frame.should_render),
                    rendered=rendered,
                    performance=performance,
                )
                if metrics:
                    metrics.write(
                        json.dumps(
                            {
                                "frame": frames,
                                "time": now,
                                "frame_ms": dt * 1000,
                                "predicted_display_time": at,
                                "should_render": bool(frame.should_render),
                                "rendered": rendered,
                                "performance": performance,
                                "capture_to_splat_visible_ms": self.performance.latest_latency,
                                "head": value.head.tolist(),
                                "head_flags": self.actions.head_flags,
                                "eyes": eye_poses,
                                "hands": {k: p.model_dump() for k, p in value.hands.items()},
                                "trackers": {k: p.model_dump() for k, p in value.trackers.items()},
                                "axes": value.axes,
                                "commands": commands,
                                "active": value.active,
                                "gaussians": self.renderer.count,
                                "sorts": self.renderer.sorts,
                                "panel_visible": self.panel_visible,
                                "revision": revision,
                                "link": current.status.link,
                                "robot_state": current.status.robot_state,
                                "e_stop": current.status.e_stop,
                                "session": self.xr.state.name,
                                "recenters": self.recenters,
                                "haptic_pulses": self.actions.haptic_pulses,
                                "captures": captures,
                            }
                        )
                        + "\n"
                    )
                    metrics.flush()
                if value.quit or self.overlay.leave:
                    break

    def close(self):
        try:
            if self.xr is not None:
                GL.glFinish()
                self.xr.close()
        finally:
            self.xr = None
            super().close()
