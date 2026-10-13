"""The headset clock drives both eyes, even while the link or scene producer stalls."""

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
from ito.desktop import DesktopWindow
from ito.reconstruction import SplatUpdate
from ito.reconstruction.admission import PUBLICATION_PACKETS
from ito.render import pose
from ito.render.performance import Performance
from ito.render.pose import quaternion

from .dispatch import XRDispatch
from .input import VALID, Actions
from .session import Session

log = logging.getLogger(__name__)
PANEL_LINGER = 0.5  # The panel stays this long after the controller stops pointing at it.
# The panel hangs this far in front of the pilot's eyes, this far below their gaze, facing
# them, so it is centred and readable with a glance down but never covers the scene ahead.
PANEL_DISTANCE = 1.2
PANEL_DROP = math.radians(25)
PANEL_SIZE = (1.05, 1.05 * 440 / 768)
# It follows the head lazily: it starts to follow once the gaze leaves it by SLACK, glides
# with time constant FOLLOW until it is centred again (within SETTLED), and holds still while
# a controller points at a control on it. A glance down at it, a pitch of up to the drop,
# stays within the pitch slack, so the panel never slides away from a pilot reading it.
PANEL_FOLLOW = 0.25
PANEL_SLACK = (math.radians(20), math.radians(32))  # yaw, pitch
PANEL_SETTLED = math.radians(2)
PANEL_PITCH = (math.radians(-45), math.radians(40))  # Where the anchor may point.


class PanelFollow:
    """Where the VR panel hangs: an anchor (eye position, yaw, pitch) eased after the head."""

    def __init__(self):
        self.position = None
        self.angles = [0.0, 0.0]
        self.following = [False, False]

    @staticmethod
    def gaze(head):
        """(yaw, pitch) of the head's forward axis; yaw is None looking straight up or down."""
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
            self.xr = Session(self.context, reference_space)
            self.actions = Actions(self.xr)
            self.input.translate = False
            self.centered = False
            self.recenters = 0
            self.panel_until = 0.0
            self.panel_visible = True
            self.panel_hovered = False  # ImGui hovers a control under the aim (last frame).
            self.panel = PanelFollow()
            self.panel_pose = self.panel.matrix()  # As last submitted; the pointer aims at it.
            pygame.display.set_caption("Ito — XR pilot controls")
        except xr.XrException as exc:
            self.close()
            raise RuntimeError(
                f"OpenXR initialization failed: {type(exc).__name__}: {exc}"
            ) from exc
        except BaseException:
            self.close()
            raise

    def aiming_hand(self):
        """Right aim takes precedence; without a tracked controller nothing points."""
        return next((hand for hand in ("right", "left") if hand in self.actions.aims), None)

    def _pointer(self):
        hand = self.aiming_hand()
        if hand is None:
            return (-10000, -10000, False)
        # The aim ray in the panel's own frame, where the quad spans x and y and faces +z.
        aim = np.linalg.inv(self.panel_pose) @ self.actions.aims[hand]
        origin, direction = aim[:3, 3], -aim[:3, 2]
        if direction[2] >= -1e-5:
            return (-10000, -10000, False)
        distance = -origin[2] / direction[2]
        hit = origin + distance * direction
        x, y = (hit[0] / PANEL_SIZE[0] + 0.5) * 768, (0.5 - hit[1] / PANEL_SIZE[1]) * 440
        inside = distance > 0 and 0 <= x < 768 and 0 <= y < 440
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
    ):
        self.performance = Performance()
        try:
            dispatch = XRDispatch(self, state)
            dispatch.run(
                lambda: self._run(source, state, on_input, max_frames, metrics, dispatch),
                on_sample,
            )
        except xr.XrException as exc:
            raise RuntimeError(f"OpenXR session failed: {type(exc).__name__}: {exc}") from exc
        finally:
            self.performance.close(metrics)

    def _run(self, source, state, on_input, max_frames, metrics, dispatch):
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
                        # One back-to-front order from between the eyes serves both, and
                        # the flat feed's panel hangs in front of that point. Canted
                        # displays turn each eye outward: face halfway between them.
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
                    # Panel remains available when positional tracking temporarily disappears.
                    pointer = dispatch.pointer
                    panel_events = []
                    for event in events:
                        if event.type == pygame.MOUSEMOTION:
                            width, height = pygame.display.get_window_size()
                            event.pos = (event.pos[0] * 768 / width, event.pos[1] * 440 / height)
                        panel_events.append(event)
                    # Mouse events in the companion window override controller aim for this frame.
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
                    # ImGui hovers from the previous frame's controls, not the whole quad, so
                    # aiming roughly ahead leaves the trigger to the robot.
                    self.panel_hovered = aimed and pointer[0] >= 0 and io.want_capture_mouse
                    # The panel covers part of the view: show it while a controller points at
                    # a control, while pinned, and whenever it says why the robot won't move.
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
                    # Hidden, the panel stays where it would appear: low ahead of the gaze.
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
                            768,
                            440,
                            0,
                            0,
                            *pygame.display.get_window_size(),
                            GL.GL_COLOR_BUFFER_BIT,
                            GL.GL_LINEAR,
                        )
                        pygame.display.flip()
                    if self.panel_visible:
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
                                size=xr.Extent2Df(*PANEL_SIZE),
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
