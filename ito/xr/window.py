"""The headset clock drives both eyes, even while the link or scene producer stalls."""

import json
import logging
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
from ito.render import pose

from .dispatch import XRDispatch
from .input import Actions
from .session import Session

log = logging.getLogger(__name__)
PANEL_LINGER = 0.5  # The panel stays this long after the controller stops pointing at it.


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
        panel = pose((0, -0.15, -1.2))
        hand = self.aiming_hand()
        if hand is None:
            return (-10000, -10000, False)
        aim = self.actions.aims[hand]
        origin, direction = aim[:3, 3] - panel[:3, 3], -aim[:3, 2]
        if direction[2] >= -1e-5:
            return (-10000, -10000, False)
        distance = -origin[2] / direction[2]
        hit = origin + distance * direction
        x, y = (hit[0] / 1.05 + 0.5) * 768, (0.5 - hit[1] / (1.05 * 440 / 768)) * 440
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
        try:
            dispatch = XRDispatch(self, state)
            dispatch.run(
                lambda: self._run(source, state, on_input, max_frames, metrics, dispatch),
                on_sample,
            )
        except xr.XrException as exc:
            raise RuntimeError(f"OpenXR session failed: {type(exc).__name__}: {exc}") from exc

    def _run(self, source, state, on_input, max_frames, metrics, dispatch):
        revision = captured_at = None
        frames = 0
        request = None
        previous = clock.now()
        mouse_until = 0.0
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
                current = self.glided(state(), now)
                for _ in range(4):
                    update = source.poll()
                    if update is None:
                        break
                    if isinstance(update, SplatUpdate):
                        self.renderer.apply(update)
                    elif update.revision != revision:
                        self.renderer.upload(update.gaussians)
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
                    self.panel_visible = (
                        self.actions.panel_pinned
                        or now < self.panel_until
                        or status.link != "CONNECTED"
                        or status.e_stop
                    )
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
                                pose=xr.Posef(position=xr.Vector3f(0, -0.15, -1.2)),
                                size=xr.Extent2Df(1.05, 1.05 * 440 / 768),
                            )
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
