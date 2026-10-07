"""OpenXR lifetime and swapchain ownership; SDL owns the current OpenGL context."""

import ctypes as ct
import logging
import math
import threading
from contextlib import ExitStack, contextmanager

import moderngl
import xr
from OpenGL import GL
from xr.utils.gl import OpenGLGraphics

from ito import clock

log = logging.getLogger(__name__)


class SDLContext:
    def make_current(self):
        pass  # DisplayDispatch binds the SDL context on the rendering thread.


class Swapchain:
    def __init__(self, owner, size):
        self.size = size
        self.handle = xr.create_swapchain(
            owner.session,
            xr.SwapchainCreateInfo(
                usage_flags=xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT
                | xr.SwapchainUsageFlags.SAMPLED_BIT,
                format=owner.format,
                sample_count=1,
                width=size[0],
                height=size[1],
                face_count=1,
                array_size=1,
                mip_count=1,
            ),
        )
        owner.resources.callback(xr.destroy_swapchain, self.handle)
        self.images = xr.enumerate_swapchain_images(self.handle, xr.SwapchainImageOpenGLKHR)
        self.targets = []
        for image in self.images:
            # External wrappers never delete the runtime-owned color images.
            texture = owner.context.external_texture(image.image, size, 4, 0, "f1")
            owner.resources.callback(texture.release)
            target = owner.context.framebuffer(color_attachments=[texture])
            self.targets.append(target)
            owner.resources.callback(target.release)
        self.sub_image = xr.SwapchainSubImage(
            swapchain=self.handle,
            image_rect=xr.Rect2Di(extent=xr.Extent2Di(*size)),
        )

    @contextmanager
    def acquire(self):
        index = xr.acquire_swapchain_image(self.handle)
        xr.wait_swapchain_image(
            self.handle, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION)
        )
        try:
            yield self.targets[index]
        finally:
            GL.glFlush()
            xr.release_swapchain_image(self.handle)


class Session:
    def __init__(self, context: moderngl.Context, reference):
        self.context = context
        self.reference_mode = reference
        self.resources = ExitStack()
        self.running = self.exiting = False
        self.state = xr.SessionState.IDLE
        self.space_changed = False
        self.space_lock = threading.Lock()
        self.frame_space = None
        self.retired_space = None
        self.clock = None
        try:
            try:
                available = {
                    p.extension_name.decode() for p in xr.enumerate_instance_extension_properties()
                }
            except xr.XrException as exc:
                raise RuntimeError(
                    "No OpenXR runtime is available. Start SteamVR/Virtual Desktop or Monado, "
                    "select it as the active OpenXR runtime (XR_RUNTIME_JSON on Linux), and retry."
                ) from exc
            if "XR_KHR_opengl_enable" not in available:
                raise RuntimeError("The active OpenXR runtime does not support OpenGL")
            self.extensions = ["XR_KHR_opengl_enable"]
            if "XR_HTCX_vive_tracker_interaction" in available:
                self.extensions.append("XR_HTCX_vive_tracker_interaction")
            self.instance = xr.create_instance(
                xr.InstanceCreateInfo(
                    application_info=xr.ApplicationInfo(application_name="Ito"),
                    enabled_extension_names=self.extensions,
                )
            )
            self.resources.callback(xr.destroy_instance, self.instance)
            self.system = xr.get_system(self.instance, xr.SystemGetInfo())
            self.graphics = OpenGLGraphics(self.instance, self.system, SDLContext())
            requirements = self.graphics.graphics_requirements
            version = xr.Version(context.version_code // 100, context.version_code % 100 // 10, 0)
            if (
                not requirements.min_api_version_supported
                <= version
                <= requirements.max_api_version_supported
            ):
                raise RuntimeError("Current OpenGL version is outside the OpenXR runtime's range")
            self.session = xr.create_session(
                self.instance,
                xr.SessionCreateInfo(
                    next=self.graphics.graphics_binding.pointer,
                    system_id=self.system,
                ),
            )
            self.resources.callback(xr.destroy_session, self.session)
            log.info("OpenXR session created")
            reference_type = (
                xr.ReferenceSpaceType.STAGE
                if reference == "standing"
                else xr.ReferenceSpaceType.LOCAL
            )
            supported = xr.enumerate_reference_spaces(self.session)
            if reference_type not in supported:
                raise RuntimeError(
                    "Standing space is unavailable; configure room setup "
                    "or use --reference-space seated"
                )
            self.space = self.reference(reference_type)
            self.base = self.space
            self.resources.callback(self._close_centered)
            self.head = self.reference(xr.ReferenceSpaceType.VIEW)
            modes = xr.enumerate_environment_blend_modes(
                self.instance, self.system, xr.ViewConfigurationType.PRIMARY_STEREO
            )
            self.blend = (
                xr.EnvironmentBlendMode.OPAQUE
                if xr.EnvironmentBlendMode.OPAQUE in modes
                else modes[0]
            )
            formats = xr.enumerate_swapchain_formats(self.session)
            self.format = next(
                (f for f in (GL.GL_SRGB8_ALPHA8, GL.GL_RGBA8, GL.GL_RGBA16F) if f in formats), None
            )
            if self.format is None:
                raise RuntimeError("OpenXR runtime has no supported RGBA swapchain format")
            views = xr.enumerate_view_configuration_views(
                self.instance, self.system, xr.ViewConfigurationType.PRIMARY_STEREO
            )
            if len(views) != 2:
                raise RuntimeError("OpenXR runtime must provide a stereo view configuration")
            self.eyes = [
                Swapchain(self, (v.recommended_image_rect_width, v.recommended_image_rect_height))
                for v in views
            ]
            self.panel = Swapchain(self, (768, 440))
        except BaseException:
            self.close()
            raise

    def reference(self, kind):
        space = xr.create_reference_space(
            self.session,
            xr.ReferenceSpaceCreateInfo(
                reference_space_type=kind, pose_in_reference_space=xr.Posef()
            ),
        )
        self.resources.callback(xr.destroy_space, space)
        return space

    def _close_centered(self):
        if self.space != self.base:
            if self.space == self.frame_space:
                self.retired_space = self.space
            else:
                xr.destroy_space(self.space)
            self.space = self.base

    def recenter(self, at):
        from .input import VALID, matrix

        location = xr.locate_space(self.head, self.base, at)
        if location.location_flags & VALID != VALID:
            return False
        head = matrix(location.pose)
        yaw = math.atan2(float(head[0, 2]), float(head[2, 2]))
        centered = xr.create_reference_space(
            self.session,
            xr.ReferenceSpaceCreateInfo(
                reference_space_type=xr.ReferenceSpaceType.STAGE
                if self.reference_mode == "standing"
                else xr.ReferenceSpaceType.LOCAL,
                pose_in_reference_space=xr.Posef(
                    orientation=xr.Quaternionf(0, math.sin(yaw / 2), 0, math.cos(yaw / 2)),
                    position=location.pose.position,
                ),
            ),
        )
        # A submitted frame retains its reference space until xrEndFrame returns.
        with self.space_lock:
            self._close_centered()
            self.space = centered
        log.info("OpenXR recentered (%s)", self.reference_mode)
        return True

    def poll(self):
        while True:
            try:
                event = xr.poll_event(self.instance)
            except xr.EventUnavailable:
                return
            if event.type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                change = ct.cast(
                    ct.byref(event), ct.POINTER(xr.EventDataSessionStateChanged)
                ).contents
                self.state = xr.SessionState(change.state)
                log.info("OpenXR session: %s", self.state.name)
                if self.state == xr.SessionState.READY:
                    xr.begin_session(
                        self.session,
                        xr.SessionBeginInfo(
                            primary_view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO
                        ),
                    )
                    self.running = True
                elif self.state == xr.SessionState.STOPPING:
                    self.running = False
                    xr.end_session(self.session)
                elif self.state in (xr.SessionState.EXITING, xr.SessionState.LOSS_PENDING):
                    self.exiting = True
            elif event.type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                self.exiting = True
            elif event.type == xr.StructureType.EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
                change = ct.cast(
                    ct.byref(event), ct.POINTER(xr.EventDataReferenceSpaceChangePending)
                ).contents
                self.space_changed = change.change_time

    @contextmanager
    def frame(self):
        frame = xr.wait_frame(self.session)
        self.clock = (frame.predicted_display_time, clock.now_ns())
        xr.begin_frame(self.session)
        with self.space_lock:
            self.frame_space = space = self.space
        layers = []
        try:
            yield frame, layers, space
        finally:
            try:
                xr.end_frame(
                    self.session,
                    xr.FrameEndInfo(
                        display_time=frame.predicted_display_time,
                        environment_blend_mode=self.blend,
                        layers=[ct.byref(layer) for layer in layers],
                    ),
                )
            finally:
                with self.space_lock:
                    self.frame_space = None
                    if self.retired_space is not None:
                        xr.destroy_space(self.retired_space)
                        self.retired_space = None

    def close(self):
        self.resources.close()
