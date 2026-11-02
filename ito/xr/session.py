import ctypes as ct
import logging
import math
import os
import sys
import threading
import time
from contextlib import ExitStack, contextmanager

import vulkan as vk
import xr

from ito import clock
from ito.app import settings
from ito.render.vk import API_VERSION, Image, address

from . import runtimes

log = logging.getLogger(__name__)
HEADSET_WAIT = 90.0
HEADSET_POLL = 0.5
HUD_PANEL_PIXELS = (1024, 640)
ROUND_PAUSE = 2.0


# Swapchain formats, best first: sRGB ones (the runtime shows what is stored, the camera's own
# sRGB values; vk.Presenter decodes into them), then UNORM (taken as linear light: the decoded
# values are stored).
FORMATS = (
    vk.VK_FORMAT_R8G8B8A8_SRGB,
    vk.VK_FORMAT_B8G8R8A8_SRGB,
    vk.VK_FORMAT_R8G8B8A8_UNORM,
    vk.VK_FORMAT_B8G8R8A8_UNORM,
)
EXTENSION = "XR_KHR_vulkan_enable2"


def _address(handle):
    return ct.cast(handle, ct.c_void_p).value


def _pointer(structure, field, at):
    """A ctypes pointer of the type pyopenxr declares for structure.field, at address `at`."""
    return ct.cast(ct.c_void_p(at), dict(structure._fields_)[field])


def _vulkan_loader():
    name = "vulkan-1" if sys.platform == "win32" else "libvulkan.so.1"
    library = ct.WinDLL(name) if sys.platform == "win32" else ct.CDLL(name)
    kind = dict(xr.VulkanInstanceCreateInfoKHR._fields_)["pfn_get_instance_proc_addr"]
    return ct.cast(library.vkGetInstanceProcAddr, kind)


class Swapchain:
    def __init__(self, owner, size):
        self.size = size
        self.handle = xr.create_swapchain(
            owner.session,
            xr.SwapchainCreateInfo(
                usage_flags=xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT
                | xr.SwapchainUsageFlags.SAMPLED_BIT
                | xr.SwapchainUsageFlags.TRANSFER_SRC_BIT,
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
        self.images = [
            Image(owner.device, size, owner.format, handle=_address(image.image), transfer=False)
            for image in xr.enumerate_swapchain_images(self.handle, xr.SwapchainImageVulkan2KHR)
        ]
        for image in self.images:
            owner.resources.callback(image.close)
        self.sub_image = xr.SwapchainSubImage(
            swapchain=self.handle,
            image_rect=xr.Rect2Di(extent=xr.Extent2Di(*size)),
        )

    def acquire(self):
        """The image to draw into; release() it once the frame drawing it is submitted, in
        the colour attachment layout (OpenXR's hand-over)."""
        index = xr.acquire_swapchain_image(self.handle)
        xr.wait_swapchain_image(
            self.handle, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION)
        )
        image = self.images[index]
        image.layout = vk.VK_IMAGE_LAYOUT_UNDEFINED
        return image

    def hand_over(self, frame, image):
        frame.transition(image, vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)

    def release(self):
        xr.release_swapchain_image(self.handle)


INFO = "vulkan_create_info"


class VulkanBinding:
    """For vk.Device: the instance and device made by the runtime (XR_KHR_vulkan_enable2)."""

    def __init__(self, instance, system):
        self.instance, self.system = instance, system
        self.loader = _vulkan_loader()

    def create_instance(self, info):
        handle, result = xr.create_vulkan_instance_khr(
            self.instance,
            xr.VulkanInstanceCreateInfoKHR(
                system_id=self.system,
                pfn_get_instance_proc_addr=self.loader,
                vulkan_create_info=_pointer(xr.VulkanInstanceCreateInfoKHR, INFO, info),
            ),
        )
        if result.value:
            raise RuntimeError(
                f"The OpenXR runtime could not create a Vulkan instance: {result.value}"
            )
        return _address(handle)

    def physical_device(self, instance):
        return _address(
            xr.get_vulkan_graphics_device2_khr(
                self.instance,
                xr.VulkanGraphicsDeviceGetInfoKHR(
                    system_id=self.system,
                    vulkan_instance=_pointer(
                        xr.VulkanGraphicsDeviceGetInfoKHR, "vulkan_instance", instance
                    ),
                ),
            )
        )

    def create_device(self, physical, info):
        handle, result = xr.create_vulkan_device_khr(
            self.instance,
            xr.VulkanDeviceCreateInfoKHR(
                system_id=self.system,
                pfn_get_instance_proc_addr=self.loader,
                vulkan_physical_device=_pointer(
                    xr.VulkanDeviceCreateInfoKHR, "vulkan_physical_device", physical
                ),
                vulkan_create_info=_pointer(xr.VulkanDeviceCreateInfoKHR, INFO, info),
            ),
        )
        if result.value:
            raise RuntimeError(
                f"The OpenXR runtime could not create a Vulkan device: {result.value}"
            )
        return _address(handle)


def runtime_name(instance):
    properties = xr.get_instance_properties(instance)
    name = properties.runtime_name
    name = name.decode(errors="replace") if isinstance(name, bytes) else str(name)
    version = properties.runtime_version
    return f"{name} {version.major}.{version.minor}.{version.patch}"


class Session:
    """An OpenXR session, in two steps: Session() finds a headset (waiting() shows that it is
    waiting), binding() gives vk.Device what it needs to create the Vulkan device the runtime
    wants, and start(device) begins the session with it."""

    def __init__(self, reference, waiting=None):
        self.reference_mode = reference
        self.resources = ExitStack()
        self.running = self.exiting = False
        self.state = xr.SessionState.IDLE
        self.space_changed = False
        self.space_lock = threading.Lock()
        self.frame_space = None
        self.retired_space = None
        self.clock = None
        self.device = None
        try:
            self.instance, self.system = self._headset(waiting)
            self.resources.callback(xr.destroy_instance, self.instance)
            requirements = xr.GraphicsRequirementsVulkan2KHR()
            get = ct.cast(
                xr.get_instance_proc_addr(self.instance, "xrGetVulkanGraphicsRequirements2KHR"),
                xr.PFN_xrGetVulkanGraphicsRequirements2KHR,
            )
            result = xr.check_result(get(self.instance, self.system, ct.byref(requirements)))
            if result.is_exception():
                raise result
            version = xr.Version(
                vk.VK_VERSION_MAJOR(API_VERSION), vk.VK_VERSION_MINOR(API_VERSION), 0
            )
            if requirements.min_api_version_supported > version:
                raise RuntimeError("The OpenXR runtime needs a newer Vulkan than Ito's 1.3")
        except BaseException:
            self.close()
            raise

    def binding(self):
        return VulkanBinding(self.instance, self.system)

    def start(self, device):
        self.device = device
        try:
            binding = xr.GraphicsBindingVulkan2KHR(
                instance=_pointer(
                    xr.GraphicsBindingVulkan2KHR, "instance", address(device.instance)
                ),
                physical_device=_pointer(
                    xr.GraphicsBindingVulkan2KHR, "physical_device", address(device.physical)
                ),
                device=_pointer(xr.GraphicsBindingVulkan2KHR, "device", address(device.device)),
                queue_family_index=device.family,
                queue_index=0,
            )
            self.session = xr.create_session(
                self.instance,
                xr.SessionCreateInfo(next=binding, system_id=self.system),
            )
            self.resources.callback(xr.destroy_session, self.session)
            log.info("OpenXR session created")
            reference = self.reference_mode
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
            self.format = next((f for f in FORMATS if f in formats), None)
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
            self.panel = Swapchain(self, HUD_PANEL_PIXELS)
        except BaseException:
            self.close()
            raise

    def _headset(self, waiting):
        candidates = runtimes.installed() or [runtimes.Runtime("OpenXR", None)]
        names = " or ".join(c.name for c in candidates)
        started = clock.now()
        logged = -math.inf
        problems = {}
        while True:
            for candidate in candidates:
                found = self._ask(candidate, problems)
                if found:
                    if candidate.manifest and not runtimes.PINNED:
                        settings.remember_xr_runtime(candidate.manifest)
                    return found
            if len(problems) == len(candidates):
                raise RuntimeError(
                    "No OpenXR runtime could start: "
                    + "; ".join(f"{name}: {why}" for name, why in problems.items())
                )
            pause_until = clock.now() + ROUND_PAUSE
            while True:
                waited = clock.now() - started
                if waited >= HEADSET_WAIT:
                    raise RuntimeError(
                        f"No headset found on {names} in {HEADSET_WAIT:.0f} s. Connect the "
                        "headset first (Steam Link, Virtual Desktop or its cable), make sure it "
                        "is awake, and try again."
                    )
                if waited - logged >= 10:
                    log.info("Waiting for a headset on %s (%.0f s)", names, waited)
                    logged = waited
                if waiting is None:
                    time.sleep(HEADSET_POLL)
                elif not waiting(names, waited):
                    raise RuntimeError("Stopped waiting for a headset")
                if clock.now() >= pause_until:
                    break

    def _ask(self, candidate, problems):
        if candidate.manifest:
            os.environ["XR_RUNTIME_JSON"] = candidate.manifest
        instance = None
        try:
            available = {
                p.extension_name.decode() for p in xr.enumerate_instance_extension_properties()
            }
            extensions = [EXTENSION]
            if "XR_HTCX_vive_tracker_interaction" in available:
                extensions.append("XR_HTCX_vive_tracker_interaction")
            vulkan = EXTENSION in available
            instance = xr.create_instance(
                xr.InstanceCreateInfo(
                    application_info=xr.ApplicationInfo(application_name="Ito"),
                    enabled_extension_names=extensions if vulkan else [],
                )
            )
            name = runtime_name(instance)
            if not vulkan:
                problems[candidate.name] = "no Vulkan support"
                xr.destroy_instance(instance)
                return None
            system = xr.get_system(
                instance, xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY)
            )
        except xr.FormFactorUnavailableError:
            problems.pop(candidate.name, None)
            xr.destroy_instance(instance)
            return None
        except xr.XrException as exc:
            problems[candidate.name] = f"{type(exc).__name__}: {exc}"
            if instance is not None:
                xr.destroy_instance(instance)
            return None
        problems.pop(candidate.name, None)
        self.extensions, self.runtime = extensions, name
        log.info("OpenXR runtime with a headset: %s (%s)", name, candidate.manifest or "system")
        return instance, system

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

    def recenter(self, at, heading=0.0):
        """Re-anchor the view so the pilot's gaze now is the robot's view, heading (yaw).

        The pilot's head is the robot's look target in its world, so centring the gaze on
        the robot's own heading leaves the robot looking where it already looks.
        """
        from .input import VALID, matrix

        location = xr.locate_space(self.head, self.base, at)
        if location.location_flags & VALID != VALID:
            return False
        head = matrix(location.pose)
        yaw = math.atan2(float(head[0, 2]), float(head[2, 2])) - heading
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
