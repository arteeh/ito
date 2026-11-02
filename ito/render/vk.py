"""Ito's Vulkan: one instance and device with its graphics queue, the frames recorded on it, and
the buffers, images and pipelines the renderers draw with.

Every frame is one command buffer (Frame), FRAMES of them in flight: the CPU records a frame
while the GPU draws the one before. Passes are few and small, so between them a frame takes a
full memory barrier (Frame.barrier) rather than tracking each resource's hazards; images change
layout through Frame.transition. Descriptors are pushed with every draw (VK_KHR_push_descriptor),
uniforms written to the frame's own ring (Frame.uniform) and uploads staged through it
(Frame.stage), so nothing the GPU may still read is written by the CPU.

Shaders are SPIR-V compiled from ito/shaders (tools/shaders.py); their constants are
specialization constants (common.glsl)."""

import ctypes
import logging
import struct
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import vulkan as vk

log = logging.getLogger(__name__)
ffi, lib = vk.ffi, vk.lib

SHADERS = Path(__file__).resolve().parents[1] / "shaders"
FRAMES = 2
API_VERSION = vk.VK_MAKE_VERSION(1, 3, 0)
DEVICE_EXTENSIONS = ("VK_KHR_swapchain", "VK_KHR_push_descriptor")
# Enabled where the GPU has them: decoded camera frames shared from Direct3D 12 (d3d12.py).
INTEROP_EXTENSIONS = (
    ("VK_KHR_external_memory_win32", "VK_KHR_external_semaphore_win32")
    if sys.platform == "win32"
    else ()
)
RING = 1 << 20  # each frame's first uniform and staging ring; grows when a frame needs more
ALL = vk.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
MEMORY = vk.VK_ACCESS_2_MEMORY_READ_BIT | vk.VK_ACCESS_2_MEMORY_WRITE_BIT
COLOUR, DEPTH = vk.VK_IMAGE_ASPECT_COLOR_BIT, vk.VK_IMAGE_ASPECT_DEPTH_BIT
DEPTH_FORMAT = vk.VK_FORMAT_D32_SFLOAT
INFINITE = 2**64 - 1


def surface_extensions():
    platform = "VK_KHR_win32_surface" if sys.platform == "win32" else "VK_KHR_xlib_surface"
    return ("VK_KHR_surface", platform)


def mat(m):
    """A 4x4 matrix as GLSL's mat4 holds it (columns first), as 16 floats."""
    return np.asarray(m, np.float32).T.ravel().tolist()


def pack(layout, *values):
    """A uniform block's bytes: `layout` is struct's format (little-endian, no implicit
    padding), values flattened."""
    flat = []
    for value in values:
        if isinstance(value, (list, tuple, np.ndarray)):
            flat.extend(np.asarray(value).ravel().tolist())
        else:
            flat.append(value)
    return struct.pack("<" + layout, *flat)


def _handle(pointer, kind):
    """A Vulkan handle another library (pyopenxr's ctypes) gave as an address, for `vulkan`."""
    return ffi.cast(kind, pointer)


def address(handle) -> int:
    return int(ffi.cast("uintptr_t", handle))


def _properties(physical):
    """The GPU's properties (a structure that holds its own memory)."""
    holder = ffi.new("VkPhysicalDeviceProperties*")
    lib.vkGetPhysicalDeviceProperties(physical, holder)
    return holder


class VulkanError(RuntimeError):
    pass


def check(result, what):
    if result != vk.VK_SUCCESS:
        raise VulkanError(f"{what} failed: VkResult {result}")


class Device:
    """The instance, the GPU and its device, the window's surface, and the frames.

    `window` is the native window to present to ((hinstance, hwnd) on Windows, (display,
    window) on X11). By default Ito creates the instance and device itself; an OpenXR session
    passes `xr` (create_instance(info), physical_device(instance), create_device(physical,
    info)) so the runtime does, on the GPU that drives the headset (XR_KHR_vulkan_enable2)."""

    def __init__(self, window, *, xr=None):
        self.instance = self.device = self.surface = self.dummy_array = None
        self.frames = []
        self.frame = None
        self.index = 0
        self.samplers = {}
        self.closing = []
        extensions = [e.encode() for e in surface_extensions()]
        # cffi frees what ffi.new() made once nothing refers to it, also while a structure
        # still points at it: everything a structure points at is held in a variable.
        self._names = [ffi.new("char[]", e) for e in extensions]
        name = ffi.new("char[]", b"Ito")
        app = ffi.new(
            "VkApplicationInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                "pApplicationName": name,
                "pEngineName": name,
                "apiVersion": API_VERSION,
            },
        )
        names = ffi.new("char*[]", self._names)
        info = ffi.new(
            "VkInstanceCreateInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                "pApplicationInfo": app,
                "enabledExtensionCount": len(extensions),
                "ppEnabledExtensionNames": names,
            },
        )
        try:
            if xr is None:
                handle = ffi.new("VkInstance*")
                check(lib.vkCreateInstance(info, ffi.NULL, handle), "vkCreateInstance")
                self.instance = handle[0]
            else:
                self.instance = _handle(xr.create_instance(address(info)), "VkInstance")
            self._surface(window)
            if xr is None:
                self.physical = self._choose()
            else:
                self.physical = _handle(
                    xr.physical_device(address(self.instance)), "VkPhysicalDevice"
                )
            self.family = self._family(self.physical)
            if self.family is None:
                raise VulkanError("The headset's GPU cannot draw to Ito's window")
            self.properties = _properties(self.physical)
            self._device(xr)
            self._memory = ffi.new("VkPhysicalDeviceMemoryProperties*")
            lib.vkGetPhysicalDeviceMemoryProperties(self.physical, self._memory)
            self.memory = self._memory[0]
            self.queue = vk.vkGetDeviceQueue(self.device, self.family, 0)
            self.push_descriptors = self.device_function("vkCmdPushDescriptorSetKHR")
            self.alignment = max(
                256,
                self.properties.limits.minUniformBufferOffsetAlignment,
                self.properties.limits.minStorageBufferOffsetAlignment,
            )
            pool = ffi.new("VkCommandPool*")
            check(
                lib.vkCreateCommandPool(
                    self.device,
                    ffi.new(
                        "VkCommandPoolCreateInfo*",
                        {
                            "sType": vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                            "flags": vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                            "queueFamilyIndex": self.family,
                        },
                    ),
                    ffi.NULL,
                    pool,
                ),
                "vkCreateCommandPool",
            )
            self.pool = pool[0]
            self.frames = [Frame(self) for _ in range(FRAMES)]
            # Bound where an image array has nothing to hold yet (the atlas before its first layer).
            self.dummy_array = Image(
                self, (1, 1), vk.VK_FORMAT_R8G8B8A8_UNORM, sampled=True, layers=1, array=True
            )
            with self.once() as frame:
                frame.clear(self.dummy_array, (0, 0, 0, 0))
                frame.transition(self.dummy_array, vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        except BaseException:
            self.close()
            raise
        self.name = name = ffi.string(self.properties.deviceName).decode()
        log.info(
            "Vulkan %d.%d.%d | %s",
            *(
                f(self.properties.apiVersion)
                for f in (vk.VK_VERSION_MAJOR, vk.VK_VERSION_MINOR, vk.VK_VERSION_PATCH)
            ),
            name,
        )

    # -- set-up --------------------------------------------------------------------------
    def instance_function(self, name):
        pointer = lib.vkGetInstanceProcAddr(self.instance, name.encode())
        if pointer == ffi.NULL:
            raise VulkanError(f"{name} is unavailable")
        return ffi.cast("PFN_" + name, pointer)

    def device_function(self, name):
        pointer = lib.vkGetDeviceProcAddr(self.device, name.encode())
        if pointer == ffi.NULL:
            raise VulkanError(f"{name} is unavailable")
        return ffi.cast("PFN_" + name, pointer)

    def _surface(self, window):
        surface = ffi.new("VkSurfaceKHR*")
        if sys.platform == "win32":
            hinstance, hwnd = window
            create = self.instance_function("vkCreateWin32SurfaceKHR")
            info = ffi.new(
                "VkWin32SurfaceCreateInfoKHR*",
                {
                    "sType": vk.VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                    "hinstance": ffi.cast("HINSTANCE", hinstance),
                    "hwnd": ffi.cast("HWND", hwnd),
                },
            )
        else:
            display, x_window = window
            create = self.instance_function("vkCreateXlibSurfaceKHR")
            info = ffi.new(
                "VkXlibSurfaceCreateInfoKHR*",
                {
                    "sType": vk.VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                    "dpy": ffi.cast("Display*", display),
                    "window": x_window,
                },
            )
        check(create(self.instance, info, ffi.NULL, surface), "Creating the window's surface")
        self.surface = surface[0]

    def _family(self, physical):
        supported = ffi.new("VkBool32*")
        get_support = self.instance_function("vkGetPhysicalDeviceSurfaceSupportKHR")
        wanted = vk.VK_QUEUE_GRAPHICS_BIT | vk.VK_QUEUE_COMPUTE_BIT
        count = ffi.new("uint32_t*")
        lib.vkGetPhysicalDeviceQueueFamilyProperties(physical, count, ffi.NULL)
        families = ffi.new("VkQueueFamilyProperties[]", count[0])
        lib.vkGetPhysicalDeviceQueueFamilyProperties(physical, count, families)
        for index, family in enumerate(families):
            if family.queueFlags & wanted != wanted:
                continue
            get_support(physical, index, self.surface, supported)
            if supported[0]:
                return index
        return None

    def _choose(self):
        """The GPU Ito draws on: a discrete one before an integrated one, before any other, that
        has Vulkan 1.3 and can draw to the window."""
        ranked = []
        for physical in vk.vkEnumeratePhysicalDevices(self.instance):
            properties = _properties(physical)
            if properties.apiVersion < API_VERSION or self._family(physical) is None:
                continue
            kind = properties.deviceType
            rank = {
                vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: 0,
                vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: 1,
            }.get(kind, 2)
            ranked.append((rank, len(ranked), physical))
        if not ranked:
            raise VulkanError("Ito needs a GPU with Vulkan 1.3 that can draw to its window")
        return min(ranked)[2]

    def _device(self, xr):
        count = ffi.new("uint32_t*")
        lib.vkEnumerateDeviceExtensionProperties(self.physical, ffi.NULL, count, ffi.NULL)
        available = ffi.new("VkExtensionProperties[]", count[0])
        lib.vkEnumerateDeviceExtensionProperties(self.physical, ffi.NULL, count, available)
        available = {ffi.string(e.extensionName).decode() for e in available}
        self.interop = bool(INTEROP_EXTENSIONS) and set(INTEROP_EXTENSIONS) <= available
        extensions = DEVICE_EXTENSIONS + (INTEROP_EXTENSIONS if self.interop else ())
        names = [ffi.new("char[]", e.encode()) for e in extensions]
        priority = ffi.new("float*", 1.0)
        queue = ffi.new(
            "VkDeviceQueueCreateInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                "queueFamilyIndex": self.family,
                "queueCount": 1,
                "pQueuePriorities": priority,
            },
        )
        features12 = ffi.new(
            "VkPhysicalDeviceVulkan12Features*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                "timelineSemaphore": 1,
                "separateDepthStencilLayouts": 1,  # DEPTH_ATTACHMENT_OPTIMAL (Frame.render)
            },
        )
        features13 = ffi.new(
            "VkPhysicalDeviceVulkan13Features*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                "pNext": features12,
                "synchronization2": 1,
                "dynamicRendering": 1,
            },
        )
        features = ffi.new(
            "VkPhysicalDeviceFeatures2*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                "pNext": features13,
                "features": {
                    "geometryShader": 1,
                    "fragmentStoresAndAtomics": 1,
                    "shaderStorageImageExtendedFormats": 1,
                },
            },
        )
        extension_names = ffi.new("char*[]", names)
        info = ffi.new(
            "VkDeviceCreateInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                "pNext": features,
                "queueCreateInfoCount": 1,
                "pQueueCreateInfos": queue,
                "enabledExtensionCount": len(names),
                "ppEnabledExtensionNames": extension_names,
            },
        )
        if xr is None:
            handle = ffi.new("VkDevice*")
            check(lib.vkCreateDevice(self.physical, info, ffi.NULL, handle), "vkCreateDevice")
            self.device = handle[0]
        else:
            self.device = _handle(
                xr.create_device(address(self.physical), address(info)), "VkDevice"
            )

    # -- memory --------------------------------------------------------------------------
    def allocate(self, requirements, *options, chain=ffi.NULL):
        """Memory for requirements, of the first kind in options (property flags) the GPU has;
        chain: the allocation's pNext (an import)."""
        found = (
            index
            for flags in options
            for index in range(self.memory.memoryTypeCount)
            if requirements.memoryTypeBits & (1 << index)
            and self.memory.memoryTypes[index].propertyFlags & flags == flags
        )
        index = next(found, None)
        if index is None:
            raise VulkanError("No GPU memory of the kind needed")
        memory = ffi.new("VkDeviceMemory*")
        check(
            lib.vkAllocateMemory(
                self.device,
                ffi.new(
                    "VkMemoryAllocateInfo*",
                    {
                        "sType": vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                        "pNext": chain,
                        "allocationSize": requirements.size,
                        "memoryTypeIndex": index,
                    },
                ),
                ffi.NULL,
                memory,
            ),
            "Allocating GPU memory",
        )
        return memory[0]

    def sampler(self, linear=True, mipmaps=False):
        key = (linear, mipmaps)
        if key not in self.samplers:
            mode = vk.VK_FILTER_LINEAR if linear else vk.VK_FILTER_NEAREST
            edge = vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
            handle = ffi.new("VkSampler*")
            check(
                lib.vkCreateSampler(
                    self.device,
                    ffi.new(
                        "VkSamplerCreateInfo*",
                        {
                            "sType": vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                            "magFilter": mode,
                            "minFilter": mode,
                            "mipmapMode": vk.VK_SAMPLER_MIPMAP_MODE_NEAREST,
                            "addressModeU": edge,
                            "addressModeV": edge,
                            "addressModeW": edge,
                            "maxLod": 1000.0 if mipmaps else 0.0,
                        },
                    ),
                    ffi.NULL,
                    handle,
                ),
                "vkCreateSampler",
            )
            self.samplers[key] = handle[0]
        return self.samplers[key]

    # -- frames --------------------------------------------------------------------------
    def begin(self):
        """Start recording the next frame (its slot's last use finished first)."""
        frame = self.frames[self.index % FRAMES]
        self.index += 1
        frame.begin()
        self.frame = frame
        return frame

    def submit(self, wait=(), signal=()):
        """Submit the frame being recorded: after the semaphores `wait` ((semaphore, the stages
        that wait) or, timeline semaphores, (semaphore, stages, value)), it signals `signal`."""
        frame, self.frame = self.frame, None
        frame.submit(wait, signal)
        return frame

    @contextmanager
    def once(self):
        """Commands recorded, submitted and waited for at once (set-up, captures)."""
        frame = Frame(self)
        frame.begin()
        try:
            yield frame
            frame.submit((), ())
            frame.wait()
        finally:
            frame.close()

    def idle(self):
        if self.device is not None:
            lib.vkDeviceWaitIdle(self.device)

    def defer(self, *resources):
        """Close these after the frames that may still use them are done."""
        self.closing.append((self.index + FRAMES, resources))

    def collect(self):
        due = [r for at, r in self.closing if at <= self.index]
        self.closing = [(at, r) for at, r in self.closing if at > self.index]
        for resources in due:
            for resource in resources:
                resource.close()

    def close(self):
        self.idle()
        for _, resources in self.closing:
            for resource in resources:
                resource.close()
        self.closing = []
        for frame in self.frames:
            frame.close()
        self.frames = []
        if self.dummy_array is not None:
            self.dummy_array.close()
            self.dummy_array = None
        if self.device is not None:
            for sampler in self.samplers.values():
                lib.vkDestroySampler(self.device, sampler, ffi.NULL)
            self.samplers = {}
            if getattr(self, "pool", None) is not None:
                lib.vkDestroyCommandPool(self.device, self.pool, ffi.NULL)
            lib.vkDestroyDevice(self.device, ffi.NULL)
            self.device = None
        if self.surface is not None:
            self.instance_function("vkDestroySurfaceKHR")(self.instance, self.surface, ffi.NULL)
            self.surface = None
        if self.instance is not None:
            lib.vkDestroyInstance(self.instance, ffi.NULL)
            self.instance = None


class Buffer:
    def __init__(self, device, size, usage, *, host=False, read=False):
        """host: mapped, for the CPU to write; read: mapped and cached, for the CPU to read (GPU
        memory the CPU does not cache reads from takes many milliseconds per megabyte)."""
        self.device, self.size = device, max(int(size), 4)
        handle = ffi.new("VkBuffer*")
        check(
            lib.vkCreateBuffer(
                device.device,
                ffi.new(
                    "VkBufferCreateInfo*",
                    {
                        "sType": vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                        "size": self.size,
                        "usage": usage,
                    },
                ),
                ffi.NULL,
                handle,
            ),
            "vkCreateBuffer",
        )
        self.buffer = handle[0]
        requirements = ffi.new("VkMemoryRequirements*")
        lib.vkGetBufferMemoryRequirements(device.device, self.buffer, requirements)
        mapped = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        if read:
            options = (mapped | vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT, mapped)
        else:
            options = (mapped,) if host else (vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,)
        self.memory = device.allocate(requirements[0], *options)
        lib.vkBindBufferMemory(device.device, self.buffer, self.memory, 0)
        self.mapped = None
        if host or read:
            pointer = ffi.new("void**")
            check(
                lib.vkMapMemory(device.device, self.memory, 0, self.size, 0, pointer),
                "vkMapMemory",
            )
            self.mapped = np.frombuffer(ffi.buffer(pointer[0], self.size), np.uint8)

    @classmethod
    def imported(cls, device, size, usage, handle):
        """A buffer in memory Direct3D 12 shares (handle: an NT handle to a D3D12 resource)."""
        kind = vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT
        self = cls.__new__(cls)
        self.device, self.size, self.mapped = device, size, None
        external = ffi.new(
            "VkExternalMemoryBufferCreateInfo*",
            {"sType": vk.VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, "handleTypes": kind},
        )
        buffer = ffi.new("VkBuffer*")
        check(
            lib.vkCreateBuffer(
                device.device,
                ffi.new(
                    "VkBufferCreateInfo*",
                    {
                        "sType": vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                        "pNext": external,
                        "size": size,
                        "usage": usage,
                    },
                ),
                ffi.NULL,
                buffer,
            ),
            "vkCreateBuffer",
        )
        self.buffer = buffer[0]
        requirements = ffi.new("VkMemoryRequirements*")
        self.memory = None
        try:
            lib.vkGetBufferMemoryRequirements(device.device, self.buffer, requirements)
            # Only the memory types the handle itself allows (the spec's
            # VUID-VkMemoryAllocateInfo-memoryTypeIndex-00645), device-local where one is.
            allowed = ffi.new(
                "VkMemoryWin32HandlePropertiesKHR*",
                {"sType": vk.VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR},
            )
            check(
                device.device_function("vkGetMemoryWin32HandlePropertiesKHR")(
                    device.device, kind, ffi.cast("HANDLE", handle), allowed
                ),
                "vkGetMemoryWin32HandlePropertiesKHR",
            )
            requirements.memoryTypeBits &= allowed.memoryTypeBits
            dedicated = ffi.new(
                "VkMemoryDedicatedAllocateInfo*",
                {
                    "sType": vk.VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                    "buffer": self.buffer,
                },
            )
            imported = ffi.new(
                "VkImportMemoryWin32HandleInfoKHR*",
                {
                    "sType": vk.VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
                    "pNext": dedicated,
                    "handleType": kind,
                    "handle": ffi.cast("HANDLE", handle),
                },
            )
            self.memory = device.allocate(
                requirements[0], vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, chain=imported
            )
            check(
                lib.vkBindBufferMemory(device.device, self.buffer, self.memory, 0),
                "vkBindBufferMemory",
            )
        except BaseException:
            lib.vkDestroyBuffer(device.device, self.buffer, ffi.NULL)
            if self.memory is not None:
                lib.vkFreeMemory(device.device, self.memory, ffi.NULL)
            self.buffer = None
            raise
        return self

    def close(self):
        if self.buffer is not None:
            if self.mapped is not None:
                self.mapped = None
                lib.vkUnmapMemory(self.device.device, self.memory)
            lib.vkDestroyBuffer(self.device.device, self.buffer, ffi.NULL)
            lib.vkFreeMemory(self.device.device, self.memory, ffi.NULL)
            self.buffer = None


class Image:
    """A 2D image (or array of them) on the GPU, with a view of it all and, mipmapped, one of
    each level. `layout` is the layout it was last left in (Frame.transition)."""

    def __init__(
        self,
        device,
        size,
        format,
        *,
        sampled=False,
        storage=False,
        attachment=False,
        transfer=True,
        layers=1,
        levels=1,
        array=False,
        handle=None,
    ):
        """handle: an image made elsewhere (only the views are this one's)."""
        self.device, self.size, self.format = device, tuple(int(v) for v in size), format
        self.layers, self.levels = layers, levels
        self.aspect = DEPTH if format == DEPTH_FORMAT else COLOUR
        self.layout = vk.VK_IMAGE_LAYOUT_UNDEFINED
        self.memory = None
        self.owned = handle is None
        if self.owned:
            usage = 0
            if sampled:
                usage |= vk.VK_IMAGE_USAGE_SAMPLED_BIT
            if storage:
                usage |= vk.VK_IMAGE_USAGE_STORAGE_BIT
            if attachment:
                usage |= (
                    vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                    if self.aspect == DEPTH
                    else vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                )
            if transfer:
                usage |= vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
            image = ffi.new("VkImage*")
            check(
                lib.vkCreateImage(
                    device.device,
                    ffi.new(
                        "VkImageCreateInfo*",
                        {
                            "sType": vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                            "imageType": vk.VK_IMAGE_TYPE_2D,
                            "format": format,
                            "extent": {"width": self.size[0], "height": self.size[1], "depth": 1},
                            "mipLevels": levels,
                            "arrayLayers": layers,
                            "samples": vk.VK_SAMPLE_COUNT_1_BIT,
                            "tiling": vk.VK_IMAGE_TILING_OPTIMAL,
                            "usage": usage,
                            "initialLayout": vk.VK_IMAGE_LAYOUT_UNDEFINED,
                        },
                    ),
                    ffi.NULL,
                    image,
                ),
                "vkCreateImage",
            )
            self.image = image[0]
            requirements = ffi.new("VkMemoryRequirements*")
            lib.vkGetImageMemoryRequirements(device.device, self.image, requirements)
            self.memory = device.allocate(requirements[0], vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            lib.vkBindImageMemory(device.device, self.image, self.memory, 0)
        else:
            self.image = _handle(handle, "VkImage") if isinstance(handle, int) else handle
        kind = vk.VK_IMAGE_VIEW_TYPE_2D_ARRAY if array else vk.VK_IMAGE_VIEW_TYPE_2D
        self.view = self._view(kind, 0, levels)
        self.level_views = (
            [self._view(kind, level, 1) for level in range(levels)] if levels > 1 else []
        )

    def _view(self, kind, level, count):
        view = ffi.new("VkImageView*")
        check(
            lib.vkCreateImageView(
                self.device.device,
                ffi.new(
                    "VkImageViewCreateInfo*",
                    {
                        "sType": vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                        "image": self.image,
                        "viewType": kind,
                        "format": self.format,
                        "subresourceRange": {
                            "aspectMask": self.aspect,
                            "baseMipLevel": level,
                            "levelCount": count,
                            "baseArrayLayer": 0,
                            "layerCount": self.layers,
                        },
                    },
                ),
                ffi.NULL,
                view,
            ),
            "vkCreateImageView",
        )
        return view[0]

    def range(self):
        return {
            "aspectMask": self.aspect,
            "baseMipLevel": 0,
            "levelCount": self.levels,
            "baseArrayLayer": 0,
            "layerCount": self.layers,
        }

    def close(self):
        if self.view is None:
            return
        for view in (self.view, *self.level_views):
            lib.vkDestroyImageView(self.device.device, view, ffi.NULL)
        self.view, self.level_views = None, []
        if self.owned:
            lib.vkDestroyImage(self.device.device, self.image, ffi.NULL)
            lib.vkFreeMemory(self.device.device, self.memory, ffi.NULL)


class Ring:
    """A host-visible buffer handed out front to back each frame; it grows (the old one closed
    once its frames are done) when a frame needs more."""

    def __init__(self, device, usage):
        self.device, self.usage = device, usage
        self.buffer = Buffer(device, RING, usage, host=True)
        self.used = 0

    def take(self, data):
        data = memoryview(np.ascontiguousarray(data)).cast("B")
        size = len(data)
        offset = -(-self.used // self.device.alignment) * self.device.alignment
        if offset + size > self.buffer.size:
            self.device.defer(self.buffer)
            self.buffer = Buffer(
                self.device, max(2 * self.buffer.size, 2 * size), self.usage, host=True
            )
            offset = 0
        self.buffer.mapped[offset : offset + size] = data
        self.used = offset + size
        return self.buffer, offset, size

    def close(self):
        self.buffer.close()


class Frame:
    """One frame's command buffer, its fence and its rings; the methods record into it."""

    def __init__(self, device):
        self.device = device
        buffer = ffi.new("VkCommandBuffer*")
        check(
            lib.vkAllocateCommandBuffers(
                device.device,
                ffi.new(
                    "VkCommandBufferAllocateInfo*",
                    {
                        "sType": vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                        "commandPool": device.pool,
                        "level": vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                        "commandBufferCount": 1,
                    },
                ),
                buffer,
            ),
            "vkAllocateCommandBuffers",
        )
        self.cmd = buffer[0]
        fence = ffi.new("VkFence*")
        check(
            lib.vkCreateFence(
                device.device,
                ffi.new(
                    "VkFenceCreateInfo*",
                    {
                        "sType": vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                        "flags": vk.VK_FENCE_CREATE_SIGNALED_BIT,
                    },
                ),
                ffi.NULL,
                fence,
            ),
            "vkCreateFence",
        )
        self.fence = fence[0]
        self.uniforms = Ring(
            device,
            vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        self.staging = Ring(
            device,
            vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        )
        self.pending = False  # submitted, its fence not waited for yet
        self.done = []  # run once the frame's GPU work is complete (readbacks)
        self.reading = []  # readback buffers in use; free again then
        self.readbacks = []  # free readback buffers
        self.pipeline = None

    def wait(self):
        """Wait for the frame's GPU work, then run what waited for it (a frame recorded but
        never submitted drops it)."""
        done, self.done = self.done, []
        reading, self.reading = self.reading, []
        try:
            if self.pending:
                lib.vkWaitForFences(
                    self.device.device, 1, ffi.new("VkFence*", self.fence), 1, INFINITE
                )
                self.pending = False
                for callback in done:
                    callback()
        finally:
            self.readbacks += reading

    def begin(self):
        self.wait()
        lib.vkResetFences(self.device.device, 1, ffi.new("VkFence*", self.fence))
        self.uniforms.used = self.staging.used = 0
        self.pipeline = None
        lib.vkResetCommandBuffer(self.cmd, 0)
        check(
            lib.vkBeginCommandBuffer(
                self.cmd,
                ffi.new(
                    "VkCommandBufferBeginInfo*",
                    {
                        "sType": vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                        "flags": vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                    },
                ),
            ),
            "vkBeginCommandBuffer",
        )
        # Whatever the frames before this one left (uploads, the camera's own submission).
        self.barrier()

    def submit(self, wait, signal):
        check(lib.vkEndCommandBuffer(self.cmd), "vkEndCommandBuffer")
        waits = ffi.new(
            "VkSemaphoreSubmitInfo[]",
            [
                {
                    "sType": vk.VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                    "semaphore": s,
                    "stageMask": stage,
                    "value": value,
                }
                for s, stage, *value in wait
                for value in (value[0] if value else 0,)
            ]
            or 1,
        )
        signals = ffi.new(
            "VkSemaphoreSubmitInfo[]",
            [
                {
                    "sType": vk.VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                    "semaphore": s,
                    "stageMask": ALL,
                }
                for s in signal
            ]
            or 1,
        )
        command = ffi.new(
            "VkCommandBufferSubmitInfo*",
            {"sType": vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, "commandBuffer": self.cmd},
        )
        info = ffi.new(
            "VkSubmitInfo2*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                "waitSemaphoreInfoCount": len(wait),
                "pWaitSemaphoreInfos": waits,
                "commandBufferInfoCount": 1,
                "pCommandBufferInfos": command,
                "signalSemaphoreInfoCount": len(signal),
                "pSignalSemaphoreInfos": signals,
            },
        )
        check(lib.vkQueueSubmit2(self.device.queue, 1, info, self.fence), "vkQueueSubmit2")
        self.pending = True

    def close(self):
        if self.fence is None:
            return
        self.wait()
        self.uniforms.close()
        self.staging.close()
        for buffer in self.readbacks:
            buffer.close()
        lib.vkDestroyFence(self.device.device, self.fence, ffi.NULL)
        lib.vkFreeCommandBuffers(
            self.device.device, self.device.pool, 1, ffi.new("VkCommandBuffer*", self.cmd)
        )
        self.fence = None

    # -- data ----------------------------------------------------------------------------
    def uniform(self, data):
        """`data` in this frame's uniform ring: (buffer, offset, size) for a descriptor."""
        return self.uniforms.take(data)

    def stage(self, data):
        """`data` in this frame's staging ring (also usable as vertex or index data)."""
        return self.staging.take(data)

    def upload(self, buffer, data, offset=0):
        source, at, size = self.stage(data)
        if size:
            region = ffi.new("VkBufferCopy*", {"srcOffset": at, "dstOffset": offset, "size": size})
            lib.vkCmdCopyBuffer(self.cmd, source.buffer, buffer.buffer, 1, region)

    def upload_image(self, image, data, rect, layer=0):
        """`data` (rows of the rectangle's width) into image's (x, y, width, height) at
        `layer`."""
        source, at, _ = self.stage(data)
        self.transition(image, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        x, y, w, h = rect
        region = ffi.new(
            "VkBufferImageCopy*",
            {
                "bufferOffset": at,
                "imageSubresource": {
                    "aspectMask": COLOUR,
                    "mipLevel": 0,
                    "baseArrayLayer": layer,
                    "layerCount": 1,
                },
                "imageOffset": {"x": x, "y": y, "z": 0},
                "imageExtent": {"width": w, "height": h, "depth": 1},
            },
        )
        lib.vkCmdCopyBufferToImage(self.cmd, source.buffer, image.image, image.layout, 1, region)

    def copy_buffer_to_image(self, buffer, image, *, offset=0, row=0):
        """All of image from buffer at offset, rows of `row` texels (0: the image's width)."""
        self.transition(image, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        region = ffi.new(
            "VkBufferImageCopy*",
            {
                "bufferOffset": offset,
                "bufferRowLength": row,
                "imageSubresource": {"aspectMask": COLOUR, "layerCount": 1},
                "imageExtent": {"width": image.size[0], "height": image.size[1], "depth": 1},
            },
        )
        lib.vkCmdCopyBufferToImage(self.cmd, buffer.buffer, image.image, image.layout, 1, region)

    def read(self, image, callback):
        """Copy image to host memory; callback(pixels, (w, h)) runs once the frame is done
        (pixels: a uint8 array of RGBA texels, rows top first as stored)."""
        w, h = image.size
        size = w * h * 4
        target = next((b for b in self.readbacks if b.size >= size), None)
        if target is None:
            target = Buffer(self.device, size, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, read=True)
        else:
            self.readbacks.remove(target)
        self.transition(image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        region = ffi.new(
            "VkBufferImageCopy*",
            {
                "imageSubresource": {"aspectMask": COLOUR, "layerCount": 1},
                "imageExtent": {"width": w, "height": h, "depth": 1},
            },
        )
        lib.vkCmdCopyImageToBuffer(self.cmd, image.image, image.layout, target.buffer, 1, region)
        # The copy made visible to the CPU's reads once the fence says the frame is done.
        self.barrier(
            (vk.VK_PIPELINE_STAGE_2_COPY_BIT, vk.VK_ACCESS_2_TRANSFER_WRITE_BIT),
            (vk.VK_PIPELINE_STAGE_2_HOST_BIT, vk.VK_ACCESS_2_HOST_READ_BIT),
        )
        self.reading.append(target)
        self.done.append(lambda: callback(target.mapped[:size].copy(), (w, h)))

    # -- synchronisation -----------------------------------------------------------------
    def barrier(self, before=(ALL, MEMORY), after=(ALL, MEMORY)):
        """Everything recorded (and submitted) before, done and visible to everything after
        (or only the (stages, accesses) given)."""
        barrier = ffi.new(
            "VkMemoryBarrier2*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                "srcStageMask": before[0],
                "srcAccessMask": before[1],
                "dstStageMask": after[0],
                "dstAccessMask": after[1],
            },
        )
        info = ffi.new(
            "VkDependencyInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                "memoryBarrierCount": 1,
                "pMemoryBarriers": barrier,
            },
        )
        lib.vkCmdPipelineBarrier2(self.cmd, info)

    def transition(self, image, layout):
        """Image into `layout`, after everything before it (a full barrier for it, also when it
        is in that layout already)."""
        barrier = ffi.new(
            "VkImageMemoryBarrier2*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                "srcStageMask": ALL,
                "srcAccessMask": MEMORY,
                "dstStageMask": ALL,
                "dstAccessMask": MEMORY,
                "oldLayout": image.layout,
                "newLayout": layout,
                "srcQueueFamilyIndex": vk.VK_QUEUE_FAMILY_IGNORED,
                "dstQueueFamilyIndex": vk.VK_QUEUE_FAMILY_IGNORED,
                "image": image.image,
                "subresourceRange": image.range(),
            },
        )
        lib.vkCmdPipelineBarrier2(
            self.cmd,
            ffi.new(
                "VkDependencyInfo*",
                {
                    "sType": vk.VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    "imageMemoryBarrierCount": 1,
                    "pImageMemoryBarriers": barrier,
                },
            ),
        )
        image.layout = layout

    def sampled(self, *images):
        for image in images:
            if image.layout != vk.VK_IMAGE_LAYOUT_GENERAL:
                self.transition(image, vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)

    # -- transfers -----------------------------------------------------------------------
    def clear(self, image, colour, layers=None):
        """Clear all of image (or its `layers`, a range) to colour."""
        if image.layout != vk.VK_IMAGE_LAYOUT_GENERAL:
            self.transition(image, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        subresource = image.range()
        if layers is not None:
            subresource.update(baseArrayLayer=layers.start, layerCount=len(layers))
        lib.vkCmdClearColorImage(
            self.cmd,
            image.image,
            image.layout,
            ffi.new("VkClearColorValue*", {"float32": list(colour)}),
            1,
            ffi.new("VkImageSubresourceRange*", subresource),
        )

    def copy_image(self, source, target, *, size=None, layers=(0, 0), count=1):
        """source's layers from layers[0] into target's from layers[1] (size: their extent)."""
        if target is source:  # between its layers
            self.transition(source, vk.VK_IMAGE_LAYOUT_GENERAL)
        else:
            self.transition(source, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
            self.transition(target, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        w, h = size or source.size
        region = ffi.new(
            "VkImageCopy*",
            {
                "srcSubresource": {
                    "aspectMask": COLOUR,
                    "baseArrayLayer": layers[0],
                    "layerCount": count,
                },
                "dstSubresource": {
                    "aspectMask": COLOUR,
                    "baseArrayLayer": layers[1],
                    "layerCount": count,
                },
                "extent": {"width": w, "height": h, "depth": 1},
            },
        )
        lib.vkCmdCopyImage(
            self.cmd, source.image, source.layout, target.image, target.layout, 1, region
        )

    # -- drawing -------------------------------------------------------------------------
    @contextmanager
    def render(self, colours=(), depth=None, *, clear=None, depth_clear=None, size=None):
        """Draw into these images (all of them; size: the area), cleared to `clear` (colour)
        and `depth_clear` when given, else loaded."""
        images = [*colours, *([depth] if depth is not None else [])]
        self.barrier()
        for image in colours:
            self.transition(image, vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        if depth is not None:
            self.transition(depth, vk.VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
        w, h = size or images[0].size

        def attachment(image, value):
            info = {
                "sType": vk.VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                "imageView": image.level_views[0] if image.levels > 1 else image.view,
                "imageLayout": image.layout,
                "loadOp": vk.VK_ATTACHMENT_LOAD_OP_LOAD
                if value is None
                else vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
                "storeOp": vk.VK_ATTACHMENT_STORE_OP_STORE,
            }
            if value is not None:
                info["clearValue"] = (
                    {"depthStencil": {"depth": value}}
                    if image.aspect == DEPTH
                    else {"color": {"float32": list(value)}}
                )
            return info

        colour_infos = ffi.new(
            "VkRenderingAttachmentInfo[]", [attachment(i, clear) for i in colours] or 1
        )
        depth_info = (
            ffi.new("VkRenderingAttachmentInfo*", attachment(depth, depth_clear))
            if depth
            else ffi.NULL
        )
        info = ffi.new(
            "VkRenderingInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_RENDERING_INFO,
                "renderArea": {"offset": {"x": 0, "y": 0}, "extent": {"width": w, "height": h}},
                "layerCount": 1,
                "colorAttachmentCount": len(colours),
                "pColorAttachments": colour_infos,
                "pDepthAttachment": depth_info,
            },
        )
        lib.vkCmdBeginRendering(self.cmd, info)
        self.area((0, 0, w, h))
        self.pipeline = None
        try:
            yield
        finally:
            lib.vkCmdEndRendering(self.cmd)

    def area(self, viewport, scissor=None):
        x, y, w, h = viewport
        lib.vkCmdSetViewport(
            self.cmd,
            0,
            1,
            ffi.new("VkViewport*", {"x": x, "y": y, "width": w, "height": h, "maxDepth": 1.0}),
        )
        sx, sy, sw, sh = scissor or viewport
        lib.vkCmdSetScissor(
            self.cmd,
            0,
            1,
            ffi.new(
                "VkRect2D*",
                {
                    "offset": {"x": int(sx), "y": int(sy)},
                    "extent": {"width": int(sw), "height": int(sh)},
                },
            ),
        )

    def clear_rect(self, rect, colour=None, depth=None):
        """Within render(): clear rect (x, y, w, h) of colour attachment 0 and/or the depth."""
        x, y, w, h = rect
        attachments = []
        if colour is not None:
            attachments.append(
                {
                    "aspectMask": COLOUR,
                    "colorAttachment": 0,
                    "clearValue": {"color": {"float32": list(colour)}},
                }
            )
        if depth is not None:
            attachments.append(
                {"aspectMask": DEPTH, "clearValue": {"depthStencil": {"depth": depth}}}
            )
        lib.vkCmdClearAttachments(
            self.cmd,
            len(attachments),
            ffi.new("VkClearAttachment[]", attachments),
            1,
            ffi.new(
                "VkClearRect*",
                {
                    "rect": {"offset": {"x": x, "y": y}, "extent": {"width": w, "height": h}},
                    "layerCount": 1,
                },
            ),
        )

    def use(self, pipeline, descriptors=None, push=None):
        """Bind pipeline, push its descriptors ({binding: resource}: (buffer, offset, size)
        for a buffer, (image, sampler) for a sampled image, an image for a storage image or
        (image, level) for one level of it) and its push constants (bytes)."""
        if self.pipeline is not pipeline:
            lib.vkCmdBindPipeline(self.cmd, pipeline.point, pipeline.pipeline)
            self.pipeline = pipeline
        if descriptors:
            writes, keep = [], []
            for binding, resource in descriptors.items():
                kind = pipeline.kinds[binding]
                write = {
                    "sType": vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    "dstBinding": binding,
                    "descriptorCount": 1,
                    "descriptorType": kind,
                }
                if kind in (
                    vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                ):
                    buffer, offset, size = resource
                    info = ffi.new(
                        "VkDescriptorBufferInfo*",
                        {"buffer": buffer.buffer, "offset": offset, "range": size},
                    )
                    write["pBufferInfo"] = info
                elif kind == vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                    image, sampler = resource
                    info = ffi.new(
                        "VkDescriptorImageInfo*",
                        {"sampler": sampler, "imageView": image.view, "imageLayout": image.layout},
                    )
                    write["pImageInfo"] = info
                else:
                    image, level = resource if isinstance(resource, tuple) else (resource, None)
                    view = image.view if level is None else image.level_views[level]
                    info = ffi.new(
                        "VkDescriptorImageInfo*",
                        {"imageView": view, "imageLayout": vk.VK_IMAGE_LAYOUT_GENERAL},
                    )
                    write["pImageInfo"] = info
                keep.append(info)
                writes.append(write)
            array = ffi.new("VkWriteDescriptorSet[]", writes)
            self.device.push_descriptors(
                self.cmd, pipeline.point, pipeline.layout, 0, len(writes), array
            )
        if push:
            lib.vkCmdPushConstants(
                self.cmd, pipeline.layout, pipeline.stages, 0, len(push), ffi.from_buffer(push)
            )

    def vertices(self, *buffers, index=None):
        """Bind vertex buffers (Buffer, or (Buffer, offset)) and an index buffer (uint32, or
        (Buffer, offset, uint16?))."""
        handles, starts = [], []
        for item in buffers:
            buffer, offset = item if isinstance(item, tuple) else (item, 0)
            handles.append(buffer.buffer)
            starts.append(offset)
        lib.vkCmdBindVertexBuffers(
            self.cmd,
            0,
            len(handles),
            ffi.new("VkBuffer[]", handles),
            ffi.new("VkDeviceSize[]", starts),
        )
        if index is not None:
            buffer, offset, short = index if isinstance(index, tuple) else (index, 0, False)
            lib.vkCmdBindIndexBuffer(
                self.cmd,
                buffer.buffer,
                offset,
                vk.VK_INDEX_TYPE_UINT16 if short else vk.VK_INDEX_TYPE_UINT32,
            )

    def draw(self, count, *, indexed=False, first=0, vertex_offset=0):
        if indexed:
            lib.vkCmdDrawIndexed(self.cmd, count, 1, first, vertex_offset, 0)
        else:
            lib.vkCmdDraw(self.cmd, count, 1, first, 0)

    def dispatch(self, size, group=8):
        lib.vkCmdDispatch(self.cmd, -(-size[0] // group), -(-size[1] // group), 1)


VERTEX_FORMATS = {
    "2f": (vk.VK_FORMAT_R32G32_SFLOAT, 8),
    "3f": (vk.VK_FORMAT_R32G32B32_SFLOAT, 12),
    "4f": (vk.VK_FORMAT_R32G32B32A32_SFLOAT, 16),
    "4u8": (vk.VK_FORMAT_R8G8B8A8_UNORM, 4),
}
STAGES = {
    "vert": vk.VK_SHADER_STAGE_VERTEX_BIT,
    "geom": vk.VK_SHADER_STAGE_GEOMETRY_BIT,
    "frag": vk.VK_SHADER_STAGE_FRAGMENT_BIT,
    "comp": vk.VK_SHADER_STAGE_COMPUTE_BIT,
}
KINDS = {
    "uniform": vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    "storage": vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    "sampler": vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    "image": vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
}
BLEND = {
    # ImGui's: colour over, alpha accumulated (premultiplied over a cleared transparent target).
    "over": (
        vk.VK_BLEND_FACTOR_SRC_ALPHA,
        vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        vk.VK_BLEND_FACTOR_ONE,
        vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    ),
}
COMPARE = {"<": vk.VK_COMPARE_OP_LESS, "<=": vk.VK_COMPARE_OP_LESS_OR_EQUAL}


def shader(device, name):
    path = SHADERS / f"{name}.spv"
    try:
        code = path.read_bytes()
    except FileNotFoundError:
        raise VulkanError(f"{path} is missing: run tools/shaders.py") from None
    module = ffi.new("VkShaderModule*")
    check(
        lib.vkCreateShaderModule(
            device.device,
            ffi.new(
                "VkShaderModuleCreateInfo*",
                {
                    "sType": vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                    "codeSize": len(code),
                    "pCode": ffi.cast("uint32_t*", ffi.from_buffer(code)),
                },
            ),
            ffi.NULL,
            module,
        ),
        f"Loading shader {name}",
    )
    return module[0]


class Pipeline:
    """A graphics or compute pipeline from shaders in ito/shaders.

    shaders: file names ("mesh.vert", "mesh.frag"; a compute pipeline one ".comp"). bindings:
    {binding: kind} in set 0, kind one of KINDS, visible to every stage. push: push constant
    bytes. constants: {constant_id: value} (common.glsl). vertex: [(location, binding, format)],
    one buffer per binding, formats VERTEX_FORMATS ("4u8" normalized bytes); strides: {binding:
    (stride, {location: offset})} where a buffer interleaves. colours: attachment formats;
    depth: a compare ("<", "<=") to test and write DEPTH_FORMAT,
    None for no depth attachment."""

    def __init__(
        self,
        device,
        shaders,
        bindings,
        *,
        push=0,
        constants=None,
        vertex=(),
        strides=None,
        colours=(),
        depth=None,
        blend=None,
        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    ):
        self.device = device
        stages = [(name.rsplit(".", 1)[1], name) for name in shaders]
        self.stages = 0
        for stage, _ in stages:
            self.stages |= STAGES[stage]
        self.kinds = {b: KINDS[k] for b, k in bindings.items()}
        compute = stages[0][0] == "comp"
        self.point = (
            vk.VK_PIPELINE_BIND_POINT_COMPUTE if compute else vk.VK_PIPELINE_BIND_POINT_GRAPHICS
        )
        set_layout = ffi.new("VkDescriptorSetLayout*")
        entries = ffi.new(
            "VkDescriptorSetLayoutBinding[]",
            [
                {"binding": b, "descriptorType": k, "descriptorCount": 1, "stageFlags": self.stages}
                for b, k in self.kinds.items()
            ]
            or 1,
        )
        check(
            lib.vkCreateDescriptorSetLayout(
                device.device,
                ffi.new(
                    "VkDescriptorSetLayoutCreateInfo*",
                    {
                        "sType": vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                        "flags": vk.VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                        "bindingCount": len(self.kinds),
                        "pBindings": entries,
                    },
                ),
                ffi.NULL,
                set_layout,
            ),
            "vkCreateDescriptorSetLayout",
        )
        self.set_layout = set_layout[0]
        ranges = ffi.new("VkPushConstantRange*", {"stageFlags": self.stages, "size": push})
        layout = ffi.new("VkPipelineLayout*")
        check(
            lib.vkCreatePipelineLayout(
                device.device,
                ffi.new(
                    "VkPipelineLayoutCreateInfo*",
                    {
                        "sType": vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                        "setLayoutCount": 1,
                        "pSetLayouts": set_layout,
                        "pushConstantRangeCount": 1 if push else 0,
                        "pPushConstantRanges": ranges,
                    },
                ),
                ffi.NULL,
                layout,
            ),
            "vkCreatePipelineLayout",
        )
        self.layout = layout[0]
        constants = constants or {}
        entries = ffi.new(
            "VkSpecializationMapEntry[]",
            [{"constantID": c, "offset": 4 * i, "size": 4} for i, c in enumerate(constants)] or 1,
        )
        data = b"".join(
            struct.pack("<f" if isinstance(v, float) else "<i", v) for v in constants.values()
        )
        data_buffer = ffi.from_buffer(data) if data else ffi.NULL
        specialization = ffi.new(
            "VkSpecializationInfo*",
            {
                "mapEntryCount": len(constants),
                "pMapEntries": entries,
                "dataSize": len(data),
                "pData": data_buffer,
            },
        )
        modules = [shader(device, name) for _, name in stages]
        entry = ffi.new("char[]", b"main")
        infos = ffi.new(
            "VkPipelineShaderStageCreateInfo[]",
            [
                {
                    "sType": vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    "stage": STAGES[stage],
                    "module": module,
                    "pName": entry,
                    "pSpecializationInfo": specialization,
                }
                for (stage, _), module in zip(stages, modules, strict=True)
            ],
        )
        handle = ffi.new("VkPipeline*")
        try:
            if compute:
                check(
                    lib.vkCreateComputePipelines(
                        device.device,
                        ffi.NULL,
                        1,
                        ffi.new(
                            "VkComputePipelineCreateInfo*",
                            {
                                "sType": vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                "stage": infos[0],
                                "layout": self.layout,
                            },
                        ),
                        ffi.NULL,
                        handle,
                    ),
                    f"Compute pipeline {shaders[0]}",
                )
            else:
                self._graphics(
                    infos,
                    len(stages),
                    handle,
                    vertex,
                    strides or {},
                    colours,
                    depth,
                    blend,
                    topology,
                    shaders,
                )
        finally:
            for module in modules:
                lib.vkDestroyShaderModule(device.device, module, ffi.NULL)
        self.pipeline = handle[0]

    def _graphics(
        self,
        infos,
        count,
        handle,
        vertex,
        strides,
        colours,
        depth,
        blend,
        topology,
        shaders,
    ):
        attributes, sizes = [], {}
        for location, binding, kind in vertex:
            fmt, size = VERTEX_FORMATS[kind]
            offset = sizes.get(binding, 0)
            if binding in strides:
                offset = strides[binding][1][location]
            attributes.append(
                {"location": location, "binding": binding, "format": fmt, "offset": offset}
            )
            sizes[binding] = offset + size
        buffers = [
            {
                "binding": b,
                "stride": strides[b][0] if b in strides else s,
                "inputRate": vk.VK_VERTEX_INPUT_RATE_VERTEX,
            }
            for b, s in sorted(sizes.items())
        ]
        binding_array = ffi.new("VkVertexInputBindingDescription[]", buffers or 1)
        attribute_array = ffi.new("VkVertexInputAttributeDescription[]", attributes or 1)
        vertex_info = ffi.new(
            "VkPipelineVertexInputStateCreateInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                "vertexBindingDescriptionCount": len(buffers),
                "pVertexBindingDescriptions": binding_array,
                "vertexAttributeDescriptionCount": len(attributes),
                "pVertexAttributeDescriptions": attribute_array,
            },
        )
        assembly = ffi.new(
            "VkPipelineInputAssemblyStateCreateInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                "topology": topology,
            },
        )
        viewport = ffi.new(
            "VkPipelineViewportStateCreateInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                "viewportCount": 1,
                "scissorCount": 1,
            },
        )
        raster = ffi.new(
            "VkPipelineRasterizationStateCreateInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                "polygonMode": vk.VK_POLYGON_MODE_FILL,
                "cullMode": vk.VK_CULL_MODE_NONE,
                "frontFace": vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                "lineWidth": 1.0,
            },
        )
        multisample = ffi.new(
            "VkPipelineMultisampleStateCreateInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                "rasterizationSamples": vk.VK_SAMPLE_COUNT_1_BIT,
            },
        )
        depth_state = ffi.new(
            "VkPipelineDepthStencilStateCreateInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                "depthTestEnable": depth is not None,
                "depthWriteEnable": depth is not None,
                "depthCompareOp": COMPARE.get(depth, vk.VK_COMPARE_OP_ALWAYS),
            },
        )
        factors = BLEND.get(blend)
        attachment = {"colorWriteMask": 0xF}
        if factors:
            attachment.update(
                blendEnable=1,
                srcColorBlendFactor=factors[0],
                dstColorBlendFactor=factors[1],
                colorBlendOp=vk.VK_BLEND_OP_ADD,
                srcAlphaBlendFactor=factors[2],
                dstAlphaBlendFactor=factors[3],
                alphaBlendOp=vk.VK_BLEND_OP_ADD,
            )
        attachments = ffi.new(
            "VkPipelineColorBlendAttachmentState[]", [attachment] * len(colours) or 1
        )
        blending = ffi.new(
            "VkPipelineColorBlendStateCreateInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                "attachmentCount": len(colours),
                "pAttachments": attachments,
            },
        )
        dynamic_states = ffi.new(
            "VkDynamicState[]", [vk.VK_DYNAMIC_STATE_VIEWPORT, vk.VK_DYNAMIC_STATE_SCISSOR]
        )
        dynamic = ffi.new(
            "VkPipelineDynamicStateCreateInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                "dynamicStateCount": 2,
                "pDynamicStates": dynamic_states,
            },
        )
        formats = ffi.new("VkFormat[]", list(colours) or 1)
        rendering = ffi.new(
            "VkPipelineRenderingCreateInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                "colorAttachmentCount": len(colours),
                "pColorAttachmentFormats": formats,
                "depthAttachmentFormat": DEPTH_FORMAT
                if depth is not None
                else vk.VK_FORMAT_UNDEFINED,
            },
        )
        info = ffi.new(
            "VkGraphicsPipelineCreateInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                "pNext": rendering,
                "stageCount": count,
                "pStages": infos,
                "pVertexInputState": vertex_info,
                "pInputAssemblyState": assembly,
                "pViewportState": viewport,
                "pRasterizationState": raster,
                "pMultisampleState": multisample,
                "pDepthStencilState": depth_state,
                "pColorBlendState": blending,
                "pDynamicState": dynamic,
                "layout": self.layout,
            },
        )
        check(
            lib.vkCreateGraphicsPipelines(self.device.device, ffi.NULL, 1, info, ffi.NULL, handle),
            f"Graphics pipeline {', '.join(shaders)}",
        )

    def close(self):
        if self.pipeline is not None:
            device = self.device.device
            lib.vkDestroyPipeline(device, self.pipeline, ffi.NULL)
            lib.vkDestroyPipelineLayout(device, self.layout, ffi.NULL)
            lib.vkDestroyDescriptorSetLayout(device, self.set_layout, ffi.NULL)
            self.pipeline = None


def semaphore(device):
    handle = ffi.new("VkSemaphore*")
    check(
        lib.vkCreateSemaphore(
            device.device,
            ffi.new(
                "VkSemaphoreCreateInfo*", {"sType": vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}
            ),
            ffi.NULL,
            handle,
        ),
        "vkCreateSemaphore",
    )
    return handle[0]


class Swapchain:
    """The window's images: acquire() one for the frame being recorded, present() it after the
    frame is submitted with wait() and signal() (Device.submit). Mailbox presentation where the
    driver has it (no tearing, never blocks), else immediate, else FIFO; a UNORM format where
    there is one (present.frag decodes onto an sRGB one)."""

    def __init__(self, device):
        self.device = device
        self.swapchain = None
        self.images, self.ready = [], []
        self.retired = []  # the semaphores of the images before the last rebuild
        self.size = None
        self.current = None
        self.stale = False
        self.acquired = [semaphore(device) for _ in range(FRAMES)]
        self.slot = 0
        get = device.instance_function
        self.capabilities = get("vkGetPhysicalDeviceSurfaceCapabilitiesKHR")
        self.formats = get("vkGetPhysicalDeviceSurfaceFormatsKHR")
        self.modes = get("vkGetPhysicalDeviceSurfacePresentModesKHR")
        function = device.device_function
        self.create = function("vkCreateSwapchainKHR")
        self.destroy = function("vkDestroySwapchainKHR")
        self.get_images = function("vkGetSwapchainImagesKHR")
        self.next = function("vkAcquireNextImageKHR")
        self.queue_present = function("vkQueuePresentKHR")

    def _list(self, function, kind, *args):
        count = ffi.new("uint32_t*")
        function(*args, count, ffi.NULL)
        items = ffi.new(f"{kind}[]", count[0])
        function(*args, count, items)
        return items  # its elements point into it: kept while they are used

    def _build(self, size):
        device = self.device
        lib.vkDeviceWaitIdle(device.device)
        capabilities = ffi.new("VkSurfaceCapabilitiesKHR*")
        self.capabilities(device.physical, device.surface, capabilities)
        caps = capabilities[0]
        if caps.currentExtent.width != 0xFFFFFFFF:
            size = (caps.currentExtent.width, caps.currentExtent.height)
        size = tuple(
            max(lo, min(hi, v))
            for v, lo, hi in zip(
                size,
                (caps.minImageExtent.width, caps.minImageExtent.height),
                (caps.maxImageExtent.width, caps.maxImageExtent.height),
                strict=True,
            )
        )
        if min(size) <= 0:
            return False
        formats = self._list(self.formats, "VkSurfaceFormatKHR", device.physical, device.surface)
        preferred = (
            vk.VK_FORMAT_B8G8R8A8_UNORM,
            vk.VK_FORMAT_R8G8B8A8_UNORM,
            vk.VK_FORMAT_B8G8R8A8_SRGB,
            vk.VK_FORMAT_R8G8B8A8_SRGB,
        )
        chosen = min(
            formats,
            key=lambda f: preferred.index(f.format) if f.format in preferred else len(preferred),
        )
        modes = self._list(self.modes, "VkPresentModeKHR", device.physical, device.surface)
        ranked = (
            vk.VK_PRESENT_MODE_MAILBOX_KHR,
            vk.VK_PRESENT_MODE_IMMEDIATE_KHR,
            vk.VK_PRESENT_MODE_FIFO_KHR,  # always there
        )
        mode = next(m for m in ranked if m in modes or m == vk.VK_PRESENT_MODE_FIFO_KHR)
        count = caps.minImageCount + 1
        if caps.maxImageCount:
            count = min(count, caps.maxImageCount)
        old = self.swapchain
        handle = ffi.new("VkSwapchainKHR*")
        check(
            self.create(
                device.device,
                ffi.new(
                    "VkSwapchainCreateInfoKHR*",
                    {
                        "sType": vk.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                        "surface": device.surface,
                        "minImageCount": count,
                        "imageFormat": chosen.format,
                        "imageColorSpace": chosen.colorSpace,
                        "imageExtent": {"width": size[0], "height": size[1]},
                        "imageArrayLayers": 1,
                        "imageUsage": vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                        | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        "imageSharingMode": vk.VK_SHARING_MODE_EXCLUSIVE,
                        "preTransform": caps.currentTransform,
                        "compositeAlpha": vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                        "presentMode": mode,
                        "clipped": 1,
                        "oldSwapchain": old if old is not None else ffi.NULL,
                    },
                ),
                ffi.NULL,
                handle,
            ),
            "Creating the window's swapchain",
        )
        self._release()
        if old is not None:
            self.destroy(device.device, old, ffi.NULL)
        self.swapchain = handle[0]
        self.format = chosen.format
        self.srgb = chosen.format in (vk.VK_FORMAT_B8G8R8A8_SRGB, vk.VK_FORMAT_R8G8B8A8_SRGB)
        self.bgr = chosen.format in (vk.VK_FORMAT_B8G8R8A8_UNORM, vk.VK_FORMAT_B8G8R8A8_SRGB)
        self.size = size
        handles = self._list(self.get_images, "VkImage", device.device, self.swapchain)
        self.images = [
            Image(device, size, chosen.format, handle=h, transfer=False) for h in handles
        ]
        self.ready = [semaphore(device) for _ in handles]
        self.stale = False
        return True

    def _release(self):
        for image in self.images:
            image.close()
        # vkDeviceWaitIdle does not cover the presentation engine's wait on an image's last
        # semaphore: those go a rebuild later, long after.
        for item in self.retired:
            lib.vkDestroySemaphore(self.device.device, item, ffi.NULL)
        self.images, self.ready, self.retired = [], [], self.ready

    def _abandon(self):
        """The image acquired last was never presented (its frame failed): an empty submission
        waits for its semaphore, so the semaphore can be signalled again, and the swapchain is
        rebuilt, which takes the image back."""
        wait = ffi.new(
            "VkSemaphoreSubmitInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                "semaphore": self.acquired[(self.slot - 1) % FRAMES],
                "stageMask": ALL,
            },
        )
        info = ffi.new(
            "VkSubmitInfo2*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                "waitSemaphoreInfoCount": 1,
                "pWaitSemaphoreInfos": wait,
            },
        )
        check(lib.vkQueueSubmit2(self.device.queue, 1, info, ffi.NULL), "vkQueueSubmit2")
        self.current = None
        self.stale = True

    def acquire(self, size):
        """The image to draw this frame into, the window at `size`; None while the window has
        no area (minimised)."""
        if self.current is not None:
            self._abandon()
        size = tuple(size)
        if min(size) <= 0:
            return None
        if self.swapchain is None or self.stale or size != self.size:
            if not self._build(size):
                return None
        index = ffi.new("uint32_t*")
        for _ in range(2):
            semaphore = self.acquired[self.slot % FRAMES]
            result = self.next(
                self.device.device, self.swapchain, INFINITE, semaphore, ffi.NULL, index
            )
            if result == vk.VK_ERROR_OUT_OF_DATE_KHR:
                if not self._build(size):
                    return None
                continue
            if result not in (vk.VK_SUCCESS, vk.VK_SUBOPTIMAL_KHR):
                check(result, "vkAcquireNextImageKHR")
            break
        else:
            return None
        self.slot += 1
        self.current = index[0]
        image = self.images[self.current]
        image.layout = vk.VK_IMAGE_LAYOUT_UNDEFINED
        return image

    def wait(self):
        """For Device.submit: the frame's drawing waits for the acquired image."""
        return [
            (
                self.acquired[(self.slot - 1) % FRAMES],
                vk.VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
                | vk.VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            )
        ]

    def signal(self):
        return [self.ready[self.current]]

    def finish(self, frame):
        """Record, last in the frame, the acquired image's hand-over to presentation."""
        frame.transition(self.images[self.current], vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)

    def present(self):
        current, self.current = self.current, None
        wait = ffi.new("VkSemaphore*", self.ready[current])
        swapchain = ffi.new("VkSwapchainKHR*", self.swapchain)
        index = ffi.new("uint32_t*", current)
        result = self.queue_present(
            self.device.queue,
            ffi.new(
                "VkPresentInfoKHR*",
                {
                    "sType": vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                    "waitSemaphoreCount": 1,
                    "pWaitSemaphores": wait,
                    "swapchainCount": 1,
                    "pSwapchains": swapchain,
                    "pImageIndices": index,
                },
            ),
        )
        if result in (vk.VK_ERROR_OUT_OF_DATE_KHR, vk.VK_SUBOPTIMAL_KHR):
            self.stale = True
        elif result != vk.VK_SUCCESS:
            check(result, "vkQueuePresentKHR")

    def close(self):
        if self.device.device is None:
            return
        lib.vkDeviceWaitIdle(self.device.device)
        self._release()
        for item in self.retired + self.acquired:
            lib.vkDestroySemaphore(self.device.device, item, ffi.NULL)
        self.retired, self.acquired = [], []
        if self.swapchain is not None:
            self.destroy(self.device.device, self.swapchain, ffi.NULL)
            self.swapchain = None


class Presenter:
    """An offscreen image onto a display image (present.frag), scaled to fill it."""

    def __init__(self, device):
        self.device = device
        self.pipelines = {}

    def __call__(self, frame, source, target, *, upright, srgb=False):
        pipeline = self.pipelines.get(target.format)
        if pipeline is None:
            pipeline = self.pipelines[target.format] = Pipeline(
                self.device,
                ("fullscreen.vert", "present.frag"),
                {0: "sampler"},
                push=8,
                colours=(target.format,),
            )
        frame.sampled(source)
        with frame.render([target]):
            frame.use(pipeline, {0: (source, self.device.sampler())}, pack("ii", upright, srgb))
            frame.draw(3)

    def close(self):
        for pipeline in self.pipelines.values():
            pipeline.close()
        self.pipelines = {}


def native_window():
    """pygame's window, for Device: (hinstance, hwnd) on Windows, (display, window) on X11."""
    import pygame

    info = pygame.display.get_wm_info()
    if sys.platform == "win32":
        kernel = ctypes.windll.kernel32
        kernel.GetModuleHandleW.restype = ctypes.c_void_p
        return kernel.GetModuleHandleW(None), info["window"]
    return info["display"], info["window"]
