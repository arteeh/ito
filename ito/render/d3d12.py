"""Decoded camera frames from Direct3D 12 to Vulkan, all on the GPU (Windows).

FFmpeg's D3D12 video decoder (media.VideoDecoder) leaves each frame in an NV12 texture of its
own that only Direct3D 12 can use. SharedFrames copies its two planes, on a D3D12 copy queue
once the decoder's fence says it is decoded, into a buffer created shared with Vulkan, and
signals a fence shared with Vulkan as a timeline semaphore. The camera's Vulkan work waits for
that value, copies the planes from the buffer into images of its own (luma R8, chroma R8G8)
and samples them as it does a frame uploaded from memory (camera.py). Nothing passes through
the CPU but the commands. (A buffer, not a shared NV12 texture: the GPU's Vulkan and Direct3D
12 need not lay out a texture's planes alike, and on an RTX 4070 Ti they do not.)

The COM interfaces are called through their vtables (method indices as d3d12.h declares them);
the D3D12 device is the decoder's own, and must be the GPU Vulkan draws on (same LUID)."""

import ctypes as ct
import uuid

import vulkan as vk

from .vk import Buffer, Image, check, ffi, lib

HRESULT = ct.c_long
# FFmpeg's AVD3D12VAFrame (libavutil/hwcontext_d3d12va.h of FFmpeg 8.1, the one PyAV 17.1.0
# bundles; pyproject.toml pins it): the texture, its subresource index, and the sync context:
# fence, event, value the decoder signals when the frame is decoded.
TEXTURE, SUBRESOURCE, FENCE, VALUE = 0, 8, 16, 32
NV12 = 103  # DXGI_FORMAT_NV12
PLANES = (vk.VK_FORMAT_R8_UNORM, 1), (vk.VK_FORMAT_R8G8_UNORM, 2)  # format, bytes per texel
COPY = 3  # D3D12_COMMAND_LIST_TYPE_COPY
HEAP_SHARED = 0x1
GENERIC_ALL = 0x10000000
EXTERNAL = 0xFFFFFFFE  # VK_QUEUE_FAMILY_EXTERNAL


def _iid(text):
    return (ct.c_ubyte * 16).from_buffer_copy(uuid.UUID(text).bytes_le)


IID_DEVICE = _iid("189819f1-1db6-4b57-be54-1821339b85f7")
IID_QUEUE = _iid("0ec870a6-5d7e-4c22-8cfc-5baae07616ed")
IID_ALLOCATOR = _iid("6102dee4-af59-4b09-b999-b44d73f09b24")
IID_LIST = _iid("5b160d0f-ac1b-4185-8ba8-b3ae42a5a455")
IID_RESOURCE = _iid("696442be-a72e-4059-bc79-5b5c98040fad")
IID_FENCE = _iid("0a753dcf-c4d8-4b91-adf6-be5a60d95a76")


class ResourceDesc(ct.Structure):
    _fields_ = [
        ("Dimension", ct.c_int),
        ("Alignment", ct.c_uint64),
        ("Width", ct.c_uint64),
        ("Height", ct.c_uint),
        ("DepthOrArraySize", ct.c_uint16),
        ("MipLevels", ct.c_uint16),
        ("Format", ct.c_int),
        ("SampleCount", ct.c_uint),
        ("SampleQuality", ct.c_uint),
        ("Layout", ct.c_int),
        ("Flags", ct.c_int),
    ]


class HeapProperties(ct.Structure):
    _fields_ = [
        ("Type", ct.c_int),
        ("CPUPageProperty", ct.c_int),
        ("MemoryPoolPreference", ct.c_int),
        ("CreationNodeMask", ct.c_uint),
        ("VisibleNodeMask", ct.c_uint),
    ]


class Footprint(ct.Structure):  # D3D12_PLACED_SUBRESOURCE_FOOTPRINT
    _fields_ = [
        ("Offset", ct.c_uint64),
        ("Format", ct.c_int),
        ("Width", ct.c_uint),
        ("Height", ct.c_uint),
        ("Depth", ct.c_uint),
        ("RowPitch", ct.c_uint),
    ]


class Where(ct.Union):
    _fields_ = [("PlacedFootprint", Footprint), ("SubresourceIndex", ct.c_uint)]


class CopyLocation(ct.Structure):  # D3D12_TEXTURE_COPY_LOCATION
    _fields_ = [("pResource", ct.c_void_p), ("Type", ct.c_int), ("where", Where)]


class QueueDesc(ct.Structure):
    _fields_ = [
        ("Type", ct.c_int),
        ("Priority", ct.c_int),
        ("Flags", ct.c_int),
        ("NodeMask", ct.c_uint),
    ]


def _call(obj, index, *args, restype=HRESULT):
    """Method `index` of COM object `obj` (an address); arguments are ctypes values, byref()s or
    None."""
    table = ct.cast(ct.c_void_p(obj), ct.POINTER(ct.POINTER(ct.c_void_p)))[0]
    kinds = [type(a) if isinstance(a, ct._SimpleCData) else ct.c_void_p for a in args]
    result = ct.WINFUNCTYPE(restype, ct.c_void_p, *kinds)(table[index])(obj, *args)
    if restype is HRESULT and result < 0:
        raise OSError(f"Direct3D 12 call {index} failed: 0x{result & 0xFFFFFFFF:08X}")
    return result


def _release(obj):
    if obj:
        _call(obj, 2, restype=ct.c_ulong)


def _new(obj, index, *args):
    """A COM method that returns an object through its last argument."""
    out = ct.c_void_p()
    _call(obj, index, *args, ct.byref(out))
    return out.value


def decoded(frame):
    """The D3D12 texture, subresource, fence and fence value of an av.VideoFrame FFmpeg's D3D12
    decoder left on the GPU, or None for a frame in memory."""
    if frame.format.name != "d3d12":
        return None
    at = frame.planes[0].buffer_ptr
    return (
        ct.c_void_p.from_address(at + TEXTURE).value,
        ct.c_int.from_address(at + SUBRESOURCE).value,
        ct.c_void_p.from_address(at + FENCE).value,
        ct.c_uint64.from_address(at + VALUE).value,
    )


class Unavailable(RuntimeError):
    pass


class SharedFrames:
    """The planes of the last frame copied (luma, chroma: vk.Image, after acquire()), and the
    timeline semaphore whose value copy() returns."""

    def __init__(self, device, texture):
        """device: Ito's vk.Device; texture: a decoded frame's (decoded()), whose D3D12 device
        does the copies, and whose size the shared buffer is made for."""
        self.device = device
        self.d3d12 = self.queue = self.allocator = self.list = self.fence = None
        self.resource = self.buffer = None
        self.planes = []
        self.footprints = []
        self.semaphore = None
        self.size = None
        self.value = 0
        if not device.interop:
            raise Unavailable("the GPU's Vulkan cannot import Direct3D 12 resources")
        desc = ResourceDesc()
        _call(texture, 10, ct.byref(desc), restype=ct.c_void_p)  # GetDesc
        if desc.Format != NV12:
            raise Unavailable(f"the decoder's frames are DXGI format {desc.Format}, not NV12")
        self.d3d12 = d3d12_device = _new(texture, 7, ct.byref(IID_DEVICE))  # GetDevice
        try:
            self._check_adapter()
            self._check_import()
            queue = QueueDesc(Type=COPY)
            self.queue = _new(d3d12_device, 8, ct.byref(queue), ct.byref(IID_QUEUE))
            self.allocator = _new(d3d12_device, 9, ct.c_int(COPY), ct.byref(IID_ALLOCATOR))
            self.list = _new(
                d3d12_device, 12, ct.c_uint(0), ct.c_int(COPY), ct.c_void_p(self.allocator),
                None, ct.byref(IID_LIST),
            )  # fmt: skip
            _call(self.list, 9)  # Close: Reset opens it for each copy
            self.fence = _new(
                d3d12_device, 36, ct.c_uint64(0), ct.c_int(1), ct.byref(IID_FENCE)
            )  # D3D12_FENCE_FLAG_SHARED
            self.semaphore = self._import_fence()
            self._shared_buffer(desc)
        except BaseException:
            self.close()
            raise

    def _check_adapter(self):
        luid = (ct.c_ubyte * 8)()
        _call(self.d3d12, 43, ct.byref(luid), restype=ct.c_void_p)  # GetAdapterLuid
        ids = ffi.new(
            "VkPhysicalDeviceIDProperties*",
            {"sType": vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES},
        )
        properties = ffi.new(
            "VkPhysicalDeviceProperties2*",
            {"sType": vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, "pNext": ids},
        )
        lib.vkGetPhysicalDeviceProperties2(self.device.physical, properties)
        if not ids.deviceLUIDValid or bytes(ids.deviceLUID) != bytes(luid):
            raise Unavailable("the video decoder runs on another GPU than Ito draws on")

    def _check_import(self):
        info = ffi.new(
            "VkPhysicalDeviceExternalBufferInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
                "usage": vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                "handleType": vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
            },
        )
        result = ffi.new(
            "VkExternalBufferProperties*",
            {"sType": vk.VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES},
        )
        lib.vkGetPhysicalDeviceExternalBufferProperties(self.device.physical, info, result)
        features = result.externalMemoryProperties.externalMemoryFeatures
        if not features & vk.VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT:
            raise Unavailable("the GPU's Vulkan cannot import Direct3D 12 buffers")

    def _shared_handle(self, obj):
        handle = ct.c_void_p()
        _call(
            self.d3d12,
            31,
            ct.c_void_p(obj),
            None,
            ct.c_ulong(GENERIC_ALL),
            None,
            ct.byref(handle),
        )  # fmt: skip  CreateSharedHandle
        return handle.value

    def _import_fence(self):
        handle = self._shared_handle(self.fence)
        try:
            return self._import_semaphore(handle)
        finally:
            ct.windll.kernel32.CloseHandle(ct.c_void_p(handle))

    def _import_semaphore(self, handle):
        kind = ffi.new(
            "VkSemaphoreTypeCreateInfo*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                "semaphoreType": vk.VK_SEMAPHORE_TYPE_TIMELINE,
            },
        )
        semaphore = ffi.new("VkSemaphore*")
        check(
            lib.vkCreateSemaphore(
                self.device.device,
                ffi.new(
                    "VkSemaphoreCreateInfo*",
                    {"sType": vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, "pNext": kind},
                ),
                ffi.NULL,
                semaphore,
            ),
            "vkCreateSemaphore",
        )
        try:
            check(
                self.device.device_function("vkImportSemaphoreWin32HandleKHR")(
                    self.device.device,
                    ffi.new(
                        "VkImportSemaphoreWin32HandleInfoKHR*",
                        {
                            "sType": vk.VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR,
                            "semaphore": semaphore[0],
                            "handleType": vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT,
                            "handle": ffi.cast("HANDLE", handle),
                        },
                    ),
                ),
                "Importing the Direct3D 12 fence",
            )
        except BaseException:
            lib.vkDestroySemaphore(self.device.device, semaphore[0], ffi.NULL)
            raise
        return semaphore[0]

    def _shared_buffer(self, desc):
        """The buffer the planes of frames like desc's are copied into, shared with Vulkan, and
        the Vulkan images they are copied on into."""
        self._release_buffer()
        layout = ResourceDesc.from_buffer_copy(desc)
        layout.DepthOrArraySize = 1
        self.footprints = (Footprint * 2)()
        total = ct.c_uint64()
        _call(
            self.d3d12,
            38,
            ct.byref(layout),
            ct.c_uint(0),
            ct.c_uint(2),
            ct.c_uint64(0),
            ct.byref(self.footprints),
            None,
            None,
            ct.byref(total),
            restype=None,
        )  # fmt: skip  GetCopyableFootprints
        heap = HeapProperties(Type=1)  # D3D12_HEAP_TYPE_DEFAULT
        buffer = ResourceDesc(
            Dimension=1,
            Width=total.value,
            Height=1,
            DepthOrArraySize=1,
            MipLevels=1,
            SampleCount=1,
            Layout=1,
        )  # fmt: skip  D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_TEXTURE_LAYOUT_ROW_MAJOR
        self.resource = _new(
            self.d3d12,
            27,
            ct.byref(heap),
            ct.c_int(HEAP_SHARED),
            ct.byref(buffer),
            ct.c_int(0),
            None,
            ct.byref(IID_RESOURCE),
        )  # fmt: skip  CreateCommittedResource, in D3D12_RESOURCE_STATE_COMMON
        shared = self._shared_handle(self.resource)
        try:
            self.buffer = Buffer.imported(
                self.device, total.value, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, shared
            )
        finally:
            ct.windll.kernel32.CloseHandle(ct.c_void_p(shared))
        for f, (format, _) in zip(self.footprints, PLANES, strict=True):
            self.planes.append(Image(self.device, (f.Width, f.Height), format, sampled=True))
        self.size = (desc.Width, desc.Height)

    def _release_buffer(self):
        """The shared buffer and planes, whatever of them was made."""
        if self.buffer is not None or self.planes:
            self.device.idle()
        for plane in self.planes:
            plane.close()
        if self.buffer is not None:
            self.buffer.close()
        _release(self.resource)
        self.planes, self.buffer, self.resource, self.size = [], None, None, None

    def copy(self, decoded):
        """Copy a decoded frame's planes ((texture, subresource, fence, value): decoded()) into
        the shared buffer; the timeline value Vulkan waits for before acquire()'s copies run."""
        texture, subresource, fence, value = decoded
        desc = ResourceDesc()
        _call(texture, 10, ct.byref(desc), restype=ct.c_void_p)  # GetDesc
        if self.size != (desc.Width, desc.Height):
            self._shared_buffer(desc)
        _call(self.queue, 15, ct.c_void_p(fence), ct.c_uint64(value))  # Wait for the decoder
        _call(self.allocator, 8)  # Reset: the last copy is done (camera.CameraFeed.collect)
        _call(self.list, 10, ct.c_void_p(self.allocator), None)  # Reset
        for plane, footprint in enumerate(self.footprints):
            destination = CopyLocation(pResource=self.resource, Type=1)  # PLACED_FOOTPRINT
            destination.where.PlacedFootprint = footprint
            source = CopyLocation(pResource=texture, Type=0)  # SUBRESOURCE_INDEX
            # NV12's planes are subresources of their own, plane-major.
            source.where.SubresourceIndex = subresource + plane * desc.DepthOrArraySize
            _call(
                self.list,
                16,
                ct.byref(destination),
                ct.c_uint(0),
                ct.c_uint(0),
                ct.c_uint(0),
                ct.byref(source),
                None,
                restype=None,
            )  # fmt: skip  CopyTextureRegion
        _call(self.list, 9)  # Close
        lists = (ct.c_void_p * 1)(self.list)
        _call(self.queue, 10, ct.c_uint(1), ct.byref(lists), restype=None)  # ExecuteCommandLists
        _call(self.queue, 14, ct.c_void_p(self.fence), ct.c_uint64(self.value + 1))  # Signal
        self.value += 1
        return self.value

    def acquire(self, frame):
        """Record what comes before the planes are sampled: Vulkan takes the buffer over from
        D3D12 and copies the planes out of it."""
        barrier = ffi.new(
            "VkBufferMemoryBarrier2*",
            {
                "sType": vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                "dstStageMask": vk.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                "dstAccessMask": vk.VK_ACCESS_2_MEMORY_READ_BIT,
                "srcQueueFamilyIndex": EXTERNAL,
                "dstQueueFamilyIndex": self.device.family,
                "buffer": self.buffer.buffer,
                "size": self.buffer.size,
            },
        )
        lib.vkCmdPipelineBarrier2(
            frame.cmd,
            ffi.new(
                "VkDependencyInfo*",
                {
                    "sType": vk.VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    "bufferMemoryBarrierCount": 1,
                    "pBufferMemoryBarriers": barrier,
                },
            ),
        )
        for plane, footprint, (_, size) in zip(self.planes, self.footprints, PLANES, strict=True):
            frame.copy_buffer_to_image(
                self.buffer, plane, offset=footprint.Offset, row=footprint.RowPitch // size
            )

    def close(self):
        if self.fence is not None and self.value:
            # Whatever was copied last is done before its buffer goes.
            event = ct.windll.kernel32.CreateEventW(None, 0, 0, None)
            _call(self.fence, 9, ct.c_uint64(self.value), ct.c_void_p(event))
            ct.windll.kernel32.WaitForSingleObject(ct.c_void_p(event), 5000)
            ct.windll.kernel32.CloseHandle(ct.c_void_p(event))
        self._release_buffer()
        if self.semaphore is not None:
            lib.vkDestroySemaphore(self.device.device, self.semaphore, ffi.NULL)
        for obj in (self.list, self.allocator, self.queue, self.fence, self.d3d12):
            _release(obj)
        self.semaphore = self.list = self.allocator = self.queue = self.fence = self.d3d12 = None
