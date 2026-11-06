"""The robot camera's frames, straightened on the GPU.

Frames arrive as Ito decodes them, Y'CbCr in memory (NV12 from a GPU decoder, planar from the
CPU's), and through the robot's lens when its camera describes one (protocol.Lens). One compute
pass (straighten.comp) turns each into RGB at the camera's described pinhole: the flat camera
view's texture, and the ToF mesh's colour frame, strided as the worker takes it
(worker.frame_size). Each frame is straightened in a submission of its own, ahead of the frame
being drawn, so its colour frame is read back as soon as the GPU has it (collect()) without
waiting for the rest of the frame, nor holding the frame loop while the GPU is busy.

A frame the D3D12 decoder left on the GPU is copied there into textures Vulkan shares
(d3d12.SharedFrames). Where that cannot be set up, the decoder is told to download its frames
as any other decoder does (media.VideoDecoder.share), and those that reach here before it does
are downloaded here (av.VideoFrame.reformat)."""

import logging

import numpy as np
import vulkan as vk

from ito import clock, trace
from ito.link import media
from ito.tofmesh.worker import frame_size

from . import d3d12
from .vk import ALL, FRAMES, Frame, Image, Pipeline, VulkanError, mat, pack

log = logging.getLogger(__name__)

# Luma weights (Kr, Kb) of ITU-R BT.601 and BT.709. FFmpeg's AVColorSpace 1 is BT.709; every
# other value, unspecified included, is converted as BT.601, as FFmpeg's own conversion does.
BT601, BT709 = (0.299, 0.114), (0.2126, 0.0722)
AVCOL_SPC_BT709, AVCOL_RANGE_JPEG = 1, 2


def ycc_to_rgb(colorspace, full):
    """The matrix and offset that take a frame's sampled Y'CbCr (0-1) to R'G'B'."""
    kr, kb = BT709 if colorspace == AVCOL_SPC_BT709 else BT601
    kg = 1 - kr - kb
    matrix = np.array(
        [
            [1, 0, 2 - 2 * kr],
            [1, -kb * (2 - 2 * kb) / kg, -kr * (2 - 2 * kr) / kg],
            [1, 2 - 2 * kb, 0],
        ]
    )
    if full:
        return matrix, (0.0, 128 / 255, 128 / 255)
    return matrix * (255 / 219, 255 / 224, 255 / 224), (16 / 255, 128 / 255, 128 / 255)


RGBA = vk.VK_FORMAT_R8G8B8A8_UNORM
# Colour frames kept: each straightened frame's stays until this many newer ones (its pose can
# trail it, the window holds it until then).
COLOURS = 8
PLANES = {1: vk.VK_FORMAT_R8_UNORM, 2: vk.VK_FORMAT_R8G8_UNORM}


class CameraFeed:
    def __init__(self, device):
        self.device = device
        self.pipeline = Pipeline(
            device,
            ("straighten.comp",),
            {0: "uniform", 1: "sampler", 2: "sampler", 3: "sampler", 4: "image"},
        )
        self.frames = [Frame(device) for _ in range(FRAMES)]
        for frame in self.frames:
            frame.label = "camera"
        self.count = 0
        self.pending = None
        self.planes = []  # the planes the frame being straightened is sampled from
        self.uploaded = []  # the planes frames from memory are uploaded into, laid out so:
        self.layout = None
        self.texture = None  # the latest frame, straightened, full size
        self.colours = []  # the same, strided to the ToF mesh's colour size: the last COLOURS
        self.colour = None  # ... of which the latest frame's
        self.straightened = 0  # frames straightened
        self.sizes = None
        self.stamp = None
        self.shared = None  # d3d12.SharedFrames; False where it is unavailable
        self.held = None  # the frame being straightened, its decoder texture in use until done

    def update(self, frame, stamp, intrinsics, lens, deliver):
        """Straighten `frame` (an av.VideoFrame) if it is new (True): into texture, and into a
        colour frame of its own (colour, the straightened-th); deliver(image, stamp) gets that
        as the ToF mesh's colour frame, (h, w, 3) uint8 at worker.frame_size, at collect()."""
        if frame is None or intrinsics is None or stamp == self.stamp:
            return False
        self.stamp = stamp
        # One frame at a time: the decoder's texture and the shared planes are the last one's
        # until the GPU is done with them.
        self.collect(wait=True)
        if (frame.width, frame.height) != (intrinsics.width, intrinsics.height):
            raise RuntimeError(
                f"the robot's camera sends {frame.width}x{frame.height} frames but describes "
                f"{intrinsics.width}x{intrinsics.height}"
            )
        name = frame.format.name
        with trace.span("camera.decoded"):
            on_gpu = d3d12.decoded(frame) if self.shared is not False else None
        wait = ()
        if on_gpu is not None:
            try:
                if self.shared is None:
                    self.shared = d3d12.SharedFrames(self.device, on_gpu[0])
                with trace.span("camera.d3d12_copy"):
                    wait = [(self.shared.semaphore, ALL, self.shared.copy(on_gpu))]
            except (d3d12.Unavailable, OSError, VulkanError) as exc:
                log.warning("Camera frames go through memory: %s", exc)
                if self.shared:
                    self.shared.close()
                self.shared, on_gpu = False, None
                media.VideoDecoder.share = False
        if name == "d3d12" and on_gpu is None:
            frame, name = d3d12.download(frame), "nv12"
        elif on_gpu is None and name not in {"nv12", "yuv420p", "yuvj420p"}:
            frame, name = frame.reformat(format="nv12"), "nv12"
        commands = self.frames[self.count % FRAMES]
        self.count += 1
        with trace.span("camera.begin"):
            commands.begin()
        if on_gpu is not None:
            name = "nv12"
            self.planes = self.shared.planes
            self.shared.acquire(commands)
        else:
            self._upload(commands, frame, name)
        (width, height), step = frame_size(intrinsics.width, intrinsics.height)
        self._targets((intrinsics.width, intrinsics.height), (width, height))
        self.colour = self.colours[self.straightened % COLOURS]
        self.straightened += 1
        with trace.span("camera.uniforms"):
            block = self._uniforms(frame, name, intrinsics, lens)
        started = clock.now_ns()
        commands.sampled(*self.planes)
        for target, stride in (
            (self.texture, (1.0, 0.0)),
            (self.colour, (float(step), float(step // 2))),
        ):
            commands.transition(target, vk.VK_IMAGE_LAYOUT_GENERAL)
            uniforms = block + pack("2f", *stride) + pack("2i8x", *self.flags)
            resources = {0: commands.uniform(uniforms), 4: target}
            sampler = self.device.sampler()
            for binding, plane in zip((1, 2, 3), self.planes + self.planes[1:2], strict=False):
                resources[binding] = (plane, sampler)
            commands.use(self.pipeline, resources)
            commands.dispatch(target.size)

        trace.add("camera.passes", started, clock.now_ns())

        def read(pixels, size):
            image = np.ascontiguousarray(pixels.reshape(size[1], size[0], 4)[..., :3])
            deliver(image, stamp)

        commands.read(self.colour, read)
        commands.sampled(self.texture)
        with trace.span("camera.submit_gpu"):
            commands.submit(wait, ())
        self.pending, self.held = commands, frame
        return True

    def collect(self, wait=False):
        """Hand on the colour frame of the frame update() submitted once the GPU has it; with
        `wait`, wait for it. Without, a GPU busy with other work (a depth model, a headset's
        compositor) leaves it to the next call instead of holding the frame loop (up to 20 ms
        in VR, a missed frame)."""
        if self.pending is not None and (wait or self.pending.finished()):
            pending, self.pending = self.pending, None
            pending.wait()
            self.held = None

    def _upload(self, commands, frame, name):
        rows = (frame.height + 1) // 2
        if name == "nv12":
            layout = (
                (frame.planes[0].line_size, frame.height, 1),
                (frame.planes[1].line_size // 2, rows, 2),
            )
        else:
            layout = tuple(
                (plane.line_size, frame.height if i == 0 else rows, 1)
                for i, plane in enumerate(frame.planes)
            )
        if layout != self.layout:
            self.device.defer(*self.uploaded)
            self.uploaded = [
                Image(self.device, (width, height), PLANES[components], sampled=True)
                for width, height, components in layout
            ]
            self.layout = layout
        for image, plane in zip(self.uploaded, frame.planes, strict=True):
            commands.upload_image(image, memoryview(plane), (0, 0, *image.size))
        self.planes = self.uploaded

    def _targets(self, full, colour):
        if self.sizes == (full, colour):
            return
        self.device.defer(*(i for i in (self.texture, *self.colours) if i is not None))
        self.texture = Image(self.device, full, RGBA, sampled=True, storage=True)
        self.colours = [Image(self.device, colour, RGBA, storage=True) for _ in range(COLOURS)]
        self.sizes = (full, colour)

    def _uniforms(self, frame, name, intrinsics, lens):
        """straighten.comp's block up to its stride (the rest per pass)."""
        planar = len(self.planes) == 3
        (luma_w, luma_h), (chroma_w, chroma_h) = (p.size for p in self.planes[:2])
        k = np.zeros(12, "f4")
        if lens is not None:
            k[: len(lens.distortion)] = lens.distortion
        full = name == "yuvj420p" or int(frame.color_range) == AVCOL_RANGE_JPEG
        matrix, offset = ycc_to_rgb(int(frame.colorspace), full)
        to_rgb = np.eye(4)
        to_rgb[:3, :3] = matrix
        self.flags = (int(planar), int(lens is not None))
        return pack(
            "16f4f4f4f12f2f2f2f",
            mat(to_rgb),
            (*offset, 0.0),
            (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy),
            (lens.fx, lens.fy, lens.cx, lens.cy) if lens is not None else (1.0, 1.0, 0.0, 0.0),
            k,
            (frame.width, frame.height),
            (frame.width / luma_w, frame.height / luma_h),
            ((frame.width + 1) // 2 / chroma_w, (frame.height + 1) // 2 / chroma_h),
        )

    def close(self):
        self.collect(wait=True)
        for commands in self.frames:
            commands.close()
        if self.shared:
            self.shared.close()
        for image in (*self.uploaded, self.texture, *self.colours):
            if image is not None:
                image.close()
        self.pipeline.close()
