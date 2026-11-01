"""The robot camera's frames, straightened on the GPU.

Frames arrive as Ito decodes them, Y'CbCr in memory (NV12 from a GPU decoder, planar from the
CPU's), and through the robot's lens when its camera describes one (protocol.Lens). One shader
pass turns each into RGB at the camera's described pinhole: the flat camera view's texture,
and the ToF mesh's colour frame, strided as the worker takes it (worker.frame_size)."""

import moderngl
import numpy as np

from ito.tofmesh.worker import frame_size

# Luma weights (Kr, Kb) of ITU-R BT.601 and BT.709. FFmpeg's AVColorSpace 1 is BT.709; every
# other value, unspecified included, is converted as BT.601, as FFmpeg's own conversion does.
BT601, BT709 = (0.299, 0.114), (0.2126, 0.0722)
AVCOL_SPC_BT709, AVCOL_RANGE_JPEG = 1, 2

STRAIGHTEN = """#version 430
uniform sampler2D luma;
uniform sampler2D chroma;  // CbCr (NV12), or Cb with chroma_v Cr (planar)
uniform sampler2D chroma_v;
uniform bool planar;
uniform vec2 size;          // the frame, pixels
uniform vec2 luma_scale;    // frame width / texture width (a decoder's padded lines)
uniform vec2 chroma_scale;
uniform vec4 pinhole;       // fx fy cx cy that the frame is straightened to
uniform vec4 lens;          // fx fy cx cy of the lens the frame came through
uniform float k[12];        // OpenCV's k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4
uniform bool distorted;
uniform vec2 stride;        // output pixel i is straightened pixel i * stride.x + stride.y
uniform mat3 to_rgb;
uniform vec3 offset;
out vec4 color;
void main() {
    // Pixel centres are whole numbers in OpenCV's camera model, and output row 0 is the
    // frame's top row: rows are read back top first.
    vec2 p = (gl_FragCoord.xy - 0.5) * stride.x + stride.y;
    vec2 source = p;
    if (distorted) {
        vec2 xy = (p - pinhole.zw) / pinhole.xy;
        float r2 = dot(xy, xy), r4 = r2 * r2, r6 = r4 * r2;
        float radial = (1 + k[0] * r2 + k[1] * r4 + k[4] * r6)
                     / (1 + k[5] * r2 + k[6] * r4 + k[7] * r6);
        vec2 d = xy * radial + vec2(
            2 * k[2] * xy.x * xy.y + k[3] * (r2 + 2 * xy.x * xy.x) + k[8] * r2 + k[9] * r4,
            k[2] * (r2 + 2 * xy.y * xy.y) + 2 * k[3] * xy.x * xy.y + k[10] * r2 + k[11] * r4);
        source = d * lens.xy + lens.zw;
    }
    vec2 uv = (source + 0.5) / size;
    vec2 c = planar
        ? vec2(texture(chroma, uv * chroma_scale).r, texture(chroma_v, uv * chroma_scale).r)
        : texture(chroma, uv * chroma_scale).rg;
    vec3 ycc = vec3(texture(luma, uv * luma_scale).r, c);
    color = vec4(clamp(to_rgb * (ycc - offset), 0.0, 1.0), 1);
}
"""

FULLSCREEN = """#version 430
const vec2 corners[4] = vec2[4](vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1));
void main() { gl_Position = vec4(corners[gl_VertexID], 0, 1); }
"""


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


class CameraFeed:
    def __init__(self, context):
        self.context = context
        self.program = context.program(vertex_shader=FULLSCREEN, fragment_shader=STRAIGHTEN)
        self.vao = context.vertex_array(self.program, [])
        self.planes = []
        self.layout = None
        self.texture = None  # the latest frame, straightened, full size
        self.framebuffer = None
        self.colour = None  # the same, strided to the ToF mesh's colour size
        self.colour_texture = None
        self.sizes = None
        self.stamp = None

    def update(self, frame, stamp, intrinsics, lens):
        """Straighten `frame` (an av.VideoFrame) if it is new; returns it as the ToF mesh's
        colour frame, (h, w, 3) uint8 at worker.frame_size, or None when nothing is new."""
        if frame is None or intrinsics is None or stamp == self.stamp:
            return None
        self.stamp = stamp
        if (frame.width, frame.height) != (intrinsics.width, intrinsics.height):
            raise RuntimeError(
                f"the robot's camera sends {frame.width}x{frame.height} frames but describes "
                f"{intrinsics.width}x{intrinsics.height}"
            )
        name = frame.format.name
        if name not in {"nv12", "yuv420p", "yuvj420p"}:
            frame, name = frame.reformat(format="nv12"), "nv12"
        self._upload(frame, name)
        (width, height), step = frame_size(intrinsics.width, intrinsics.height)
        self._targets((intrinsics.width, intrinsics.height), (width, height))
        self._uniforms(frame, name, intrinsics, lens)
        self.context.enable_only(moderngl.NOTHING)
        self._draw(self.framebuffer, (1.0, 0.0))
        self._draw(self.colour, (float(step), float(step // 2)))
        pixels = self.colour.read(components=3, alignment=1)
        return np.frombuffer(pixels, np.uint8).reshape(height, width, 3)

    def _upload(self, frame, name):
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
            for texture in self.planes:
                texture.release()
            self.planes = []
            for width, height, components in layout:
                texture = self.context.texture((width, height), components, alignment=1)
                texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                texture.repeat_x = texture.repeat_y = False
                self.planes.append(texture)
            self.layout = layout
        for texture, plane in zip(self.planes, frame.planes, strict=True):
            texture.write(plane, alignment=1)

    def _targets(self, full, colour):
        if self.sizes == (full, colour):
            return
        self._release_targets()
        self.texture = self.context.texture(full, 3, alignment=1)
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.framebuffer = self.context.framebuffer([self.texture])
        self.colour_texture = self.context.texture(colour, 3, alignment=1)
        self.colour = self.context.framebuffer([self.colour_texture])
        self.sizes = (full, colour)

    def _uniforms(self, frame, name, intrinsics, lens):
        program = self.program
        planar = len(self.planes) == 3
        program["planar"] = planar
        program["luma"], program["chroma"] = 0, 1
        program["chroma_v"] = 2 if planar else 1
        program["size"] = (frame.width, frame.height)
        program["luma_scale"] = (frame.width / self.layout[0][0], 1.0)
        program["chroma_scale"] = ((frame.width + 1) // 2 / self.layout[1][0], 1.0)
        program["pinhole"] = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
        program["distorted"] = lens is not None
        if lens is not None:
            program["lens"] = (lens.fx, lens.fy, lens.cx, lens.cy)
            k = np.zeros(12, "f4")
            k[: len(lens.distortion)] = lens.distortion
            program["k"].write(k.tobytes())
        full = name == "yuvj420p" or int(frame.color_range) == AVCOL_RANGE_JPEG
        matrix, offset = ycc_to_rgb(int(frame.colorspace), full)
        program["to_rgb"].write(np.asarray(matrix, "f4").T.copy().tobytes())
        program["offset"] = offset

    def _draw(self, framebuffer, stride):
        framebuffer.use()
        self.context.viewport = (0, 0, *framebuffer.size)
        for unit, texture in enumerate(self.planes):
            texture.use(unit)
        self.program["stride"] = stride
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

    def _release_targets(self):
        for resource in (self.framebuffer, self.texture, self.colour, self.colour_texture):
            if resource is not None:
                resource.release()
        self.framebuffer = self.texture = self.colour = self.colour_texture = None

    def close(self):
        self._release_targets()
        for texture in self.planes:
            texture.release()
        self.vao.release()
        self.program.release()
