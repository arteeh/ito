"""Depth is axial metres, poses are world-from-camera in Ito's OpenGL axes.

A source's full-resolution retirement mask survives newer-patch eviction, preventing
contradicted surfaces from returning. Geometry and retirement execute on the GPU;
CPU readback is reserved for the offline evaluator and saved-world viewer.
"""

from collections import deque
from dataclasses import dataclass

import moderngl
import numpy as np

VERTEX = """
#version 430
in vec2 grid_uv;
uniform sampler2D depths;
uniform vec4 calibration;
uniform vec2 image_size;
uniform mat4 source_pose;
out VS { vec2 uv; vec3 world; float z; } vertex;
void main() {
    float z = texture(depths, grid_uv).r;
    vec2 pixel = grid_uv * image_size;
    vec3 local = vec3((pixel.x-calibration.z)/calibration.x,
                      (calibration.w-pixel.y)/calibration.y, -1) * z;
    vertex.uv = grid_uv;
    vertex.world = (source_pose * vec4(local, 1)).xyz;
    vertex.z = z;
    gl_Position = vec4(vertex.world, 1);
}
"""
GEOMETRY = """
#version 430
layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;
in VS { vec2 uv; vec3 world; float z; } vertex[];
out vec2 uv;
uniform mat4 view_projection;
uniform vec2 jump;
void main() {
    float lo = min(vertex[0].z, min(vertex[1].z, vertex[2].z));
    float hi = max(vertex[0].z, max(vertex[1].z, vertex[2].z));
    if (lo < 0.05 || hi > 40 || isnan(hi) || hi-lo > jump.x+jump.y*lo) return;
    for (int k=0; k<3; k++) {
        uv = vertex[k].uv;
        gl_Position = view_projection * vec4(vertex[k].world, 1);
        EmitVertex();
    }
    EndPrimitive();
}
"""
FRAGMENT = """
#version 430
in vec2 uv;
uniform sampler2D colors;
uniform usampler2D alive;
out vec4 color;
void main() {
    if (texture(alive, uv).r == 0u) discard;
    color = vec4(texture(colors, uv).rgb, 1);
}
"""
RETIRE = """
#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(r8ui, binding=0) uniform uimage2D alive;
uniform sampler2D old_depth;
uniform sampler2D new_depth;
uniform mat4 old_to_new;
uniform vec4 calibration;
uniform vec2 image_size;
uniform vec2 tolerance;
uniform vec2 jump;
void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pixel, ivec2(image_size)))) return;
    if (imageLoad(alive, pixel).r == 0u) return;
    float z = texelFetch(old_depth, pixel, 0).r;
    vec3 p = vec3((float(pixel.x)+0.5-calibration.z)/calibration.x,
                 (calibration.w-float(pixel.y)-0.5)/calibration.y, -1)*z;
    p = (old_to_new * vec4(p, 1)).xyz;
    float range = -p.z;
    if (range < 0.05) return;
    vec2 target = vec2(calibration.x*p.x/range+calibration.z,
                       calibration.w-calibration.y*p.y/range);
    if (any(lessThan(target, vec2(0))) || any(greaterThanEqual(target, image_size))) return;
    ivec2 q = ivec2(target);
    float fresh = texelFetch(new_depth, q, 0).r;
    // Do not carve with unsupported triangles at a new depth discontinuity.
    float lo=fresh, hi=fresh;
    for (int dy=-1; dy<=1; dy++) for (int dx=-1; dx<=1; dx++) {
        float d=texelFetch(new_depth, clamp(q+ivec2(dx,dy)*4, ivec2(0),
                                          ivec2(image_size)-1), 0).r;
        lo=min(lo,d); hi=max(hi,d);
    }
    if (lo < 0.05 || hi > 40 || isnan(hi) || hi-lo > jump.x+jump.y*lo) return;
    if (range <= fresh+tolerance.x+tolerance.y*fresh)
        imageStore(alive, pixel, uvec4(0));
}
"""


@dataclass
class Patch:
    frame: int
    captured: float
    pose: np.ndarray
    color: object
    depth: object
    active: object

    def close(self):
        for texture in (self.color, self.depth, self.active):
            texture.release()


class MeshWorld:
    def __init__(
        self,
        context,
        intrinsics,
        *,
        cap=64,
        grid=(90, 160),
        jump=(0.02, 0.08),
        tolerance=(0.02, 0.03),
        carve=True,
    ):
        if cap < 1 or min(grid) < 2:
            raise ValueError("Patch cap must be positive and grid dimensions at least two")
        self.ctx, self.i, self.cap = context, intrinsics, cap
        self.jump, self.tolerance, self.carve = jump, tolerance, carve
        self.patches = deque()
        self.program = context.program(
            vertex_shader=VERTEX, geometry_shader=GEOMETRY, fragment_shader=FRAGMENT
        )
        self.retire = context.compute_shader(RETIRE)
        gx, gy = grid
        x, y = np.meshgrid(np.linspace(0, 1, gx), np.linspace(0, 1, gy))
        self.vertices = context.buffer(np.stack((x, y), -1).astype("f4").tobytes())
        a = (np.arange(gy - 1)[:, None] * gx + np.arange(gx - 1)).ravel()
        triangles = np.stack((a, a + gx, a + 1, a + 1, a + gx, a + gx + 1), -1).astype("u4")
        self.indices = context.buffer(triangles.tobytes())
        self.vao = context.vertex_array(
            self.program, [(self.vertices, "2f", "grid_uv")], self.indices
        )
        calibration = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
        for program in (self.program, self.retire):
            program["calibration"].value = calibration
            program["image_size"].value = (intrinsics.width, intrinsics.height)
            program["jump"].value = jump
        for name, unit in (("depths", 0), ("colors", 1), ("alive", 2)):
            self.program[name].value = unit
        self.retire["old_depth"].value, self.retire["new_depth"].value = 0, 1
        self.retire["tolerance"].value = tolerance

    def add(self, rgb, depth, pose, *, frame, captured):
        i, ctx = self.i, self.ctx
        if rgb.shape != (i.height, i.width, 3) or depth.shape != rgb.shape[:2]:
            raise ValueError("RGB/depth shape differs from camera calibration")
        pose = np.asarray(pose, dtype="f4")
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], (0, 0, 0, 1), atol=1e-5)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3)
        ):
            raise ValueError("Camera pose must be a finite rigid transform")
        if self.patches and (
            frame <= self.patches[-1].frame or captured < self.patches[-1].captured
        ):
            raise ValueError("Patches must arrive in capture order")
        depth = np.asarray(depth, dtype="f4")
        depth = np.where(np.isfinite(depth) & (depth >= 0.05) & (depth <= 40), depth, 0)
        if not np.any(depth):
            return False
        if len(self.patches) == self.cap:
            self.patches.popleft().close()
        color = ctx.texture((i.width, i.height), 3, np.ascontiguousarray(rgb, "u1").tobytes())
        color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        d = ctx.texture((i.width, i.height), 1, depth.tobytes(), dtype="f4")
        active = ctx.texture(
            (i.width, i.height), 1, np.ones(depth.shape, "u1").tobytes(), dtype="u1"
        )
        for texture in (color, d, active):
            texture.repeat_x = texture.repeat_y = False
        d.filter = active.filter = (moderngl.NEAREST, moderngl.NEAREST)
        if self.carve:
            d.use(1)
            inverse = np.linalg.inv(pose)
            for old in self.patches:
                old.depth.use(0)
                old.active.bind_to_image(0, read=True, write=True)
                self.retire["old_to_new"].write((inverse @ old.pose).T.astype("f4").tobytes())
                self.retire.run((i.width + 15) // 16, (i.height + 15) // 16)
            ctx.memory_barrier()
        self.patches.append(Patch(frame, float(captured), pose.copy(), color, d, active))
        return True

    def draw(self, camera, projection, target):
        target.use()
        target.clear(0, 0, 0, 0, depth=1)
        self.ctx.enable_only(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<="
        self.program["view_projection"].write(
            (projection @ np.linalg.inv(camera)).T.astype("f4").tobytes()
        )
        for patch in self.patches:
            patch.depth.use(0)
            patch.color.use(1)
            patch.active.use(2)
            self.program["source_pose"].write(patch.pose.T.tobytes())
            self.vao.render(moderngl.TRIANGLES)

    def save(self, path):
        if not self.patches:
            raise ValueError("Cannot save an empty world")
        i = self.i
        np.savez_compressed(
            path,
            calibration=np.array([i.width, i.height, i.fx, i.fy, i.cx, i.cy]),
            frames=[p.frame for p in self.patches],
            captured=[p.captured for p in self.patches],
            poses=[p.pose for p in self.patches],
            rgb=np.stack(
                [
                    np.frombuffer(p.color.read(), "u1").reshape(i.height, i.width, 3)
                    for p in self.patches
                ]
            ),
            depth=np.stack(
                [
                    np.frombuffer(p.depth.read(), "f4").reshape(i.height, i.width)
                    for p in self.patches
                ]
            ),
            active=np.stack(
                [
                    np.frombuffer(p.active.read(), "u1").reshape(i.height, i.width)
                    for p in self.patches
                ]
            ),
        )

    def close(self):
        while self.patches:
            self.patches.popleft().close()
        for resource in (self.vao, self.indices, self.vertices, self.retire, self.program):
            resource.release()
