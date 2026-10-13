"""Original RGB views with disposable depth proxies and binocular source ownership.

Selection is made at the eye midpoint; each eye still tests its own visibility.
Two observed depth layers preserve background without fusing appearance or surfaces.
RGB can arrive before depth, but unsupported RGB never textures unrelated geometry.
"""

from dataclasses import dataclass

import cv2
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
out vec3 source_uv_depth;
out vec3 world_point;
uniform mat4 view_projection;
uniform vec2 jump;
void main() {
    float lo=min(vertex[0].z,min(vertex[1].z,vertex[2].z));
    float hi=max(vertex[0].z,max(vertex[1].z,vertex[2].z));
    if (lo<0.05 || hi>40 || isnan(hi) || hi-lo>jump.x+jump.y*lo) return;
    for (int k=0;k<3;k++) {
        source_uv_depth=vec3(vertex[k].uv*vertex[k].z,vertex[k].z);
        world_point=vertex[k].world;
        gl_Position=view_projection*vec4(world_point,1);
        EmitVertex();
    }
    EndPrimitive();
}
"""
FRAGMENT = """
#version 430
in vec3 source_uv_depth;
in vec3 world_point;
uniform sampler2D colors, sharpness, nearest_layer, front_owner, back_owner, history;
uniform mat4 midpoint_view, midpoint_projection, previous_vp;
uniform vec3 source_center;
uniform float source_id, base_cost, hysteresis;
uniform int pass_kind, layer;
out vec4 color;
void main() {
    vec2 uv=source_uv_depth.xy/source_uv_depth.z;
    vec4 local=midpoint_view*vec4(world_point,1);
    float z=-local.z;
    vec4 projected=midpoint_projection*local;
    vec2 q=projected.xy/projected.w*0.5+0.5;
    vec4 front=texture(front_owner,q);
    float tol=0.02+0.015*z;
    if (pass_kind==0) {
        if (layer==1 && (front.a==0 || z<=front.r+tol)) discard;
        color=vec4(z,source_id,0,1);
    } else if (pass_kind==1) {
        vec4 nearest=texture(nearest_layer,q);
        if (nearest.a==0 || abs(z-nearest.r)>tol) discard;
        float border=min(min(uv.x,1-uv.x),min(uv.y,1-uv.y));
        float cost=base_cost+0.04/(0.03+border)
                    +0.18/(0.03+texture(sharpness,uv).r)
                    +0.4*length(world_point-source_center)/max(z,0.1);
        vec4 p=previous_vp*vec4(world_point,1);
        vec2 prev=p.xy/p.w*0.5+0.5;
        if (p.w>0 && all(greaterThanEqual(prev,vec2(0)))
                   && all(lessThanEqual(prev,vec2(1)))) {
            vec4 old=texture(history,prev);
            if (old.a>0 && abs(old.g-source_id)<0.1) cost-=hysteresis;
        }
        cost=max(cost,0.001);
        gl_FragDepth=cost/(1+cost);
        color=vec4(z,source_id,cost,1);
    } else {
        // Ownership is shared, visibility/depth testing is independent per eye.
        bool inside=projected.w>0 && all(greaterThanEqual(q,vec2(0)))
                                   && all(lessThanEqual(q,vec2(1)));
        if (inside) {
            vec4 back=texture(back_owner,q);
            if (front.a>0 && abs(z-front.r)<=tol) {
                if (abs(front.g-source_id)>0.1) discard;
            } else if (back.a>0 && abs(z-back.r)<=tol) {
                if (abs(back.g-source_id)>0.1) discard;
            } else if (front.a>0) discard;
        }
        color=vec4(texture(colors,uv).rgb,1);
        gl_FragDepth=gl_FragCoord.z;
    }
}
"""


@dataclass
class Source:
    frame: int
    captured: float
    pose: np.ndarray
    color: object
    sharpness: object
    quality: float
    depth: object = None

    def close(self):
        for texture in (self.color, self.sharpness, self.depth):
            if texture is not None:
                texture.release()


class ViewCache:
    def __init__(
        self,
        context,
        intrinsics,
        *,
        cap=64,
        candidates=12,
        grid=(90, 160),
        jump=(0.02, 0.08),
        hysteresis=0.15,
    ):
        if cap < 2 or candidates < 2 or min(grid) < 2:
            raise ValueError("Cache/candidates need at least two entries; grid at least 2x2")
        self.ctx, self.i, self.cap = context, intrinsics, cap
        self.candidates, self.grid, self.jump = candidates, grid, jump
        self.hysteresis = hysteresis
        self.sources, self.selected = [], []
        self.last_frame, self.last_capture = -1, -np.inf
        self.previous_vp = np.zeros((4, 4), "f4")
        self.program = context.program(
            vertex_shader=VERTEX, geometry_shader=GEOMETRY, fragment_shader=FRAGMENT
        )
        gx, gy = grid
        x, y = np.meshgrid(np.linspace(0, 1, gx), np.linspace(0, 1, gy))
        self.vertices = context.buffer(np.stack((x, y), -1).astype("f4").tobytes())
        a = (np.arange(gy - 1)[:, None] * gx + np.arange(gx - 1)).ravel()
        self.indices = context.buffer(
            np.stack((a, a + gx, a + 1, a + 1, a + gx, a + gx + 1), -1).astype("u4").tobytes()
        )
        self.vao = context.vertex_array(
            self.program, [(self.vertices, "2f", "grid_uv")], self.indices
        )
        self.program["calibration"].value = (
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
        )
        self.program["image_size"].value = (intrinsics.width, intrinsics.height)
        self.program["jump"].value = jump
        self.program["hysteresis"].value = hysteresis
        for unit, name in enumerate(
            (
                "depths",
                "colors",
                "sharpness",
                "nearest_layer",
                "front_owner",
                "back_owner",
                "history",
            )
        ):
            self.program[name].value = unit
        self.layers = []
        for _ in range(5):
            tex = context.texture((intrinsics.width, intrinsics.height), 4, dtype="f4")
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            tex.repeat_x = tex.repeat_y = False
            depth = context.depth_renderbuffer(tex.size)
            fbo = context.framebuffer([tex], depth)
            fbo.clear(0, 0, 0, 0, depth=1)
            self.layers.append((tex, depth, fbo))

    @staticmethod
    def rigid(pose):
        pose = np.asarray(pose, dtype="f4")
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], (0, 0, 0, 1), atol=1e-5)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3)
            or np.linalg.det(pose[:3, :3]) < 0.99
        ):
            raise ValueError("Camera pose must be a finite rigid transform")
        return pose

    def publish_rgb(self, rgb, pose, *, frame, captured):
        i = self.i
        if rgb.shape != (i.height, i.width, 3) or rgb.dtype != np.uint8:
            raise ValueError("RGB differs from calibration")
        pose = self.rigid(pose)
        if frame <= self.last_frame or captured < self.last_capture or not np.isfinite(captured):
            raise ValueError("RGB must arrive in capture order with finite timestamps")
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype("f4") / 255
        sharp = cv2.GaussianBlur(np.abs(cv2.Laplacian(grey, cv2.CV_32F)), (0, 0), 8)
        # No normalization by this image: blur should reduce its absolute score.
        sharp = np.clip(sharp * 20, 0, 1).astype("f4")
        color = self.ctx.texture((i.width, i.height), 3, np.ascontiguousarray(rgb).tobytes())
        color.build_mipmaps()
        color.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        quality = self.ctx.texture(color.size, 1, sharp.tobytes(), dtype="f4")
        quality.filter = (moderngl.LINEAR, moderngl.LINEAR)
        for texture in (color, quality):
            texture.repeat_x = texture.repeat_y = False
        self.sources.append(
            Source(frame, float(captured), pose.copy(), color, quality, float(sharp.mean()))
        )
        self.last_frame, self.last_capture = frame, captured
        if len(self.sources) > self.cap:
            # Drop a redundant viewpoint, preserving isolated angular/spatial coverage.
            old = self.sources[:-2]
            costs = []
            for s in old:
                neighbours = [
                    np.linalg.norm(s.pose[:3, 3] - t.pose[:3, 3]) * 4
                    + np.linalg.norm(s.pose[:3, 2] - t.pose[:3, 2])
                    for t in self.sources
                    if t is not s
                ]
                costs.append(min(neighbours) * (0.2 + s.quality))
            victim = old[int(np.argmin(costs))]
            self.sources.remove(victim)
            victim.close()
        return True

    def publish_depth(self, depth, *, frame):
        source = next((s for s in self.sources if s.frame == frame), None)
        if source is None:
            return False  # Late completion for an evicted RGB never resurrects a slot.
        if source.depth is not None:
            return False
        depth = np.asarray(depth, dtype="f4")
        if depth.shape != (self.i.height, self.i.width):
            raise ValueError("Depth differs from calibration")
        depth = np.where(np.isfinite(depth) & (depth >= 0.05) & (depth <= 40), depth, 0)
        if not np.any(depth):
            return False
        source.depth = self.ctx.texture(source.color.size, 1, depth.tobytes(), dtype="f4")
        source.depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
        source.depth.repeat_x = source.depth.repeat_y = False
        return True

    def add(self, rgb, depth, pose, *, frame, captured):
        self.publish_rgb(rgb, pose, frame=frame, captured=captured)
        return self.publish_depth(depth, frame=frame)

    def raster(self, camera, projection, target, kind, layer=0):
        target.use()
        target.clear(0, 0, 0, 0, depth=1)
        self.ctx.enable_only(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<="
        p = self.program
        p["view_projection"].write((projection @ np.linalg.inv(camera)).T.astype("f4").tobytes())
        p["pass_kind"].value, p["layer"].value = kind, layer
        for source, cost in self.selected:
            source.depth.use(0)
            source.color.use(1)
            source.sharpness.use(2)
            p["source_pose"].write(source.pose.T.tobytes())
            p["source_center"].value = tuple(source.pose[:3, 3])
            p["source_id"].value = float(source.frame + 1)
            p["base_cost"].value = cost
            self.vao.render(moderngl.TRIANGLES)

    def prepare(self, midpoint, projection):
        midpoint = self.rigid(midpoint)
        p = self.program
        ranked = []
        for s in self.sources:
            if s.depth is None:
                continue
            angle = float(1 - np.dot(s.pose[:3, 2], midpoint[:3, 2]))
            distance = float(np.linalg.norm(s.pose[:3, 3] - midpoint[:3, 3]))
            cost = 1.5 * angle + distance * 1.5 + 0.005 * min(40, self.last_capture - s.captured)
            ranked.append((s, cost))
        self.selected = sorted(ranked, key=lambda v: v[1])[: self.candidates]
        p["midpoint_view"].write(np.linalg.inv(midpoint).T.astype("f4").tobytes())
        p["midpoint_projection"].write(np.asarray(projection, "f4").T.tobytes())
        p["previous_vp"].write(self.previous_vp.T.tobytes())
        near, front, behind, back, history = self.layers
        # Bind distinct attachments for every pass: never sample the draw attachment.
        history[0].use(4)
        history[0].use(5)
        history[0].use(6)
        history[0].use(3)
        self.raster(midpoint, projection, near[2], 0)
        near[0].use(3)
        self.raster(midpoint, projection, front[2], 1)
        front[0].use(4)
        self.raster(midpoint, projection, behind[2], 0, 1)
        behind[0].use(3)
        self.raster(midpoint, projection, back[2], 1, 1)
        back[0].use(5)
        self.ctx.copy_framebuffer(history[2], front[2])
        self.previous_vp = np.asarray(projection @ np.linalg.inv(midpoint), "f4")

    def draw(self, camera, projection, target):
        self.raster(camera, projection, target, 2)

    def save(self, path):
        sources = [s for s in self.sources if s.depth is not None]
        if not sources:
            raise ValueError("Cannot save an empty cache")
        i = self.i
        np.savez_compressed(
            path,
            grid=self.grid,
            jump=self.jump,
            cap=self.cap,
            candidates=self.candidates,
            hysteresis=self.hysteresis,
            calibration=[i.width, i.height, i.fx, i.fy, i.cx, i.cy],
            frames=[s.frame for s in sources],
            captured=[s.captured for s in sources],
            poses=[s.pose for s in sources],
            rgb=[
                np.frombuffer(s.color.read(), "u1").reshape(i.height, i.width, 3) for s in sources
            ],
            depth=[np.frombuffer(s.depth.read(), "f4").reshape(i.height, i.width) for s in sources],
        )

    @classmethod
    def load(cls, context, path):
        from ito.protocol import Intrinsics

        with np.load(path, allow_pickle=False) as data:
            w, h, fx, fy, cx, cy = data["calibration"]
            cache = cls(
                context,
                Intrinsics(width=int(w), height=int(h), fx=fx, fy=fy, cx=cx, cy=cy),
                grid=tuple(data["grid"]),
                jump=tuple(data["jump"]),
                cap=int(data["cap"]),
                candidates=int(data["candidates"]),
                hysteresis=float(data["hysteresis"]),
            )
            try:
                for frame, captured, pose, rgb, depth in zip(
                    data["frames"],
                    data["captured"],
                    data["poses"],
                    data["rgb"],
                    data["depth"],
                    strict=True,
                ):
                    cache.add(rgb, depth, pose, frame=int(frame), captured=float(captured))
            except Exception:
                cache.close()
                raise
            return cache

    def close(self):
        for source in self.sources:
            source.close()
        self.sources.clear()
        for tex, depth, fbo in self.layers:
            fbo.release()
            depth.release()
            tex.release()
        for resource in (self.vao, self.indices, self.vertices, self.program):
            resource.release()
