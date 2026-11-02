"""Draws the ToF mesh, textured by projecting camera frames onto it.

The mesh carries a colour per vertex (the volume's voxel colours, a few centimetres coarse:
the map's memory). On top, a handful of camera frames act as slide projectors: every fragment
looks itself up in each projector that sees it unoccluded (a distance map per projector,
rendered from the mesh) and takes the frames' colours, weighted towards the projector looking at
it most nearly along the eye's ray, so detail comes at camera resolution where a frame saw it.
Slot 0 is the live frame; the others are kept views (fusion.py chooses them).

One scene per displayed frame, however many views draw it (a headset's two eyes): prepare() runs
once per frame (distance maps, atlas bake, guess pyramid, the fill's depth map) and every view
chooses its colours from one viewer, the point between the pilot's eyes. Each eye only sees that
scene from its own position; were the projector choice made per eye, the eyes, a few centimetres
apart, would pick different frames for the same surface and see different textures on it.

Guessed depth (#87): the ToF sees ~45 degrees, the camera far more, so most of the live frame
has no mesh behind it. A guess places it in the world anyway, on the GPU, every time the live
frame or the mesh changes: the live projector's distance map (the mesh, i.e. everything the ToF
ever measured, seen from the live camera) has its holes filled by push-pull (an image pyramid
that averages inverse range down, then fills each hole from the coarser level up: the nearest
measured surfaces extended outwards), and a ray that meets the floor (y = 0 in the robot's
odometry frame) before that is put on the floor. The live frame is then drawn as a grid at
those ranges, in the frame's own colours like any measured surface (user, 1 Nov: no dimming or
hatching), and broken where neighbouring guesses jump (an edge). Guesses are drawn into a layer
of their own with ordinary depth (so nearer guesses hide farther ones from any eye position),
and that layer is composited only into pixels where no mesh was drawn: a guess never covers a
measured surface and gives way as soon as the ToF measures that spot.

Texture memory: the kept views are few (PROJECTORS) and the oldest is replaced, so a slow 360
degree turn outlived them and everything seen more than about half a turn ago fell back to the
vertex colours: 4 cm voxels averaged over many frames, a smear (the floor's planks gone). So
every live frame is also baked into a texture atlas at about the camera's resolution: every
block owns six square tiles, one per axis direction (+X, -X, +Y, ...), and a triangle lives in
the tiles of the directions its face points along (a planar projection of the block onto that
axis, fixed in the world, so re-meshing a block keeps its texture in place). A texel keeps the
frame that saw it best (facing it most squarely, nearest and nearest the image's middle) and
is replaced only by a view about as good (BAKE_KEEP): seen
areas stay as sharp as their best view, and a view as good refreshes them (a moved object). The
atlas joins the projector blend with the weight of the best view it had. Tiles of a block the
volume forgot are cleared when its slot is given to a new block (MeshFrame.cleared).
"""

import logging

import numpy as np
import vulkan as vk

from ito.render.pose import validate_pose
from ito.render.vk import DEPTH_FORMAT, Buffer, Image, Pipeline, mat, pack

from .fusion import PROJECTORS

log = logging.getLogger(__name__)

SHADOW_SIZE = (256, 144)
NEAR, FAR = 0.05, 30.0

# Texture atlas: a tile is TILE texels square and spans TILE_SPAN voxels of its block, from one
# voxel before the block's corner (a block's vertices reach half a voxel past its faces): 6.25 mm
# texels at 4 cm voxels, the camera frame's own resolution (640 wide) about 2 m away.
TILE = 64
TILE_SPAN = 10.0
DIRECTIONS = 6
LAYER = 4096  # an atlas layer is LAYER texels square: (LAYER / TILE)^2 tiles
LAYER_TILES = (LAYER // TILE) ** 2
# A triangle is baked into the tile of every direction its face is within ~78 degrees of, and
# drawn from those within ~75 (so a drawn tile was always baked).
BAKE_FACING = 0.2
DRAW_FACING = 0.25
# A texel takes a new view's colour when that view's weight is at least this share of the best
# weight a view of it ever had.
BAKE_KEEP = 0.85

# Guessed depth: one vertex per GUESS_GRID pixels of the shadow-map resolution, ranges capped at
# GUESS_FAR (beyond the ToF's 4 m nothing is known), a break where neighbouring guesses differ
# by more than GUESS_EDGE.
GUESS_GRID = (128, 72)
GUESS_FAR = 8.0
GUESS_EDGE = 1.3
FLOOR_HEIGHT = 0.0  # the odometry frame's floor
# The floor rule needs the camera above it: odometry that drifted the camera down to the floor
# would put every downward ray on the floor a few centimetres away.
FLOOR_CLEARANCE = 0.05

# Guessed depth beyond the live frame (the view is wider and taller than the camera, and leads it
# while turning): every pixel the mesh and the live guess leave empty gets a range from the
# viewer's depth map, its holes filled from the map's mipmaps (the nearest level that has
# something there: one mipmap build instead of a push-pull's twenty passes), the floor rule, and
# the colour of the kept view that saw that spot from the most similar direction. A spot no frame
# ever saw stays empty. The map is the viewer's, drawn once per frame over every view's field and
# its holes filled once (resolve.comp), so the filled surface lies in the world: each eye marches
# its own ray to the first point behind it, and both eyes see one surface.
FILL_DEPTH_WIDTH = 256
FILL_STEPS = 24  # ray march steps from 0.1 m to GUESS_FAR, each ~20 % farther, then bisection

# The shaders' specialization constants (common.glsl).
CONSTANTS = {
    0: PROJECTORS, 1: TILE, 2: LAYER // TILE, 3: TILE_SPAN, 4: DRAW_FACING, 5: BAKE_FACING,
    6: BAKE_KEEP, 7: LAYER, 8: FILL_STEPS,
}  # fmt: skip

COLOUR = vk.VK_FORMAT_R8G8B8A8_UNORM  # every view's image (and the guess layer)
RANGE, INVERSE, MAP = vk.VK_FORMAT_R32_SFLOAT, vk.VK_FORMAT_R32G32_SFLOAT, vk.VK_FORMAT_R32_SFLOAT
GENERAL = vk.VK_IMAGE_LAYOUT_GENERAL
CLEAR = (0.025, 0.035, 0.055, 1.0)
EMPTY = 1e9  # a distance map where nothing is
# Vertex inputs: position, normal, colour, block, one buffer each (bindings 0-3).
MESH_INPUTS = ((0, 0, "3f"), (1, 1, "3f"), (2, 2, "4u8"), (3, 3, "4f"))


def _pyramid(device, size):
    """Push-pull levels from `size` down to 1x1: (size, pushed image, pulled image)."""
    levels = []
    w, h = size
    while True:
        levels.append(
            ((w, h), *(Image(device, (w, h), INVERSE, sampled=True, storage=True) for _ in "pp"))
        )
        if w == 1 and h == 1:
            return levels
        w, h = (w + 1) // 2, (h + 1) // 2


def _view(world_from_eye):
    rotation = world_from_eye[:3, :3]
    view = np.eye(4)
    view[:3, :3] = rotation.T
    view[:3, 3] = -rotation.T @ world_from_eye[:3, 3]
    return view


def pinhole_projection(fx, fy, cx, cy, width, height, near=NEAR, far=FAR):
    """OpenGL clip from camera (Ito axes) for an image whose row 0 is the top."""
    p = np.zeros((4, 4))
    p[0, 0] = 2 * fx / width
    p[0, 2] = -((2 * cx + 1) / width - 1)
    p[1, 1] = 2 * fy / height
    p[1, 2] = -(1 - (2 * cy + 1) / height)
    p[2, 2] = -(far + near) / (far - near)
    p[2, 3] = -2 * far * near / (far - near)
    p[3, 2] = -1
    return p


class MeshRenderer:
    """The mesh, its projectors and atlas on the GPU. Its work goes into the device's frame
    being recorded (vk.Device.frame): upload() and prepare() once per displayed frame, draw()
    once per view."""

    def __init__(self, device, slots=PROJECTORS, image_size=(640, 360)):
        if slots != PROJECTORS:
            raise ValueError("The shaders are built for fusion.PROJECTORS projector slots")
        self.device = device
        self.slots = slots
        self.image_size = tuple(image_size)
        mesh = dict(vertex=MESH_INPUTS, constants=CONSTANTS)
        self.program = Pipeline(
            device, ("mesh.vert", "mesh.frag"),
            {0: "uniform", 1: "storage", 2: "sampler", 3: "sampler", 4: "sampler"},
            colours=(COLOUR,), depth="<", **mesh,
        )  # fmt: skip
        self.bake_program = Pipeline(
            device, ("bake.vert", "bake.geom", "bake.frag"),
            {0: "uniform", 1: "image", 2: "sampler", 3: "sampler"}, **mesh,
        )  # fmt: skip
        self.shadow_program = Pipeline(
            device, ("shadow.vert", "shadow.frag"), {}, push=80, vertex=MESH_INPUTS[:1],
            colours=(RANGE,), depth="<",
        )  # fmt: skip
        self.fill_depth_program = Pipeline(
            device, ("shadow.vert", "fill_depth.frag"), {}, push=80, vertex=MESH_INPUTS[:1],
            colours=(INVERSE,), depth="<",
        )  # fmt: skip
        self.push_program = Pipeline(device, ("push.comp",), {0: "sampler", 1: "image"}, push=36)
        self.pull_program = Pipeline(
            device, ("pull.comp",), {0: "sampler", 1: "sampler", 2: "image"}, push=8
        )
        self.downsample_program = Pipeline(device, ("downsample.comp",), {0: "image", 1: "image"})
        self.resolve_program = Pipeline(device, ("resolve.comp",), {0: "sampler", 1: "image"})
        self.guess_program = Pipeline(
            device, ("guess.vert", "guess.frag"),
            {0: "uniform", 1: "sampler", 2: "sampler", 3: "sampler"},
            vertex=((0, 0, "2f"),), colours=(COLOUR,), depth="<", constants=CONSTANTS,
        )  # fmt: skip
        self.composite_program = Pipeline(
            device, ("composite.vert", "composite.frag"), {0: "sampler"}, colours=(COLOUR,),
            depth="<=",
        )  # fmt: skip
        self.fill_program = Pipeline(
            device, ("composite.vert", "fill.frag"),
            {0: "uniform", 1: "storage", 2: "sampler", 3: "sampler", 4: "sampler"},
            colours=(COLOUR,), depth="<=", constants=CONSTANTS,
        )  # fmt: skip
        self.atlas = None  # the texture atlas, layers added as the volume's slots fill
        self.bake_due = False  # a new live frame to bake
        self.images = Image(device, self.image_size, COLOUR, sampled=True, layers=slots, array=True)
        w, h = SHADOW_SIZE
        self.shadows = Image(device, (w * slots, h), RANGE, sampled=True, attachment=True)
        self.shadow_depth = Image(device, (w * slots, h), DEPTH_FORMAT, attachment=True)
        self.projectors = [None] * slots  # (camera_from_world, intrinsics, origin)
        self.live_time = None  # exposure of the image in slot 0
        self.shadow_dirty = set()
        self.buffers = None  # positions, normals, colours, blocks, triangles
        self.capacity = (0, 0)
        self.count = 0  # triangles
        self.depth = None  # the view's depth buffer
        self.epoch = None  # the worker run whose mesh and projectors are shown
        # Kept views whose distance map is older than the mesh: refreshed one per drawn frame,
        # so a mesh update (~12.7 a second) redraws only the live view's map at once.
        self.shadow_stale = set()
        self.guess_dirty = False
        self.levels = _pyramid(device, SHADOW_SIZE)
        gw, gh = GUESS_GRID
        u, v = np.meshgrid(np.linspace(0, 1, gw + 1), np.linspace(0, 1, gh + 1))
        uv = np.stack([u, v], -1).astype(np.float32).reshape(-1, 2)
        i = np.arange(gh)[:, None] * (gw + 1) + np.arange(gw)[None, :]
        quads = np.stack([i, i + 1, i + gw + 1, i + 1, i + gw + 2, i + gw + 1], -1)
        quads = quads.astype(np.uint32)
        self.guess_vbo = Buffer(
            device, uv.nbytes, self._usage(vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
        )
        self.guess_ibo = Buffer(
            device, quads.nbytes, self._usage(vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT)
        )
        self.guess_count = quads.size
        self.guess_target = None  # (size, colour, depth)
        # The viewer's depth map (mipmapped), its depth buffer, and its holes filled.
        self.fill_depth = None
        self.viewer = self.fill_clip = None  # prepare(): world from the viewer, the map's clip
        self._projectors = None  # prepare(): the projectors' block, in this frame's uniforms
        with device.once() as frame:
            frame.upload(self.guess_vbo, uv)
            frame.upload(self.guess_ibo, quads)
            frame.clear(self.shadows, (EMPTY, 0, 0, 0))
            frame.clear(self.images, (0, 0, 0, 0))
            for _, pushed, pulled in self.levels:
                frame.transition(pushed, GENERAL)
                frame.transition(pulled, GENERAL)

    @staticmethod
    def _usage(kind):
        return kind | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT

    # -- updates -------------------------------------------------------------------------
    def upload(self, frame):
        """Apply one fusion.MeshFrame."""
        if frame.epoch != self.epoch:
            # A new worker (restart, reconnect): its map starts empty in its own odometry frame,
            # so nothing of the old one may stay, mesh or projected views.
            self.epoch = frame.epoch
            self.count = 0
            self.projectors = [None] * self.slots
            self.live_time = None
            self.shadow_dirty.clear()
            self.shadow_stale.clear()
            self._clear_atlas()
        if self.atlas is not None:
            for slot in getattr(frame, "cleared", ()):
                self._clear_tiles(slot)
        if frame.positions is not None:
            blocks = getattr(frame, "blocks", None)
            if blocks is None:
                blocks = np.zeros((len(frame.positions), 4), np.float32)
                blocks[:, 3] = -1e6  # no slot: no atlas tile
            elif len(blocks):
                self._atlas_for(int(blocks[:, 3].max()))
            self._mesh(frame.positions, frame.normals, frame.colours, frame.triangles, blocks)
            if self.projectors[0] is not None:
                self.shadow_dirty.add(0)
            self.shadow_stale.update(
                k for k, p in enumerate(self.projectors) if k and p is not None
            )
        # The live frame first: a kept view is usually that same frame (fusion.py keeps the
        # frame it is projecting live), and is then copied on the GPU from slot 0.
        for p in sorted(frame.projectors, key=lambda p: p.slot):
            self._projector(p)
        return True

    apply = upload

    # -- texture atlas -------------------------------------------------------------------
    def _atlas_for(self, slot):
        """Grow the atlas to hold this volume slot's tiles (a layer, 64 MB, at a time; the old
        layers copied over on the GPU)."""
        layers = (slot * DIRECTIONS + DIRECTIONS - 1) // LAYER_TILES + 1
        have = 0 if self.atlas is None else self.atlas.layers
        if layers <= have:
            return
        frame = self.device.frame
        atlas = Image(
            self.device, (LAYER, LAYER), COLOUR, sampled=True, storage=True, layers=layers,
            array=True,
        )  # fmt: skip
        frame.transition(atlas, GENERAL)
        frame.clear(atlas, (0, 0, 0, 0), layers=range(have, layers))
        if self.atlas is not None:
            frame.copy_image(self.atlas, atlas, count=have)
            self.device.defer(self.atlas)
        self.atlas = atlas
        log.info(
            "ToF mesh texture atlas: %d layer(s), %d MB, for volume slots up to %d",
            layers, layers * LAYER * LAYER * 4 // 2**20, layers * LAYER_TILES // DIRECTIONS - 1,
        )  # fmt: skip

    def _clear_tiles(self, slot):
        """Forget the texture of the block that held this slot."""
        side = LAYER // TILE
        zeros = np.zeros((TILE, TILE, 4), np.uint8)
        frame = self.device.frame
        for d in range(DIRECTIONS):
            t = slot * DIRECTIONS + d
            layer, within = divmod(t, LAYER_TILES)
            if layer >= self.atlas.layers:
                return
            y, x = divmod(within, side)
            frame.upload_image(self.atlas, zeros, (x * TILE, y * TILE, TILE, TILE), layer)

    def _clear_atlas(self):
        if self.atlas is not None:
            self.device.frame.clear(self.atlas, (0, 0, 0, 0))

    def _bake(self, frame):
        """The live frame into the atlas, wherever it sees the mesh better than the atlas's view."""
        self.bake_due = False
        camera_from_world, (fx, fy, cx, cy), origin = self.projectors[0]
        clip = pinhole_projection(fx, fy, cx, cy, *self.image_size) @ camera_from_world
        block = pack(
            "16f16f4f4f2ff4x", mat(clip), mat(camera_from_world), (fx, fy, cx, cy),
            (*origin, 0.0), self.image_size, float(self.slots),
        )  # fmt: skip
        frame.transition(self.atlas, GENERAL)
        frame.sampled(self.images, self.shadows)
        sampler = self.device.sampler()
        with frame.render(size=(LAYER, LAYER)):
            frame.use(
                self.bake_program,
                {0: frame.uniform(block), 1: self.atlas, 2: (self.images, sampler),
                 3: (self.shadows, sampler)},
            )  # fmt: skip
            self._vertices(frame)
            frame.draw(self.count * 3, indexed=True)

    def _vertices(self, frame):
        positions, normals, colours, blocks, triangles = self.buffers
        frame.vertices(positions, normals, colours, blocks, index=triangles)

    def _mesh(self, positions, normals, colours, triangles, blocks):
        n, t = len(positions), len(triangles)
        if n > self.capacity[0] or t > self.capacity[1]:
            if self.buffers is not None:
                self.device.defer(*self.buffers)
            cap = (max(1024, int(n * 1.5)), max(1024, int(t * 1.5)))
            vertex = self._usage(vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
            self.buffers = (
                *(Buffer(self.device, cap[0] * size, vertex) for size in (12, 12, 4, 16)),
                Buffer(self.device, cap[1] * 12, self._usage(vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT)),
            )
            self.capacity = cap
        if n:
            frame = self.device.frame
            # After the copies of a mesh uploaded before it in this frame (a display frame takes
            # up to two: window.MESHES_PER_FRAME).
            frame.barrier()
            for buffer, data, kind in zip(
                self.buffers,
                (positions, normals, colours, blocks, triangles),
                (np.float32, np.float32, np.uint8, np.float32, np.uint32),
                strict=True,
            ):
                frame.upload(buffer, np.ascontiguousarray(data, kind))
        self.count = t

    def _projector(self, p):
        fx, fy, cx, cy, w, h = p.intrinsics
        frame = self.device.frame
        if p.image is not None and p.slot and self.live_time == p.time:
            # The image slot 0 holds already: a GPU copy, no second upload.
            frame.copy_image(self.images, self.images, size=self.image_size, layers=(0, p.slot))
            sx, sy = self.image_size[0] / w, self.image_size[1] / h
            fx, fy = fx * sx, fy * sy
            cx, cy = (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5
        elif p.image is not None:
            if p.slot == 0:
                self.live_time = p.time
                self.bake_due = True
            image = p.image
            if (w, h) != self.image_size:
                # Rescale intrinsics to the slot size; nearest resampling is enough here.
                sx, sy = self.image_size[0] / w, self.image_size[1] / h
                ys = np.minimum((np.arange(self.image_size[1]) / sy).astype(int), h - 1)
                xs = np.minimum((np.arange(self.image_size[0]) / sx).astype(int), w - 1)
                image = image[ys][:, xs]
                fx, fy = fx * sx, fy * sy
                cx, cy = (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5
            rgba = np.empty((*image.shape[:2], 4), np.uint8)
            rgba[..., :3] = image
            rgba[..., 3] = 255
            frame.upload_image(self.images, rgba, (0, 0, *self.image_size), p.slot)
        else:
            sx, sy = self.image_size[0] / w, self.image_size[1] / h
            fx, fy = fx * sx, fy * sy
            cx, cy = (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5
        world_from_camera = np.asarray(p.world_from_camera, np.float64)
        self.projectors[p.slot] = (
            np.linalg.inv(world_from_camera),
            (fx, fy, cx, cy),
            world_from_camera[:3, 3].copy(),
        )
        self.shadow_dirty.add(p.slot)

    def _shadows(self, frame):
        self.shadow_stale -= self.shadow_dirty
        if self.shadow_stale:
            self.shadow_dirty.add(self.shadow_stale.pop())
        if not self.shadow_dirty:
            return
        w, h = SHADOW_SIZE
        with frame.render([self.shadows], self.shadow_depth):
            for k in sorted(self.shadow_dirty):
                rect = (k * w, 0, w, h)
                frame.area(rect)
                frame.clear_rect(rect, colour=(EMPTY, 0, 0, 0), depth=1.0)
                entry = self.projectors[k]
                if entry is None or self.buffers is None or not self.count:
                    continue
                camera_from_world, (fx, fy, cx, cy), origin = entry
                clip = pinhole_projection(fx, fy, cx, cy, *self.image_size) @ camera_from_world
                frame.use(self.shadow_program, push=pack("16f4f", mat(clip), (*origin, 0.0)))
                self._vertices(frame)
                frame.draw(self.count * 3, indexed=True)
        if 0 in self.shadow_dirty:
            self.guess_dirty = True
        self.shadow_dirty.clear()

    def _guess_pyramid(self, frame):
        """Fill the holes of the live projector's distance map (slot 0 of the shadow atlas)."""
        if not self.guess_dirty:
            return
        self.guess_dirty = False
        _, (fx, fy, cx, cy), _ = self.projectors[0]
        sx, sy = SHADOW_SIZE[0] / self.image_size[0], SHADOW_SIZE[1] / self.image_size[1]
        intrinsics = (fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5)
        self._push_pull(frame, self.shadows, SHADOW_SIZE, intrinsics)

    def _push_pull(self, frame, source, source_size, intrinsics):
        """Push-pull over the levels; level 0 from `source`, a distance map with these
        intrinsics."""
        nearest, linear = self.device.sampler(False), self.device.sampler()
        frame.sampled(source)
        for index, (size, pushed, _) in enumerate(self.levels):
            frame.barrier()
            frame.use(
                self.push_program, {0: (source, nearest), 1: pushed},
                pack("4f2i2ii", intrinsics, source_size, size, int(index == 0)),
            )  # fmt: skip
            frame.dispatch(size)
            source, source_size = pushed, size
        coarse = self.levels[-1][1]  # the top level's pull is its push
        for size, pushed, pulled in reversed(self.levels):
            frame.barrier()
            frame.use(
                self.pull_program, {0: (pushed, nearest), 1: (coarse, linear), 2: pulled},
                pack("2i", *size),
            )  # fmt: skip
            frame.dispatch(size)
            coarse = pulled
        frame.barrier()

    def _guess_layer(self, size):
        if self.guess_target is None or self.guess_target[0] != size:
            if self.guess_target is not None:
                self.device.defer(*self.guess_target[1:])
            colour = Image(self.device, size, COLOUR, sampled=True, attachment=True)
            depth = Image(self.device, size, DEPTH_FORMAT, attachment=True)
            self.guess_target = (size, colour, depth)
        return self.guess_target

    def _fill_depth(self, frame, projection, view_projection):
        """The viewer's depth map for the fill (once per frame), its mipmaps built and its holes
        filled."""
        w = FILL_DEPTH_WIDTH
        # Square pixels: the projection's aspect (width over height) from its focal lengths.
        size = (w, max(1, round(w * projection[0, 0] / projection[1, 1])))
        if self.fill_depth is None or self.fill_depth[0].size != size:
            if self.fill_depth is not None:
                self.device.defer(*self.fill_depth)
            levels = int(np.log2(max(size))) + 1
            self.fill_depth = (
                Image(self.device, size, INVERSE, sampled=True, storage=True, attachment=True,
                      levels=levels),
                Image(self.device, size, DEPTH_FORMAT, attachment=True),
                Image(self.device, size, MAP, sampled=True, storage=True),
            )  # fmt: skip
        texture, depth, surface = self.fill_depth
        with frame.render([texture], depth, clear=(0, 0, 0, 0), depth_clear=1.0):
            if self.count and self.buffers is not None:
                frame.use(
                    self.fill_depth_program, push=pack("16f4f", mat(view_projection), (0,) * 4)
                )
                self._vertices(frame)
                frame.draw(self.count * 3, indexed=True)
        frame.transition(texture, GENERAL)
        for level in range(1, texture.levels):
            frame.barrier()
            frame.use(self.downsample_program, {0: (texture, level - 1), 1: (texture, level)})
            frame.dispatch((max(1, size[0] >> level), max(1, size[1] >> level)))
        frame.barrier()
        frame.transition(surface, GENERAL)
        frame.use(
            self.resolve_program,
            {0: (texture, self.device.sampler(mipmaps=True)), 1: surface},
        )
        frame.dispatch(size)
        frame.barrier()

    def _fill(self, frame, view_projection, world_from_eye, size):
        """Pixels still empty after the mesh: a guessed range from the viewer's depth map,
        coloured by the kept view that saw that spot most nearly along the viewer's ray."""
        if not any(p is not None for p in self.projectors):
            return
        block = pack(
            "16f16f4f4f4f2f2f3f4x",
            mat(np.linalg.inv(view_projection)), mat(self.fill_clip),
            (*world_from_eye[:3, 3], 0.0), (*self.viewer[:3, 3], 0.0),
            (*-self.viewer[:3, 2], 0.0), size, self.image_size,
            FLOOR_HEIGHT, FLOOR_CLEARANCE, GUESS_FAR,
        )  # fmt: skip
        sampler = self.device.sampler()
        frame.use(
            self.fill_program,
            {0: frame.uniform(block), 1: self._projectors, 2: (self.fill_depth[2], sampler),
             3: (self.images, sampler), 4: (self.shadows, sampler)},
        )  # fmt: skip
        frame.draw(3)

    def _draw_guess(self, frame, view_projection):
        camera_from_world, (fx, fy, cx, cy), origin = self.projectors[0]
        world_from_camera = np.eye(4)
        world_from_camera[:3, :3] = np.linalg.inv(camera_from_world)[:3, :3]
        block = pack(
            "16f16f4f4f2f2f4f", mat(view_projection), mat(world_from_camera), (*origin, 0.0),
            (fx, fy, cx, cy), self.image_size, (1 / GUESS_GRID[0], 1 / GUESS_GRID[1]),
            (FLOOR_HEIGHT, FLOOR_CLEARANCE, GUESS_FAR, GUESS_EDGE),
        )  # fmt: skip
        sampler = self.device.sampler()
        frame.use(
            self.guess_program,
            {0: frame.uniform(block), 1: (self.images, sampler),
             2: (self.levels[0][2], sampler), 3: (self.levels[0][1], sampler)},
        )  # fmt: skip
        frame.vertices(self.guess_vbo, index=self.guess_ibo)
        frame.draw(self.guess_count, indexed=True)

    # -- drawing -------------------------------------------------------------------------
    def prepare(self, robot_camera, head, projection):
        """Once per displayed frame, before its views are drawn: what every view of the frame
        shares. head is the viewer, the point between the pilot's eyes (a desktop's one view),
        and projection covers every view's field (the fill's depth map is drawn with it)."""
        self.viewer = validate_pose(robot_camera) @ validate_pose(head)
        projection = np.asarray(projection, dtype=np.float64)
        if projection.shape != (4, 4) or not np.isfinite(projection).all():
            raise ValueError("Projection must be a finite 4x4 matrix")
        frame = self.device.frame
        self._shadows(frame)
        if (
            self.bake_due and self.atlas is not None and self.count and self.buffers is not None
            and self.projectors[0] is not None
        ):  # fmt: skip
            self._bake(frame)
        if self.projectors[0] is not None:
            self._guess_pyramid(frame)
        block = bytearray()
        for k, entry in enumerate(self.projectors):
            if entry is None:
                block += bytes(96)
                continue
            block += pack("16f4f4f", mat(entry[0]), entry[1], (*entry[2], 2.0 if k == 0 else 1.0))
        self._projectors = frame.uniform(bytes(block))
        self.fill_clip = projection @ _view(self.viewer)
        if self.projectors[0] is not None:
            self._fill_depth(frame, projection, self.fill_clip)

    def draw(self, robot_camera, head, projection, target, *, clear=CLEAR):
        """One view of the frame prepare() set up: the eye at head, with this projection, into
        target (an image of format COLOUR, OpenGL's row order: vk.Presenter)."""
        world_from_eye = validate_pose(robot_camera) @ validate_pose(head)
        projection = np.asarray(projection, dtype=np.float64)
        if projection.shape != (4, 4) or not np.isfinite(projection).all():
            raise ValueError("Projection must be a finite 4x4 matrix")
        frame = self.device.frame
        size = target.size
        if self.depth is None or self.depth.size != size:
            if self.depth is not None:
                self.device.defer(self.depth)
            self.depth = Image(self.device, size, DEPTH_FORMAT, attachment=True)
        guess = self.projectors[0] is not None
        view_projection = projection @ _view(world_from_eye)
        sampler = self.device.sampler()
        frame.sampled(self.images, self.shadows)
        if self.atlas is not None:
            frame.sampled(self.atlas)
        if guess:
            frame.sampled(self.fill_depth[2])
        with frame.render([target], self.depth, clear=clear or (0, 0, 0, 1), depth_clear=1.0):
            if self.count and self.buffers is not None:
                block = pack(
                    "16f4f2fi4x", mat(view_projection), (*self.viewer[:3, 3], 0.0),
                    self.image_size, 0 if self.atlas is None else self.atlas.layers,
                )  # fmt: skip
                atlas = self.atlas or self.device.dummy_array
                frame.use(
                    self.program,
                    {0: frame.uniform(block), 1: self._projectors, 2: (self.images, sampler),
                     3: (self.shadows, sampler), 4: (atlas, sampler)},
                )  # fmt: skip
                self._vertices(frame)
                frame.draw(self.count * 3, indexed=True)
            if guess:
                # Under the live guess (both at depth 1.0; the live frame's own guess wins).
                self._fill(frame, view_projection, world_from_eye, size)
        if guess:
            # The guesses into their own layer (ordinary depth among themselves), then into the
            # view where it still has its clear depth: wherever no mesh was drawn.
            _, layer, layer_depth = self._guess_layer(size)
            with frame.render([layer], layer_depth, clear=(0, 0, 0, 0), depth_clear=1.0):
                self._draw_guess(frame, view_projection)
            frame.sampled(layer)
            with frame.render([target], self.depth):
                frame.use(self.composite_program, {0: (layer, self.device.sampler(False))})
                frame.draw(3)

    def close(self):
        resources = [
            self.program, self.bake_program, self.shadow_program, self.fill_depth_program,
            self.push_program, self.pull_program, self.downsample_program, self.resolve_program,
            self.guess_program, self.composite_program, self.fill_program, self.atlas,
            self.images, self.shadows, self.shadow_depth, self.guess_vbo, self.guess_ibo,
            self.depth, *(self.buffers or ()), *(self.fill_depth or ()),
            *(image for level in self.levels for image in level[1:]),
            *(self.guess_target[1:] if self.guess_target is not None else ()),
        ]  # fmt: skip
        for resource in resources:
            if resource is not None:
                resource.close()
