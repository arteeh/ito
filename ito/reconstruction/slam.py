"""MASt3R-SLAM (Murai et al., CVPR 2025) as its own main.py runs it, in Ito's worker.

The bundled upstream (models/slam, ito.build) does the SLAM: FrameTracker tracks every frame
against the last keyframe and picks keyframes, the backend links each keyframe to the one
before and to what retrieval recalls and solves the global Sim(3) graph, and a lost camera
relocalises by retrieval, mapping nothing meanwhile. This module is main.py's loop, its
backend's loop (its work per keyframe and its relocalisation), ported line for line, in one
process: the frontend in the worker's thread, the backend in a thread of its own, as upstream's
base config runs them in two processes (single_thread off): the frontend never waits for the
global solve. Ito's own code is the edges: Input makes
frames as upstream's dataloader and create_frame do, and the camera pose and the keyframes'
pointmaps (at their current poses, as upstream's viewer draws them) go out to the pilot.
Where the robot has a ToF sensor, tof.RangeScale fixes once how many metres a map unit is, and
the pilot gets the map at that scale; the solve never sees it.
"""

import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from ito.render.gpu import GPU

from .graphs import graph_model
from .rgbd import RGBDBackend
from .tof import RangeScale

AXES = np.array([1.0, -1.0, -1.0])  # OpenCV camera/world axes (+Y down, +Z forward) to Ito.
# MASt3R's input, long side, per Input resolution setting; 512 is upstream's.
LONG_SIDE = {25: 256, 50: 368, 75: 448, 100: 512}
# Upstream's buffers, all allocated at start: the keyframe count is the map size setting
# (upstream: 512); graph edges get the most the backend can add per keyframe, so they never
# fill before the keyframes do: run_backend links a keyframe to the one before it and to at most
# k (base.yaml: 3) that retrieval recalls, a relocalisation to at most k. Bytes per input pixel
# per keyframe: the keyframe (44: image, points, confidence, MASt3R features), its share of the
# solve's inputs (16) and its graph edges (26 each).
EDGES_PER_KEYFRAME = 4
KEYFRAME_BYTES_PER_PIXEL = 44 + 16 + 26 * EDGES_PER_KEYFRAME
# Keyframe slots beyond the map size, only for relocalising a full map (see relocalization).
SPARE = 1
# Keyframe pairs decoded together when the backend links a keyframe (see pairwise).
PAIRS_PER_DECODE = 1
CONFIDENCE = 1.5  # upstream's viewer draws points above this average confidence
MOVED = 0.005  # a keyframe the solve moved by more than this share of its depth is redrawn
RETRIES = 2  # keyframes the splat pool could not take whole, offered again per frame
BACKEND_WAIT = 2.0  # seconds the worker's exit waits for the backend thread (< EXIT_WAIT)
# Keyframes redrawn per frame because the solve moved them or their splat size changed, the
# most moved first; the rest wait for the next frames. All at once stalled tracking.
REDRAWS = 8
# ... and only while a frame's redraws took less than this: a moved keyframe's redraw takes
# 10-40 ms, and eight of them stalled a frame by 0.35 s after a relocalisation.
REDRAW_SECONDS = 0.025
# A loop closure, for the status: an edge the match kept to a keyframe more than this many
# before the new one (upstream prints every retrieval but the keyframe before: mostly the
# recent neighbours).
LOOP_GAP = 5
# Free VRAM after setup below this: the driver is already spilling into system memory.
SPILLING = 256 << 20
# VRAM a frame's work takes beyond the buffers, per input pixel (MASt3R's activations, matching,
# the backend's pairs): at 512x288 0.5-0.7 GiB at the peak, 0.7-1.1 GiB with the streams' caches.
WORK_BYTES_PER_PIXEL = 5000
OOM_SHOWN = 10.0  # seconds the status keeps saying a frame was dropped for VRAM
# A frame out of VRAM again within this many seconds drops MASt3R's CUDA graphs for the session
# (~40 % slower frames); one alone, as another program's brief spike, only drops the frame.
GRAPHS_OOM_WINDOW = 30.0


def crop_size(width, height):
    """Half width and half height of mast3r_utils.resize_img's centre crop of a resized image:
    multiples of 8, so the crop lies on MASt3R's 16-pixel patch grid. A square image is cropped
    to 4:3 as upstream does; Ito also rounds that height down to the grid, which changes nothing
    at upstream's 512 and keeps the other long sides on it (368: 276 rows would not be)."""
    cx, cy = width // 2, height // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if width == height:
        halfh = 3 * halfw // 4 // 8 * 8
    return halfw, halfh


def input_shape(width, height, percent):
    """Width and height of the frames MASt3R sees for a camera (see crop)."""
    long = LONG_SIDE[percent]
    size = max(width, height)
    halfw, halfh = crop_size(*(int(round(x * long / size)) for x in (width, height)))
    return 2 * halfw, 2 * halfh


def map_bytes(keyframes, width, height, percent):
    """VRAM MASt3R-SLAM's buffers take at start, for a camera and an Input resolution."""
    w, h = input_shape(width, height, percent)
    return (keyframes + SPARE) * KEYFRAME_BYTES_PER_PIXEL * w * h


_bands = []  # the resize's thread pool, made at first use


def resize(image, long):
    """mast3r_utils._resize_pil_image, with an RGB image's three bands resampled at once on
    threads: Pillow releases the GIL while it resamples, and per band it computes the same bytes
    (1280x720 to 512x288: 8.6 -> 5.2 ms on Ceres)."""
    import PIL.Image

    S = max(image.size)
    interp = PIL.Image.LANCZOS if S > long else PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long / S)) for x in image.size)
    if image.mode != "RGB":
        return image.resize(new_size, interp)
    if not _bands:
        from concurrent.futures import ThreadPoolExecutor

        _bands.append(ThreadPoolExecutor(3, thread_name_prefix="input-resize"))
    bands = _bands[0].map(lambda band: band.resize(new_size, interp), image.split())
    return PIL.Image.merge("RGB", list(bands))


def crop(image, long):
    """mast3r_utils.resize_img's 512 branch for any long side: resize the long side to it,
    centre-crop to multiples of 16. Returns the image and its return_transformation."""
    W1, H1 = image.size
    image = resize(image, long)
    W, H = image.size
    cx, cy = W // 2, H // 2
    halfw, halfh = crop_size(W, H)
    image = image.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
    return image, (W1 / W, H1 / H, (W - image.size[0]) / 2, (H - image.size[1]) / 2)


class Input:
    """Upstream's dataloader for a calibrated camera: frames as create_frame makes them from the
    driver's RGB, and K for them as Intrinsics derives it (K_frame). The driver's frames are
    already pinhole: there is nothing to undistort."""

    def __init__(self, intrinsics, percent, device="cuda"):
        import PIL.Image

        self.long, self.device = LONG_SIDE[percent], device
        blank = PIL.Image.new("RGB", (intrinsics.width, intrinsics.height))
        image, (scale_w, scale_h, half_crop_w, half_crop_h) = crop(blank, self.long)
        self.width, self.height = image.size
        self.K = np.array(
            [
                [intrinsics.fx / scale_w, 0, intrinsics.cx / scale_w - half_crop_w],
                [0, intrinsics.fy / scale_h, intrinsics.cy / scale_h - half_crop_h],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    def frame(self, index, rgb, T_WC):
        import PIL.Image
        import torch
        from dust3r.utils.image import ImgNorm
        from mast3r_slam.frame import Frame

        image, _ = crop(PIL.Image.fromarray(rgb), self.long)
        img = ImgNorm(image)[None].to(device=self.device)
        img_shape = torch.tensor(np.int32([image.size[::-1]]), device=self.device)
        uimg = torch.from_numpy(np.array(image)) / 255.0
        return Frame(index, img, img_shape, img_shape.clone(), uimg, T_WC)


def pairwise(match):
    """Upstream's mast3r_match_symmetric, one keyframe pair at a time: the backend links a new
    keyframe to up to 1 + k at once, and matching them as one batch held every pair's full-size
    decoder outputs (points, descriptors, confidences: four images a pair, stacked and
    concatenated copies) together. Upstream decodes pair by pair anyway and matches each pair
    on its own, so the results are the same, concatenated."""
    import torch

    def one_at_a_time(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
        if len(feat_i) <= PAIRS_PER_DECODE:
            return match(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j)
        parts = [
            match(model, feat_i[s], pos_i[s], feat_j[s], pos_j[s], shape_i[s], shape_j[s])
            for s in (
                slice(n, n + PAIRS_PER_DECODE) for n in range(0, len(feat_i), PAIRS_PER_DECODE)
            )
        ]
        return tuple(torch.cat(outputs) for outputs in zip(*parts))

    one_at_a_time.pairwise = True
    return one_at_a_time


class StreamLock:
    """A reentrant lock that also orders CUDA work across threads. Upstream's frontend and
    backend share the keyframes and the frame state (SharedKeyframes, SharedStates) and guard
    them with one lock each; in Ito they are two threads, each on its own CUDA stream, and a
    lock orders only the CPU. So a holder's release records an event on its stream, and the
    next holder's stream waits for it: what one side queued under the lock (a solve's poses, the
    tracker's refined pointmap, a snapshot copy) is done before the other side's work under it
    runs, in both directions, and nobody blocks on the GPU. Shared tensors are only read or
    written under the lock, or copied there; views kept past it are of data the other side
    never writes."""

    def __init__(self, cuda):
        self.lock = threading.RLock()
        self.cuda = cuda
        self.depth = 0
        self.released = None  # the last holder's event

    def __enter__(self):
        self.lock.acquire()
        self.depth += 1
        if self.depth == 1 and self.released is not None:
            self.cuda.current_stream().wait_event(self.released)
        return self

    def try_enter(self):
        """__enter__ if no other thread holds the lock, without waiting; True if entered."""
        if not self.lock.acquire(blocking=False):
            return False
        self.depth += 1
        if self.depth == 1 and self.released is not None:
            self.cuda.current_stream().wait_event(self.released)
        return True

    def __exit__(self, *exc):
        self.depth -= 1
        try:
            if self.depth == 0:
                event = self.cuda.Event()
                event.record()
                self.released = event
        finally:
            self.lock.release()


class TrackedKeyframes:
    """The keyframes as FrameTracker sees them. Its keyframe's pose is a copy, as the backend's
    solve may move the keyframe while it tracks; its write-back stores what the tracker changes,
    the keyframe's pointmap (SharedKeyframes.__setitem__'s part for it), not that copy."""

    def __init__(self, keyframes, Sim3):
        self.keyframes, self.Sim3 = keyframes, Sim3

    def __len__(self):
        return len(self.keyframes)

    def last_keyframe(self):
        with self.keyframes.lock:
            keyframe = self.keyframes.last_keyframe()
            if keyframe is not None:
                keyframe.T_WC = self.Sim3(keyframe.T_WC.data.clone())
        return keyframe

    def __setitem__(self, idx, keyframe):
        shared = self.keyframes
        with shared.lock:
            shared.X[idx] = keyframe.X_canon
            shared.C[idx] = keyframe.C
            shared.N[idx] = keyframe.N
            shared.N_updates[idx] = keyframe.N_updates
            shared.is_dirty[idx] = True


class Manager:
    """What SharedKeyframes and SharedStates take from a multiprocessing Manager, for one
    process: upstream shares them between its frontend, backend and viewer processes. cuda:
    torch.cuda, for locks that also order the threads' CUDA streams (StreamLock)."""

    def __init__(self, cuda=None):
        self.cuda = cuda

    def RLock(self):
        return threading.RLock() if self.cuda is None else StreamLock(self.cuda)

    @staticmethod
    def Value(typecode, value):
        return SimpleNamespace(value=value)

    @staticmethod
    def list():
        return []


class SLAMBackend(RGBDBackend):
    def __init__(
        self,
        capacity,
        intrinsics,
        *,
        report,
        precision="full",
        input_resolution=100,
        keyframes=128,
        precision_report=None,
        **options,
    ):
        from .mast3r_runtime import load_model, prepare

        source, weights, retrieval = prepare(report)
        import lietorch
        import torch
        from mast3r_slam.config import config, load_config
        from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates
        from mast3r_slam import global_opt
        from mast3r_slam.global_opt import FactorGraph
        from mast3r_slam.mast3r_utils import load_retriever, mast3r_inference_mono
        from mast3r_slam.tracker import FrameTracker

        if not hasattr(global_opt.mast3r_match_symmetric, "pairwise"):
            global_opt.mast3r_match_symmetric = pairwise(global_opt.mast3r_match_symmetric)
        load_config(str(source / "config/base.yaml"))
        if 1 + config["retrieval"]["k"] > EDGES_PER_KEYFRAME:
            raise RuntimeError("MASt3R-SLAM's retrieval k outgrew Ito's graph edge buffer")
        # On top of base.yaml, what upstream's config/calib.yaml sets: a calibrated camera.
        # base.yaml's single_thread stays off: a frontend that waited for every global solve
        # (eval_calib.yaml) lost a camera turning while it waited (close tape, 0.5 s stalls).
        config["use_calib"] = True
        # Upstream relocalises a lost frame only if every keyframe retrieval recalls matches it
        # 30 % (reloc.strict in base.yaml); its own ETH3D config turns that off, and so does
        # Ito. Back at a mapped place on the s2-twist tape (64-80 s) the three recalled matched
        # 20-33 % each, frame after frame, and strict never relocalised. Non-strict drops the
        # edges under 30 % and relocalises on the rest.
        config["reloc"]["strict"] = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)
        options["device"] = "cuda"
        super().__init__(capacity, intrinsics, **options)
        self.report = report
        # Out of VRAM: Torch's, or CuPy's in the splat map.
        self.oom = (torch.cuda.OutOfMemoryError,) + (
            (self.xp.cuda.memory.OutOfMemoryError,) if self.xp is not np else ()
        )
        self.torch, self.lietorch, self.config, self.Mode = torch, lietorch, config, Mode
        self.inference = mast3r_inference_mono
        self.input = Input(intrinsics, input_resolution)
        h, w = self.input.height, self.input.width
        self.settings = (
            f"{precision.title()} precision, {input_resolution}% input resolution, "
            f"{keyframes} keyframes"
        )
        self.lower = "lower the Map size or Input resolution" + (
            " or choose Half" if precision == "full" else ""
        )
        self.capacity = keyframes
        buffer = keyframes + SPARE
        try:
            self.model = load_model(
                weights, report, precision=precision, precision_report=precision_report
            )
            K = torch.as_tensor(self.input.K, device="cuda")
            manager = Manager(torch.cuda)
            self.keyframes = SharedKeyframes(manager, h, w, buffer=buffer)
            self.states = SharedStates(manager, h, w)
            self.keyframes.set_intrinsics(K)
            self.tracker = FrameTracker(
                self.model, TrackedKeyframes(self.keyframes, lietorch.Sim3), "cuda"
            )
            self.graph = FactorGraph(
                self.model, self.keyframes, K, "cuda", max_edges=EDGES_PER_KEYFRAME * buffer
            )
            self.retrieval = load_retriever(self.model, str(retrieval))
            # Ito's own, also once: each keyframe's pose when last drawn, and its depth.
            self.drawn = torch.zeros(buffer, 1, lietorch.Sim3.embedded_dim, device="cuda")
            self.depth = torch.ones(buffer, device="cuda")
            # ... and whether the splat pool still owes it splats (see draw).
            self.pending = np.zeros(buffer, bool)
            # ... and whether it waits to be redrawn: the scale or splat size changed.
            self.stale = np.zeros(buffer, bool)
            free, _ = torch.cuda.mem_get_info()
            # On Windows the runtime reports this process's own budget: with another program
            # holding 6 GiB of a 12 GiB card it said 10.8 GiB free, NVML 4.9 (#83).
            card = GPU(torch.cuda.current_device())
            if card.library is not None:
                free = min(free, card.free)
            card.close()
        except self.oom as exc:
            raise self.setup_oom() from exc
        # Windows' driver does not fail an allocation past the card: it spills into system
        # memory, so running short shows only as a crawl. Say so at the start instead.
        working = WORK_BYTES_PER_PIXEL * h * w
        self.warning = (
            f"VRAM short ({free / 2**20:.0f} MiB free after MASt3R-SLAM's setup, its frames "
            f"need about {(working + SPILLING) / 2**20:.0f}): the driver spills into system "
            f"memory and everything slows; {self.lower}"
            if free < working + SPILLING
            else None
        )
        v, u = np.mgrid[:h, :w].reshape(2, -1).astype(np.float32)
        K = self.input.K
        rays = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
        self.rays = torch.as_tensor(np.stack(rays), device="cuda")
        self.axes = torch.tensor(AXES, device="cuda", dtype=torch.float32)
        self.world = None
        self.i = 0
        self.shown = 0
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.tracked = 0
        self.lost = False
        self.spare = None  # the graph's edge count before a full map's spare keyframe, if any
        self.loops = 0
        self.relocalized = 0
        self.appends = 0  # keyframes the frontend added: a frame that did is not dropped on OOM
        self.ooms, self.oom_at = 0, -OOM_SHOWN
        self.frame_oom_at = -GRAPHS_OOM_WINDOW  # the last frame dropped for VRAM
        self.vram_full_at, self.vram_full_free = -OOM_SHOWN, 0  # see vram_full
        self.tof = RangeScale(self.input.K, (h, w))
        self.scale = 1.0  # metres per map unit, for display; 1 until the ToF fixes it
        self.drawn_pixels = self.footprint()
        self.draw_seconds = 0.0
        self.pose_ready = None  # the worker's: hands the pilot the pose before draw's work
        self.ranged = None
        report("MASt3R-SLAM warming up")
        # MASt3R's units as CUDA graphs, captured by the warm-up.
        graph_model(self.model)
        try:
            self.warm_up()
            short = False
        except self.oom:
            short = True  # handled out here, where the warm-up's tensors are gone
        if short:
            self.shed_graphs()
            try:
                self.warm_up()
            except self.oom as exc:
                raise self.setup_oom() from exc
        # What setup and the warm-up cached was on the default stream; the frontend and backend
        # run on streams of their own, whose caches never reuse it (0.4-0.8 GiB held for nothing).
        if self.xp is not np:
            self.xp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        # The backend thread: upstream's run_backend, on a CUDA stream of its own. Its lock
        # keeps the frontend's drop_spare out of a relocalisation or solve.
        self.backend_lock = threading.RLock()
        self.backend_stream = torch.cuda.Stream()
        self.reloc_frames = self.reloc_tried = 0  # frames the frontend set for relocalising
        self.resumed = 0  # relocalisations the frontend has started tracking from
        self.failure = None
        self.closing = False
        self.backend_thread = threading.Thread(
            target=self.run_backend, name="mast3r-slam-backend", daemon=True
        )
        self.backend_thread.start()
        report("MASt3R-SLAM ready; waiting for camera")

    def warm_up(self):
        """A frame's work on a noise image before the first real one, none of it kept: CUDA,
        cuSOLVER and CuPy load or compile their kernels at first use, which took the first frame
        over 6 s (CuPy's voxel kernels 5 s). Live, a first frame that slow is far from the
        next, and upstream's tracker loses it: a camera that turns meanwhile never tracks."""
        from mast3r_slam.mast3r_utils import mast3r_match_asymmetric

        torch, intrinsics = self.torch, self.intrinsics
        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 256, (intrinsics.height, intrinsics.width, 3), dtype=np.uint8)
        identity = self.lietorch.Sim3.Identity(1, device="cuda")
        keyframe, frame = (self.input.frame(0, rgb, identity) for _ in range(2))
        keyframe.update_pointmap(*self.inference(self.model, keyframe))
        mast3r_match_asymmetric(self.model, frame, keyframe)
        features = self.retrieval.prep_features(keyframe.feat)[0]
        self.retrieval.quantize_custom(features, self.retrieval.asmk.params.get("query_ivf"))
        torch.linalg.cholesky(torch.eye(7, device="cuda"))
        (identity.inv() * identity).act(torch.zeros(1, 3, device="cuda"))
        xp = self.xp
        points = xp.asarray(rng.normal(size=(4096, 3)), dtype=xp.float32)
        colors = xp.asarray(rng.integers(0, 256, (4096, 3)), dtype=xp.uint8)
        sizes = xp.full(4096, 0.01, dtype=xp.float32)
        RGBDBackend(64, intrinsics, device="cuda").integrate_points(points, colors, 0.0, sizes)
        torch.cuda.synchronize()

    def shed_graphs(self):
        """Out of VRAM with MASt3R's CUDA graphs: they keep their own buffers and outputs (0.3-0.4
        GiB at Half, 100 %), so drop them and run MASt3R eagerly, at the same precision, slower.
        True if they were dropped now."""
        graphs = getattr(self.model, "graphs", None)
        if graphs is None or graphs.off:
            return False
        graphs.drop()
        note = "VRAM short for MASt3R's CUDA graphs: running without them, slower"
        self.warning = f"{note}; {self.lower}" if self.warning is None else f"{self.warning}. {note}"
        return True

    def bind_admission(self, admission):
        from .admission import bind_model, bind_native

        super().bind_admission(admission)
        self.admission_hooks = bind_model(self.model, admission)
        bind_native(admission)

    def integrate(self, rgb, depth, camera, now):
        """One frame through upstream's main loop. camera: the robot's own estimate of its camera
        pose (world-from-camera, Ito axes); only the first frame's is used, once, to place the
        map in the robot's gravity-aligned world. It never enters the solve."""
        if self.failure is not None:
            exc, self.failure = self.failure, None
            if isinstance(exc, self.torch.cuda.OutOfMemoryError):
                # Half a keyframe's backend work leaves upstream's keyframes, graph and retrieval
                # out of step: that ends this map.
                raise RuntimeError(
                    f"Out of VRAM in MASt3R-SLAM's backend at {self.settings}: {self.lower} in "
                    "Settings"
                ) from exc
            raise RuntimeError(f"MASt3R-SLAM's backend failed: {exc}") from exc
        before = self.appends  # not len(keyframes): the backend adds relocalisations
        try:
            self.step(rgb, camera)
        except self.torch.cuda.OutOfMemoryError as exc:
            # The worker drops a frame that ran out of memory and goes on: safe for tracking,
            # not once the frame went into the keyframes.
            if self.appends == before:
                if now - self.frame_oom_at < GRAPHS_OOM_WINDOW:
                    self.shed_graphs()
                self.ooms, self.oom_at, self.frame_oom_at = self.ooms + 1, now, now
                self.report(self.out_of_memory())
                raise
            raise RuntimeError(
                f"Out of VRAM in MASt3R-SLAM at {self.settings}: {self.lower} in Settings"
            ) from exc
        if self.ranged is not None and self.tof.value is None and self.measure(self.ranged):
            self.scale = self.tof.value
            self.stale[:] = True  # every keyframe again, at the new scale
        self.pose(self.posed)
        if self.pose_ready is not None:
            self.pose_ready()
        began = time.perf_counter()
        try:
            self.draw(now)
        except self.oom:
            # The frame is tracked; the keyframes it did not redraw stay stale or owed (show)
            # and are redrawn in the next frames. The status says so (publish).
            self.ooms, self.oom_at = self.ooms + 1, now
            self.torch.cuda.empty_cache()
        self.draw_seconds = time.perf_counter() - began
        self.publish(self.posed, now)

    def vram_full(self, now, free):
        """The worker's VRAM check found the card nearly full (memory.VRAM): it cuts the splat
        limit, and the map fades without an error. The status says so for a while."""
        self.vram_full_at, self.vram_full_free = now, free

    def setup_oom(self):
        return RuntimeError(
            f"Out of VRAM for MASt3R-SLAM at {self.settings}: {self.lower} in Settings"
        )

    def out_of_memory(self):
        return (
            f"Out of VRAM at {self.settings}, {self.ooms} times (frames dropped, map redraws "
            f"put off); {self.lower} in Settings"
        )

    def add_ranges(self, grid):
        self.tof.add(grid)

    def measure(self, frame):
        """Hand the ToF scale a posed frame's depth in map units: its own points' depth times
        its Sim(3) scale. source_captured: the frame's capture time, set by the worker."""
        torch = self.torch
        z = frame.X_canon[:, 2]
        known = (frame.get_average_conf()[:, 0] > CONFIDENCE) & (z > 0)
        depth = torch.where(known, z * frame.T_WC.data[0, 7], torch.nan)
        return self.tof.measure(self.source_captured, depth)

    def step(self, rgb, camera):
        Mode, states, keyframes = self.Mode, self.states, self.keyframes
        mode = states.get_mode()
        # get frames last camera pose
        T_WC = (
            self.lietorch.Sim3.Identity(1, device=self.axes.device)
            if self.i == 0
            else states.get_frame().T_WC
        )
        if mode == Mode.TRACKING and self.resumed != self.relocalized:
            # Ito: the first frame after a relocalisation starts from the relocalised keyframe's
            # pose. Upstream's starts from the last tracked one, from before the loss: a camera
            # that moved while lost is far from it, calibrated tracking then fails at once and
            # relocalises again, a keyframe more each time (loop tape: 18 in 10 s).
            self.resumed = self.relocalized
            # Upstream resets the tracker's match guess whenever its keyframe is new; after a
            # relocalisation it is, and the guess left is from before the loss.
            self.tracker.reset_idx_f2k()
            with keyframes.lock:
                T_WC = self.lietorch.Sim3(keyframes.T_WC[len(keyframes) - 1].clone())
        frame = self.input.frame(self.i, rgb, T_WC)
        posed = self.ranged = None
        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = self.inference(self.model, frame)
            frame.update_pointmap(X_init, C_init)
            self.appends += 1
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            self.place(camera)
            posed = frame.T_WC
            self.ranged = frame
        else:
            add_new_kf = False
            if mode == Mode.TRACKING:
                add_new_kf, _, try_reloc = self.tracker.track(frame)
                if try_reloc:
                    self.drop_spare()  # before RELOC: the backend relocalises into that slot
                    # Ito: the frame first, then the mode the backend acts on.
                    with states.lock:
                        states.set_frame(frame)
                        self.reloc_frames += 1
                    states.set_mode(Mode.RELOC)
                else:
                    posed = frame.T_WC
                    self.ranged = frame
                    states.set_frame(frame)
            elif mode == Mode.RELOC:
                X, C = self.inference(self.model, frame)
                frame.update_pointmap(X, C)
                with states.lock:
                    states.set_frame(frame)
                    self.reloc_frames += 1
                states.queue_reloc()
            else:
                raise RuntimeError(f"Invalid MASt3R-SLAM mode {mode}")
            # Ito: a full map maps nothing new; upstream's buffer would overflow (IndexError).
            if add_new_kf and len(keyframes) < self.capacity:
                self.appends += 1
                keyframes.append(frame)
                states.queue_global_optimization(len(keyframes) - 1)
        self.i += 1
        self.posed = posed

    def close(self):
        """Stop the backend thread. With the worker stopped, admission ends its GPU work at the
        next chunk (InterruptedError); a call that never returns is left to the process's exit."""
        self.closing = True
        self.backend_thread.join(timeout=BACKEND_WAIT)

    def run_backend(self):
        self.torch.set_grad_enabled(False)  # thread-local: the constructor's covers its thread
        try:
            while not self.closing:
                if not self.backend_step():
                    time.sleep(0.01)
        except BaseException as exc:  # noqa: BLE001 (handed to the frontend, which raises it)
            self.failure = exc

    def backend_step(self):
        """One pass of upstream's run_backend loop (main.py). False when it had nothing to do.
        Ito: on the backend's own CUDA stream; what it shares with the frontend it reads and
        writes under upstream's locks, which order the two streams (StreamLock)."""
        Mode, states = self.Mode, self.states
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            return False
        torch = self.torch
        stream = self.backend_stream
        with self.backend_lock, torch.cuda.stream(stream) if stream is not None else nullcontext():
            # Ito: queued keyframes first, in RELOC too, where upstream's loop relocalises first.
            # Retrieval numbers its entries in the order they come and Ito takes those numbers for
            # keyframe indices: a relocalised keyframe added before a queued one swapped them for
            # good (wrong recalls, wrong relocalisation poses). And a full map's spare keyframe's
            # edges must be the graph's last, as drop_spare removes the graph's tail.
            idx = -1
            with states.lock:
                if len(states.global_optimizer_tasks) > 0:
                    idx = states.global_optimizer_tasks[0]
            if idx != -1:
                self.optimize(idx)
                if stream is not None:
                    stream.synchronize()
                with states.lock:
                    states.global_optimizer_tasks.pop(0)
                return True
            if mode == Mode.RELOC:
                with states.lock:
                    # Ito: each frame once. Upstream's loop tries the frame it has over and
                    # over until the next comes, with the same result: live, that kept the GPU
                    # and the GIL so busy that the frontend's frames took 0.9 s, not 0.12.
                    if self.reloc_tried == self.reloc_frames:
                        return False
                    self.reloc_tried = self.reloc_frames
                    # Ito: a copy, the frontend overwrites states' frame with the next one.
                    frame = states.get_frame()
                    for name in ("img", "uimg", "X_canon", "C", "feat", "pos"):
                        setattr(frame, name, getattr(frame, name).clone())
                    frame.T_WC = self.lietorch.Sim3(frame.T_WC.data.clone())
                # Ito: a full map relocalises into its spare slot (see relocalization).
                spare = len(self.keyframes) >= self.capacity
                edges = self.graph.n_edges
                success = self.relocalization(frame, keep=not spare)
                if success:
                    self.relocalized += 1
                    if spare:
                        self.spare = edges
                if stream is not None:
                    stream.synchronize()
                if success:
                    states.set_mode(Mode.TRACKING)
                states.dequeue_reloc()
                return True
            return False

    def drop_spare(self):
        """A full map's spare keyframe goes when the camera is lost again, so the spare slot is
        free for the next relocalisation. Its edges are the graph's last (a full map adds no
        keyframe after it) and retrieval never held it."""
        if self.spare is None:
            # Ito: the usual loss, no spare keyframe. Checked without the backend's lock, which
            # the frontend would otherwise wait on for the backend's queued keyframes (0.3-1 s
            # on a fast turn): only the backend sets spare, during a relocalisation, before it
            # sets TRACKING, and the frontend calls this only while TRACKING.
            return
        with self.backend_lock:
            self._drop_spare()

    def _drop_spare(self):
        if self.spare is None:
            return
        k = len(self.keyframes) - 1
        self.graph.n_edges = self.spare
        self.keyframes.pop_last()
        self.retire_source(k)
        self.pending[k] = self.stale[k] = False
        self.shown = min(self.shown, k)
        self.spare = None

    def optimize(self, idx):
        """run_backend's work on one queued keyframe (main.py)."""
        config, keyframes, factor_graph = self.config, self.keyframes, self.graph
        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        self.check_retrieval(idx)
        retrieval_inds = self.retrieval.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        edges = factor_graph.n_edges
        if kf_idx:
            factor_graph.add_factors(kf_idx, frame_idx, config["local_opt"]["min_match_frac"])
        kept = factor_graph.ii2[2 * edges : 2 * factor_graph.n_edges : 2].tolist()
        self.loops += any(i < idx - LOOP_GAP for i in kept)

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

    def check_retrieval(self, idx):
        """Retrieval's next entry gets keyframe index idx: Ito takes its ids for indices."""
        if self.retrieval.kf_counter != idx:
            raise RuntimeError(
                f"MASt3R-SLAM's retrieval holds {self.retrieval.kf_counter} entries where keyframe "
                f"{idx} comes next: its ids no longer name keyframes"
            )

    def relocalization(self, frame, keep=True):
        """main.py's relocalization, as run_backend runs it for a frame in RELOC mode.

        keep=False (Ito, a full map): the relocalised frame becomes a keyframe in the spare slot
        but not a retrieval entry, as it goes again at the next loss (drop_spare) and
        retrieval's ids must stay keyframe indices. Upstream has no full map: its buffer
        overflows instead."""
        config, keyframes = self.config, self.keyframes
        factor_graph, retrieval_database = self.graph, self.retrieval
        # we are adding and then removing from the keyframe, so we need to be careful.
        # The lock slows viz down but safer this way...
        with keyframes.lock:
            self.check_retrieval(len(keyframes))
            kf_idx = []
            retrieval_inds = retrieval_database.update(
                frame,
                add_after_query=False,
                k=config["retrieval"]["k"],
                min_thresh=config["retrieval"]["min_thresh"],
            )
            kf_idx += retrieval_inds
            successful_loop_closure = False
            if kf_idx:
                keyframes.append(frame)
                n_kf = len(keyframes)
                kf_idx = list(kf_idx)  # convert to list
                frame_idx = [n_kf - 1] * len(kf_idx)
                edges = factor_graph.n_edges
                if factor_graph.add_factors(
                    frame_idx,
                    kf_idx,
                    config["reloc"]["min_match_frac"],
                    is_reloc=config["reloc"]["strict"],
                ):
                    if keep:
                        retrieval_database.update(
                            frame,
                            add_after_query=True,
                            k=config["retrieval"]["k"],
                            min_thresh=config["retrieval"]["min_thresh"],
                        )
                    successful_loop_closure = True
                    # Ito: from the best recalled keyframe the match kept. Upstream takes
                    # retrieval's first, which non-strict relocalisation may have dropped (the
                    # same one under strict: it keeps all or none).
                    seed = int(factor_graph.jj2[2 * edges])
                    keyframes.T_WC[n_kf - 1] = keyframes.T_WC[seed].clone()
                else:
                    keyframes.pop_last()

            if successful_loop_closure:
                if config["use_calib"]:
                    factor_graph.solve_GN_calib()
                else:
                    factor_graph.solve_GN_rays()
            return successful_loop_closure

    def place(self, camera):
        """The map's one gauge: upstream starts at the first keyframe's identity pose, Ito puts
        that pose where the robot says its camera is."""
        torch = self.torch
        self.world = np.asarray(camera, dtype=np.float64)
        device = self.axes.device
        self.world_rotation = torch.as_tensor(self.world[:3, :3], device=device).float()
        self.world_translation = torch.as_tensor(self.world[:3, 3], device=device).float()

    def placed(self, transform):
        """A Sim(3) camera pose of upstream's as a rigid world-from-camera pose in Ito's world."""
        matrix = transform.matrix()[0].double().cpu().numpy()
        if not np.isfinite(matrix).all():
            return None
        # Sim3 rotations drift from orthonormal in float32; snap to the nearest one.
        u, _, vh = np.linalg.svd(matrix[:3, :3])
        rigid = np.eye(4)
        rigid[:3, :3] = (u @ vh) * AXES[:, None] * AXES[None, :]
        rigid[:3, 3] = matrix[:3, 3] * AXES * self.scale
        return self.world @ rigid

    def pose(self, posed):
        """The frame's camera pose for the pilot (pose_ready hands it over before draw)."""
        pose = None if posed is None else self.placed(posed)
        self.lost = pose is None
        if pose is not None:
            self.camera_pose = pose.astype(np.float32)
            self.tracked += 1

    def publish(self, posed, now):
        """The status line, after draw (whose out-of-VRAM it reports)."""
        full = len(self.keyframes) >= self.capacity
        n = min(len(self.keyframes), self.capacity)
        owed = int(self.pending[: len(self.keyframes)].sum())
        counts = (
            f"{self.tracked} frames, {n}/{self.capacity} keyframes, {self.graph.n_edges} edges, "
            f"{self.loops} loop closures, {self.relocalized} relocalised, "
            f"{owed} keyframes owed splats, "
            + (
                f"{self.scale:.3f} m per map unit (ToF)"
                if self.tof.value
                else "unscaled (MASt3R units)"
            )
        )
        if self.lost:
            state = "lost: look back at a mapped place to relocalise"
            if full:
                state = "map full, " + state + "; nothing new is mapped"
        elif full:
            state = "map full: tracking, mapping nothing new (a larger Map size maps more)"
        else:
            state = "tracking"
        if owed and self.count - int(self.retiring.sum()) >= self.budget:
            state += (
                f"; splat budget full: {owed} keyframes not fully shown (raise Max splats, or "
                "free VRAM)"
            )
        warning = self.warning
        if now - self.oom_at < OOM_SHOWN:
            warning = (
                self.out_of_memory() if warning is None else f"{self.out_of_memory()}. {warning}"
            )
        if now - self.vram_full_at < OOM_SHOWN:
            full = (
                f"VRAM full ({self.vram_full_free / 2**20:.0f} MiB free): the map's least "
                "recently seen splats fade to make room"
                + ("" if self.warning else f"; {self.lower} in Settings")
            )
            warning = full if warning is None else f"{warning}. {full}"
        if warning:
            self.report(f"{warning}. MASt3R-SLAM {state}; {counts}")
        else:
            # The pilot shows any status but the plain "tracking |" one.
            self.report(f"MASt3R-SLAM {state} | {counts}")

    def draw(self, now):
        """Keep the splat map showing every keyframe's pointmap at its current pose: those
        upstream marked dirty (new, or refined by the tracker) and those the solve moved are
        drawn anew. A keyframe the splat pool could not take whole (full, or its old splats
        still fading), or that lost splats to the pool's pressure, stays pending: it is
        offered again, a few per frame while the pool has room, until all of it is shown."""
        # Ito: not while the backend thread holds the keyframes (a relocalisation holds them
        # for its retrieval, matching and solve, ~0.3 s): tracking would wait for it. Whatever
        # is due stays due (dirty, moved or new) and is drawn at the next frame. Held, the lock
        # covers only copying what this frame redraws; the splats are placed after it, so the
        # backend does not wait for them either.
        lock = self.keyframes.lock
        if not lock.try_enter():
            return
        try:
            n, redrawn, moving, retries = self.due()
        finally:
            lock.__exit__(None, None, None)
        began = time.perf_counter()
        for k, copy in redrawn:
            self.show(k, copy, now)
        # Moved keyframes, most moved first, while this frame's redraws took under
        # REDRAW_SECONDS (at least one): the rest are still moved, and come next frame.
        for i, (k, copy) in enumerate(moving):
            if i and time.perf_counter() - began > REDRAW_SECONDS:
                break
            self.stale[k] = True
            self.show(k, copy, now)
        if self.budget > self.count:
            for k, copy in retries:
                self.show(k, copy, now, retry=True)
        self.shown = n

    def due(self):
        """Under the keyframes' lock: which keyframes this frame redraws, each with a copy of
        what show needs (new and dirty ones, moved ones up to REDRAWS, pending retries)."""
        torch, lietorch, keyframes = self.torch, self.lietorch, self.keyframes
        n = len(keyframes)
        for k in self.evicted:
            self.pending[k] = k < n
        self.evicted.clear()
        pixels = self.footprint()
        if abs(pixels / self.drawn_pixels - 1) > 0.25:
            # Max splats (or the VRAM manager's limit) changed the splat size a lot: redraw.
            self.stale[:] = True
            self.drawn_pixels = pixels
        redraw = torch.zeros(n, dtype=torch.bool, device=self.drawn.device)
        dirty = keyframes.get_dirty_idx()
        redraw[dirty[dirty < n]] = True
        redraw[self.shown :] = True
        shown = self.shown
        moving = []
        if shown:
            drawn = lietorch.Sim3(self.drawn[:shown])
            poses = keyframes.T_WC[:shown].clone()
            delta = (drawn.inv() * lietorch.Sim3(poses)).data[:, 0]
            # How far each keyframe's points moved, over their depth: translation, rotation
            # and scale of the change in the keyframe's own frame.
            vector, w = delta[:, 3:6].norm(dim=1), delta[:, 6].abs()
            moved = (
                delta[:, :3].norm(dim=1) / self.depth[:shown]
                + 2 * torch.atan2(vector, w)
                + (delta[:, 7] - 1).abs()
            )
            stale = torch.as_tensor(self.stale[:shown], device=moved.device)
            moved = torch.nan_to_num(moved, nan=1.0)
            moved = torch.where(stale, moved + 1, moved)
            moved[redraw[:shown]] = 0
            due = moved > MOVED
            if due.any():
                most = torch.argsort(moved, descending=True)[: min(REDRAWS, int(due.sum()))]
                moving = most.tolist()
        redrawn = torch.nonzero(redraw).flatten().tolist()
        if redrawn:
            self.stale[redrawn] = True  # until shown: a redraw cut short by OOM is done later
        retries = []
        if self.budget > self.count:
            owed = set(np.flatnonzero(self.pending[:n]).tolist()) - set(redrawn) - set(moving)
            retries = sorted(owed)[:RETRIES]
        return (
            n,
            [(k, self.copy(k)) for k in redrawn],
            [(k, self.copy(k)) for k in moving],
            [(k, self.copy(k, retry=True)) for k in retries],
        )

    def copy(self, k, retry=False):
        """What show needs of keyframe k, copied under the keyframes' lock: its pose (where it
        was drawn, retrying), depth, confidence and colours."""
        torch, keyframes = self.torch, self.keyframes
        pose = (self.drawn[k] if retry else keyframes.T_WC[k]).clone()
        z = keyframes.X[k][:, 2].clone()
        confidence = keyframes.C[k][:, 0] / keyframes.N[k]
        colors = (keyframes.uimg[k].reshape(-1, 3).to(z.device) * 255).to(torch.uint8)
        return pose, z, confidence, colors

    def footprint(self):
        """A splat's size in pixels of the keyframe that places it: bigger for a smaller budget."""
        return 3 * max(1, (400_000 / self.budget) ** 0.5)

    def show(self, k, copy, now, retry=False):
        """Place keyframe k's points (copy, from due) as splats: anew (its old ones fade, and it
        may evict the longest-unseen to make room), or, retrying, what is missing of it where
        it was drawn, into free slots only."""
        torch = self.torch
        pose, z, confidence, colors = copy
        # Constrained to its rays, as use_calib does, and as upstream's viewer draws it.
        local = torch.stack((z * self.rays[0], z * self.rays[1], z), dim=1)
        valid = (confidence > CONFIDENCE) & (z > 0) & torch.isfinite(local).all(dim=1)
        if not retry:
            self.drawn[k] = pose
            self.stale[k] = False
            self.retire_source(k)
        self.pending[k] = False
        if not bool(valid.any()):
            return
        self.depth[k] = z[valid].median()
        points = self.lietorch.Sim3(pose).act(local) * self.axes * self.scale
        points = points @ self.world_rotation.T + self.world_translation
        pixels = self.footprint()
        footprint = z * pose[0, 7] * self.scale * pixels / float(self.input.K[0, 0])
        xp = self.xp
        self.pending[k] = True  # until placed: one cut short by OOM is offered again
        dropped = self.integrate_points(
            xp.from_dlpack(points[valid].contiguous()),
            xp.from_dlpack(colors[valid].contiguous()),
            now,
            xp.from_dlpack(footprint[valid].contiguous()),
            source=k,
            headroom=False,
            evict=not retry,
        )
        self.pending[k] = dropped > 0
