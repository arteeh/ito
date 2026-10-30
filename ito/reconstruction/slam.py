"""MASt3R-SLAM (Murai et al., CVPR 2025) as its own main.py runs it, in Ito's worker.

The bundled upstream (models/slam, ito.build) does the SLAM: FrameTracker tracks every frame
against the last keyframe and picks keyframes, the backend links each keyframe to the one
before and to what retrieval recalls and solves the global Sim(3) graph, and a lost camera
relocalises by retrieval, mapping nothing meanwhile. This module is main.py's loop, its
backend's work per keyframe and its relocalisation, ported line for line, in one process and
one thread: upstream's single_thread mode, in which the frontend waits for the backend after
every keyframe and every relocalisation attempt. Ito's own code is the edges: Input makes
frames as upstream's dataloader and create_frame do, and the camera pose and the keyframes'
pointmaps (at their current poses, as upstream's viewer draws them) go out to the pilot.
Where the robot has a ToF sensor, tof.RangeScale fixes once how many metres a map unit is, and
the pilot gets the map at that scale; the solve never sees it.
"""

import threading
from types import SimpleNamespace

import numpy as np

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
CONFIDENCE = 1.5  # upstream's viewer draws points above this average confidence
MOVED = 0.005  # a keyframe the solve moved by more than this share of its depth is redrawn
# Free VRAM after setup below this: the driver is already spilling into system memory.
SPILLING = 256 << 20


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
    return keyframes * KEYFRAME_BYTES_PER_PIXEL * w * h


def crop(image, long):
    """mast3r_utils.resize_img's 512 branch for any long side: resize the long side to it,
    centre-crop to multiples of 16. Returns the image and its return_transformation."""
    from mast3r_slam.mast3r_utils import _resize_pil_image

    W1, H1 = image.size
    image = _resize_pil_image(image, long)
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


class Manager:
    """What SharedKeyframes and SharedStates take from a multiprocessing Manager, for one
    process: upstream shares them between its frontend, backend and viewer processes."""

    @staticmethod
    def RLock():
        return threading.RLock()

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
        keyframes=256,
        precision_report=None,
        **options,
    ):
        from .mast3r_runtime import load_model, prepare

        source, weights, retrieval = prepare(report)
        import lietorch
        import torch
        from mast3r_slam.config import config, load_config
        from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates
        from mast3r_slam.global_opt import FactorGraph
        from mast3r_slam.mast3r_utils import load_retriever, mast3r_inference_mono
        from mast3r_slam.tracker import FrameTracker

        load_config(str(source / "config/base.yaml"))
        if 1 + config["retrieval"]["k"] > EDGES_PER_KEYFRAME:
            raise RuntimeError("MASt3R-SLAM's retrieval k outgrew Ito's graph edge buffer")
        # On top of base.yaml, what upstream's config/eval_calib.yaml sets: a calibrated camera,
        # and a frontend that waits for the backend.
        config["use_calib"] = True
        config["single_thread"] = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)
        options["device"] = "cuda"
        super().__init__(capacity, intrinsics, **options)
        self.report = report
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
        self.capacity, self.edges = keyframes, EDGES_PER_KEYFRAME * keyframes
        try:
            self.model = load_model(
                weights, report, precision=precision, precision_report=precision_report
            )
            K = torch.as_tensor(self.input.K, device="cuda")
            manager = Manager()
            self.keyframes = SharedKeyframes(manager, h, w, buffer=keyframes)
            self.states = SharedStates(manager, h, w)
            self.keyframes.set_intrinsics(K)
            self.tracker = FrameTracker(self.model, self.keyframes, "cuda")
            self.graph = FactorGraph(self.model, self.keyframes, K, "cuda", max_edges=self.edges)
            self.retrieval = load_retriever(self.model, str(retrieval))
            # Ito's own, also once: each keyframe's pose when last drawn, and its depth.
            self.drawn = torch.zeros(keyframes, 1, lietorch.Sim3.embedded_dim, device="cuda")
            self.depth = torch.ones(keyframes, device="cuda")
            free, _ = torch.cuda.mem_get_info()
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                f"Out of VRAM for MASt3R-SLAM at {self.settings}: {self.lower} in Settings"
            ) from exc
        self.warning = (
            f"VRAM full ({free / 2**20:.0f} MiB free after MASt3R-SLAM's setup): the driver "
            f"spills into system memory and everything slows; {self.lower}"
            if free < SPILLING
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
        self.full = False
        self.loops = 0
        self.relocalized = 0
        self.tof = RangeScale(self.input.K, (h, w))
        self.scale = 1.0  # metres per map unit, for display; 1 until the ToF fixes it
        self.ranged = None
        report("MASt3R-SLAM ready; waiting for camera")

    def bind_admission(self, admission):
        from .admission import bind_model, bind_native

        super().bind_admission(admission)
        self.admission_hooks = bind_model(self.model, admission)
        bind_native(admission)

    def integrate(self, rgb, depth, camera, now):
        """One frame through upstream's main loop. camera: the robot's own estimate of its camera
        pose (world-from-camera, Ito axes); only the first frame's is used, once, to place the
        map in the robot's gravity-aligned world. It never enters the solve."""
        before = len(self.keyframes), self.graph.n_edges, self.retrieval.kf_counter
        try:
            self.step(rgb, camera)
        except self.torch.cuda.OutOfMemoryError as exc:
            # The worker drops a frame that ran out of memory and goes on. Only tracking is
            # safe to drop: half a keyframe's backend work leaves upstream's keyframes, graph
            # and retrieval out of step, so that ends this map.
            if (len(self.keyframes), self.graph.n_edges, self.retrieval.kf_counter) == before:
                raise
            raise RuntimeError(
                f"Out of VRAM in MASt3R-SLAM's backend at {self.settings}: {self.lower} in Settings"
            ) from exc
        if self.ranged is not None and self.tof.value is None and self.measure(self.ranged):
            self.scale = self.tof.value
            self.shown = 0  # every keyframe again, at the new scale
        self.publish(self.posed)
        self.draw(now)

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
            self.lietorch.Sim3.Identity(1, device="cuda")
            if self.i == 0
            else states.get_frame().T_WC
        )
        frame = self.input.frame(self.i, rgb, T_WC)
        posed = self.ranged = None
        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = self.inference(self.model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            self.optimize(len(keyframes) - 1)
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
                    states.set_mode(Mode.RELOC)
                else:
                    posed = frame.T_WC
                    self.ranged = frame
                states.set_frame(frame)
            elif mode == Mode.RELOC:
                X, C = self.inference(self.model, frame)
                frame.update_pointmap(X, C)
                states.set_frame(frame)
                # Ito: a relocalisation attempt is a keyframe until it fails; none when full.
                if self.room(self.config["retrieval"]["k"]) and self.relocalization(
                    states.get_frame()
                ):
                    states.set_mode(Mode.TRACKING)
                    self.relocalized += 1
                    posed = self.lietorch.Sim3(keyframes.T_WC[len(keyframes) - 1])
            else:
                raise RuntimeError(f"Invalid MASt3R-SLAM mode {mode}")
            if add_new_kf and self.room(1 + self.config["retrieval"]["k"]):
                keyframes.append(frame)
                self.optimize(len(keyframes) - 1)
        self.i += 1
        self.posed = posed

    def room(self, edges):
        """Upstream's buffers would overflow (IndexError) past their size: Ito adds no keyframe
        then, and says so."""
        self.full = len(self.keyframes) >= self.capacity or self.graph.n_edges + edges > self.edges
        return not self.full

    def optimize(self, idx):
        """run_backend's work on one queued keyframe (main.py), which single_thread waits for."""
        config, keyframes, factor_graph = self.config, self.keyframes, self.graph
        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = self.retrieval.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            self.loops += 1

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(kf_idx, frame_idx, config["local_opt"]["min_match_frac"])

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

    def relocalization(self, frame):
        """main.py's relocalization, as run_backend runs it for a frame in RELOC mode."""
        config, keyframes = self.config, self.keyframes
        factor_graph, retrieval_database = self.graph, self.retrieval
        # we are adding and then removing from the keyframe, so we need to be careful.
        # The lock slows viz down but safer this way...
        with keyframes.lock:
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
                if factor_graph.add_factors(
                    frame_idx,
                    kf_idx,
                    config["reloc"]["min_match_frac"],
                    is_reloc=config["reloc"]["strict"],
                ):
                    retrieval_database.update(
                        frame,
                        add_after_query=True,
                        k=config["retrieval"]["k"],
                        min_thresh=config["retrieval"]["min_thresh"],
                    )
                    successful_loop_closure = True
                    keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
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
        self.world_rotation = torch.as_tensor(self.world[:3, :3], device="cuda").float()
        self.world_translation = torch.as_tensor(self.world[:3, 3], device="cuda").float()

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

    def publish(self, posed):
        pose = None if posed is None else self.placed(posed)
        self.lost = pose is None
        if pose is not None:
            self.camera_pose = pose.astype(np.float32)
            self.tracked += 1
        n = len(self.keyframes)
        counts = (
            f"{self.tracked} frames, {n}/{self.capacity} keyframes, {self.graph.n_edges} edges, "
            f"{self.loops} loop closures, {self.relocalized} relocalised, "
            + (
                f"{self.scale:.3f} m per map unit (ToF)"
                if self.tof.value
                else "unscaled (MASt3R units)"
            )
        )
        if self.full:
            state = f"map full at {n} keyframes: tracking on, mapping nothing new"
        elif self.lost:
            state = "lost: look back at a mapped place to relocalise"
        else:
            state = "tracking"
        if self.warning:
            self.report(f"{self.warning}. MASt3R-SLAM {state}; {counts}")
        else:
            # The pilot shows any status but the plain "tracking |" one.
            self.report(f"MASt3R-SLAM {state} | {counts}")

    def draw(self, now):
        """Keep the splat map showing every keyframe's pointmap at its current pose: those
        upstream marked dirty (new, or refined by the tracker) and those the solve moved."""
        torch, lietorch, keyframes = self.torch, self.lietorch, self.keyframes
        n = len(keyframes)
        redraw = torch.zeros(n, dtype=torch.bool, device="cuda")
        dirty = keyframes.get_dirty_idx()
        redraw[dirty[dirty < n]] = True
        redraw[self.shown :] = True
        shown = self.shown
        if shown:
            drawn = lietorch.Sim3(self.drawn[:shown])
            delta = (drawn.inv() * lietorch.Sim3(keyframes.T_WC[:shown])).data[:, 0]
            # How far each keyframe's points moved, over their depth: translation, rotation
            # and scale of the change in the keyframe's own frame.
            vector, w = delta[:, 3:6].norm(dim=1), delta[:, 6].abs()
            moved = (
                delta[:, :3].norm(dim=1) / self.depth[:shown]
                + 2 * torch.atan2(vector, w)
                + (delta[:, 7] - 1).abs()
            )
            redraw[:shown] |= ~(moved <= MOVED)
        for k in torch.nonzero(redraw).flatten().tolist():
            self.show(k, now)
        self.shown = n

    def show(self, k, now):
        torch, keyframes = self.torch, self.keyframes
        with keyframes.lock:
            pose = keyframes.T_WC[k].clone()
            z = keyframes.X[k][:, 2].clone()
            confidence = keyframes.C[k][:, 0] / keyframes.N[k]
            colors = (keyframes.uimg[k].reshape(-1, 3).to("cuda") * 255).to(torch.uint8)
        # Constrained to its rays, as use_calib does, and as upstream's viewer draws it.
        local = torch.stack((z * self.rays[0], z * self.rays[1], z), dim=1)
        valid = (confidence > CONFIDENCE) & (z > 0) & torch.isfinite(local).all(dim=1)
        self.drawn[k] = pose
        self.retire_source(k)
        if not bool(valid.any()):
            return
        self.depth[k] = z[valid].median()
        points = self.lietorch.Sim3(pose).act(local) * self.axes * self.scale
        points = points @ self.world_rotation.T + self.world_translation
        pixels = 3 * max(1, (400_000 / self.budget) ** 0.5)
        footprint = z * pose[0, 7] * self.scale * pixels / float(self.input.K[0, 0])
        xp = self.xp
        self.integrate_points(
            xp.from_dlpack(points[valid].contiguous()),
            xp.from_dlpack(colors[valid].contiguous()),
            now,
            xp.from_dlpack(footprint[valid].contiguous()),
            source=k,
        )
