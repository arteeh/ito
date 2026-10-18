import math
import os

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from ito import clock

from .align import to_map
from .depth import (
    MapFrontend,
    MonoGeometry,
    MultiViewDepth,
    PromptedDepth,
    log_field,
    nearest_views,
    resample,
    scale_to,
)
from .fusion import UP, PoseGraph
from .gaussian import EXPOSURE_INPUT, GaussianBackend, exposed
from .motion import MotionBuffer
from .sensors import HeadSensors
from .slam_vision import RESTART_CAUSES

__all__ = ["RESTART_CAUSES", "SLAMBackend"]

# A frame smeared further than this by the camera's turn during its exposure still gets a
# pose, from the robot, but neither a visual factor nor splats: no tracker matches through
# it, and its splats would be streaks. Blur is |angular velocity| x exposure x focal length.
BLUR_PIXELS = 8.0
EXPOSURE = 0.010
DELAY_RANGE = (-0.1, 0.2)
FLOOR_RAY = math.radians(10)
FLOOR_SHARE = 0.1
FLOOR_HEIGHT = 0.05
FLOOR_LEVEL = 0.05
# The camera rides ~20 cm up and sees the floor far off: a degree of tilt error in the
# kinematic up (the head's gear play is two in the sim) moves far floor points centimetres
# and the floor's scale by percent (about 6% high, measured); the ToF reads range directly.
FLOOR_SIGMA = 0.4
RANGE_SIGMA = 0.05
SCALE_STEP = 0.01
# ITO_DA3_EVERY=N: DA3 on every Nth frame with depth, MASt3R's depth between (with ITO_FUSE the
# field holds both, DA3's weighed ITO_DA3_TRUST times: it is about a third as far off).
DA3_EVERY = int(os.environ.get("ITO_DA3_EVERY", "1"))
DA3_TRUST = float(os.environ.get("ITO_DA3_TRUST", "4"))
REFERENCES = int(os.environ.get("ITO_DA3_VIEWS", "2"))
# ITO_DA3_CAMS=1: DA3 is told each view's pose and the rectified view's intrinsics (its camera
# tokens); offline on shelf-close its shape error fell from 5.2% to 4.5%.
DA3_CAMS = os.environ.get("ITO_DA3_CAMS", "0") != "0"  # retained views seen with each new frame
VIEWS_SIGMA = 0.05  # log-scale: a retained view's metric depth, seen in the same call
SCALE_SIGMAS = dict(tof=RANGE_SIGMA, floor=FLOOR_SIGMA)
# A frame laid onto the map's surfaces (align.py) tells fusion where the camera was this well.
MAP_SIGMA = np.repeat((math.radians(0.5), 0.01), 3)
SILHOUETTE = 0.1  # depth span over 5x5 pixels, of depth, that marks a silhouette's ramp
# ITO_TRUTH=compare (a diagnostic): map the network's depth, but log its scale against the
# simulator's exact depth in the timeline.
COMPARE = os.environ.get("ITO_TRUTH") == "compare"
# With compare, ITO_TRUE_SCALE=1 (a diagnostic) maps each frame's estimated depth at the scale
# of the sim's exact depth: the estimate's shape and poses, a perfect per-frame scale.
TRUE_SCALE = os.environ.get("ITO_TRUE_SCALE", "0") != "0"
# ITO_DEPTH=moge maps MoGe-2's single-image depth (depth.py MonoGeometry) at MASt3R's metric
# scale for the frame: its walls and silhouettes, MASt3R's scale and tracking. ITO_DEPTH=hybrid
# keeps MASt3R's depth wherever it is within HYBRID_AGREE of MoGe-2's, MoGe-2's elsewhere.
HYBRID_AGREE = 0.08
# Experiments, developer-only: ITO_DEPTH=da3 maps depth.py's multi-view depth, each frame
# laid onto the map's surfaces (align.py), instead of MASt3R's pairwise depth. ITO_TRACKER=map
# also drops the MASt3R tracker: frames are placed by the robot's motion and the map alone.
TRACKER = os.environ.get("ITO_TRACKER", "mast3r")
DEPTH = "da3" if TRACKER == "map" else os.environ.get("ITO_DEPTH", "mast3r")
# ITO_ALIGN=1 (always with ITO_TRACKER=map) lays each frame's depth onto the map's surfaces
# (align.py): where it was placed tells fusion where the camera was.
ALIGN = TRACKER == "map" or os.environ.get("ITO_ALIGN", "0") != "0"
# ITO_SCALE=graph: a frame's depth in metres by the scale fusion keeps over many ToF and floor
# readings, not by the one this frame read (about 7% apart from frame to frame in the sim);
# with it, registration lays a frame onto retained views without rescaling it to them, so
# the first views' one-frame scale stops setting the whole map's.
GRAPH_SCALE = os.environ.get("ITO_SCALE", "frame") == "graph"
# ITO_MAPSCALE=1: a frame whose view the map already covers takes the map's scale: its depth
# is scaled by the median of the map's drawn depth over its own there. A frame's metric scale
# from ToF and floor scatters by several percent, and a pot seen at 2.4 m from one frame and
# again later lands twice, 15 cm apart; the map's own scale stays one scale.
MAPSCALE = os.environ.get("ITO_MAPSCALE", "0") != "0"
# ITO_MAPSCALE=field: instead of one ratio, a smooth field (depth.py log_field) takes the frame's
# depth onto the map's where the map covers it, and holds still where it does not: a frame
# agrees with the surfaces already mapped, so a revisit lays nothing a second time.
MAPFIELD = os.environ.get("ITO_MAPSCALE") == "field"
MAPFIELD_GRID = (3, 4)
MAPFIELD_COVERED = 0.1
MAPFIELD_PRIOR = 0.05
# ITO_MONO_FIELD=1: MoGe-2's depth (ITO_DEPTH=moge) goes onto MASt3R's metric depth through a
# smooth field (2 x 2 nodes), not one ratio: MoGe-2 holds shape but bends near against far.
MONO_FIELD = os.environ.get("ITO_MONO_FIELD", "0") != "0"
MONO_GRID = (2, 2)
# ITO_MONO_SCALE=1: MoGe-2's depth takes its metric scale from the ToF and the floor read on its
# own surfaces, MASt3R's only beside them (MONO_MAST3R, log sigma).
MONO_SCALE = os.environ.get("ITO_MONO_SCALE", "0") != "0"
MONO_MAST3R = 0.1
# ITO_TOF_VIA=moge: MASt3R's depth stays the map's, but the ToF reads its scale on MoGe-2's
# surfaces (ranging_surface). On MASt3R's own, the reading jumped 10-20% for seconds after the
# first turn and close to the shelf (its centre bends against its edges); on MoGe-2's it held.
TOF_VIA_MOGE = os.environ.get("ITO_TOF_VIA") == "moge"
# Nothing is mapped until fusion has this many ToF or floor scale readings (ITO_SCALE_SETTLED=0:
# from the first frame): the first seconds' scale swung from 7% short to 10% long in the sim,
# and what they mapped stays.
SCALE_SETTLED = int(os.environ.get("ITO_SCALE_SETTLED", "25"))
# ITO_SETTLE_RANGES=N: with a well-shaped depth (ITO_DEPTH=moge|da3) the ToF reads the scale to a
# few percent and the floor alone does not (10% off on MoGe-2's surfaces), so mapping waits for N
# ToF readings, or for 4 x SCALE_SETTLED tracked frames on a robot whose ToF never answers.
SETTLE_RANGES = int(os.environ.get("ITO_SETTLE_RANGES", "0"))
MAPSCALE_COVERED = 0.3
# ITO_MAP_AGREE=T: a frame maps nothing where the map already shows a surface (opacity over 0.9)
# more than T (log depth) from its own depth. Seen again from elsewhere, an object mapped once
# landed a second time beside itself: the new view's depth at its edge was off, not the object.
MAP_AGREE = float(os.environ.get("ITO_MAP_AGREE", "0"))
# ITO_MAP_MARGIN=F: map no depth from the outer F of the frame on each side. The plant pot, seen at
# the edge of view while the robot walked up to the shelf and looked along it, landed beside
# itself; at the edge of the rectified wide view every depth network is least sure.
MARGIN = float(os.environ.get("ITO_MAP_MARGIN", "0"))
# Only where the map's Gaussians there are finer than MAP_AGREE_WIDTH pixels of this view: a frame
# much closer than the views that mapped a surface sees it in more detail and still maps it.
MAP_AGREE_WIDTH = float(os.environ.get("ITO_MAP_AGREE_WIDTH", "1.5"))
MAPSCALE_RANGE = (0.85, 1.15)
# ITO_REFINE=1 lays each frame onto the map photometrically before mapping it (training.py
# FrameAlign) and tells fusion where it landed.
REFINE = os.environ.get("ITO_REFINE", "0") != "0"
# Map only depth another view agrees on (consensus.py); ITO_CONSENSUS=0 maps every guess.
CONSENSUS = os.environ.get("ITO_CONSENSUS", "1") != "0"
# ITO_FUSE=1: every frame's depth is fused into a signed distance field (volume.py) and the
# frame is mapped by the field's depth along its rays; it takes consensus's place.
# ITO_TOF_MAP=1: the ToF's readings go into the fused field (and turn it on) as measured empty
# space: a VL53 reports the nearest target anywhere in a zone, so nothing lies in the zone's
# footprint closer than its reading. Every valid zone of every grid clears that space (with
# ITO_TOF_MAP_WEIGHT per grid), short of the reading by TOF_MARGIN; a guessed surface there,
# like the pot a rim view put a third too near, is refuted and its splats retire.
TOF_MAP = os.environ.get("ITO_TOF_MAP", "0") != "0"
TOF_MAP_WEIGHT = float(os.environ.get("ITO_TOF_MAP_WEIGHT", "1"))
TOF_SPREAD = 1.0  # middle share of a zone's width and height cleared
TOF_MARGIN = (0.03, 0.05)  # metres plus a share of the reading left uncleared before it
TOF_NEAR = 0.12  # metres: the sensor's own minimum range
FUSE = TOF_MAP or os.environ.get("ITO_FUSE", "0") != "0"
# ITO_TOF_FIELD=1: bend each frame's depth to the ToF's zone readings where they land
# (rangefield.py), not only scale the whole frame by them.
TOF_FIELD = os.environ.get("ITO_TOF_FIELD", "0") != "0"
# ITO_TOF_ZONES=nearest: a zone's reading is the nearest target anywhere in its footprint, as a
# real VL53 (and the sim with MICRODUCK_TOF_FOOTPRINT) reports it; default "centre": the range
# along the zone's centre ray, as the sim's one ray per zone gives it.
TOF_ZONES = os.environ.get("ITO_TOF_ZONES", "centre")
TRUST_UNREAD = 0.3  # fusion's trust in a frame no ToF zone read, with the field on
# ITO_DUMP=<dir> (a diagnostic): each mapped frame's image, depth as estimated and as agreed,
# the sim's exact depth when ITO_TRUTH gives it, and its pose, one npz per frame.
DUMP = os.environ.get("ITO_DUMP")
# Retained views keep the camera's own frame at DETAIL times the network's grid (512 across)
# for photometric training, which then refines splats finer than the network sees
# (ITO_DETAIL=1: the network's own grid).
DETAIL = float(os.environ.get("ITO_DETAIL", "2"))


class SLAMBackend(GaussianBackend):
    """frame_delay: the driver's estimate of how long before its capture_time a frame was
    exposed; exposure: how long the sensor integrates one, None if the driver cannot say.
    aligner: a tracking.MapAligner, the map's say in every tracked frame's pose."""

    def __init__(
        self,
        capacity,
        intrinsics,
        *,
        report,
        motion_noise,
        frame_delay=0.0,
        exposure=None,
        frontend=None,
        precision="full",
        input_resolution=100,
        precision_report=None,
        aligner=None,
        **options,
    ):
        if frontend is None and TRACKER == "map":
            frontend = MapFrontend(intrinsics, input_resolution)
        if frontend is None:
            from .mast3r_frontend import MASt3RFrontend

            frontend = MASt3RFrontend(
                intrinsics,
                report,
                precision=precision,
                input_resolution=input_resolution,
                precision_report=precision_report,
            )
        options["device"] = "cuda"
        super().__init__(capacity, intrinsics, precision=precision, **options)
        self.report = report
        self.frontend = frontend
        self.aligner = aligner
        self.intrinsics = intrinsics
        self.aligned = 0
        self.motion = MotionBuffer(motion_noise, seconds=60.0)
        self.sensors = HeadSensors(seconds=60.0)
        bounds = (max(0.0, frame_delay + DELAY_RANGE[0]), frame_delay + DELAY_RANGE[1])
        self.graph = PoseGraph(
            self.motion, td=frame_delay, td_bounds=bounds, estimate_td=True, inertial=self.sensors
        )
        self.focal = float(intrinsics.fx)
        self.exposure = EXPOSURE if exposure is None else float(exposure)
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.correction = np.eye(4, dtype=np.float32)
        self.tracked = 0
        self.matched = 0
        self.blurred = 0
        self.restarts = dict.fromkeys(RESTART_CAUSES, 0)
        self.lost = False
        self.unmatched_since = None
        self.recalls = 0
        self.waiting = False
        self.retired_keyframes = 0
        self.truth = None
        self.pose_error = None
        self.scale_readings = 0
        self.range_readings = 0
        self.map_conflicts = 0
        self.depth = MultiViewDepth() if DEPTH == "da3" else None
        self.mono = MonoGeometry() if DEPTH in ("moge", "hybrid") or TOF_VIA_MOGE else None
        # ITO_DEPTH=dmdc: dToF-DMDC's metric depth, prompted with the ToF's readings.
        self.prompted = PromptedDepth() if DEPTH == "dmdc" else None
        self.refine = None
        if REFINE:
            from .training import FrameAlign

            self.refine = FrameAlign(self)
        self.consensus = None
        self.range_field = None
        self.range_trust = None
        if TOF_FIELD:
            from .rangefield import RangeField

            self.range_field = RangeField()
        self.volume = None
        if FUSE:
            from .volume import Volume

            self.volume = Volume()
        elif CONSENSUS:
            from .consensus import Consensus

            self.consensus = Consensus()
        self.depth_scales = {}
        self.placed = None
        from .registration import Registration

        self.registration = Registration()
        report("MASt3R ready; waiting for camera and robot motion")

    def bind_admission(self, admission):
        super().bind_admission(admission)
        if hasattr(self.frontend, "bind_admission"):
            self.frontend.bind_admission(admission)

    def add_sensors(self, imu, ranges):
        self.sensors.add(imu, ranges)

    def add_motion(self, rows):
        self.motion.extend(rows)

    def integrate(self, rgb, depth, camera, now, measured=False, *, captured, truth=None):
        """captured: the frame's stamp on the pilot clock, the clock motion samples are on.
        truth: (world-from-camera or None, exposure on the pilot clock, axial depth or None)
        to publish and map this frame by instead of the estimates; the estimator still runs,
        unchanged, beside them."""
        started = clock.now()
        if EXPOSURE_INPUT:
            self.input_exposure_used = getattr(self, "input_exposure", np.zeros(3)).copy()
            rgb = exposed(rgb, self.input_exposure_used)
        self.truth = truth
        self.from_da3 = False
        graph = self.graph
        exposure = captured - graph.td
        if not self.motion.covers(exposure):
            if not self.waiting:
                self.report("Robot-first tracking waiting for the robot's motion")
            self.waiting = True
            return
        self.waiting = False
        propagated = graph.predict(captured) if graph.latest() is not None else None
        expired = graph.expire(self.motion.times[0])
        self.retired_keyframes += len(expired)
        self.frontend.forget(expired)
        if expired and not graph.keyframes:
            graph.anchor = propagated.copy()
        node = graph.add_frame(
            captured, pose=propagated if expired and not graph.keyframes else None
        )
        predicted = graph.pose(node)
        last = graph.latest_keyframe()
        reference_time = graph.nodes[last].time if last is not None else captured
        _, covariance = self.motion.between(reference_time - graph.td, exposure)
        gyro = self.sensors.angular_velocity(exposure)
        if gyro is None:
            gyro = self.motion.angular_velocity(exposure)
        spin = float(np.linalg.norm(gyro))
        if spin * self.exposure * self.focal > BLUR_PIXELS:
            self.blurred += 1
            graph.solve_frame()
            self._publish(node)
            return
        result = self.frontend.track(rgb, node, predicted, np.sqrt(np.diag(covariance)))
        cloud = result.cloud
        if result.relative is not None:
            graph.add_visual(result.reference, node, result.relative, result.sigma)
            graph.solve_delay()
            self.matched += 1
        measured_scale = None
        if cloud is not None:
            robot = self.motion.at(exposure)
            seen = []
            below = floor_scale(cloud, robot[:3, :3].T @ UP, float(robot[1, 3]))
            if below is not None:
                seen.append((math.log(below), FLOOR_SIGMA))
            ranged = self.sensors.range_scale(
                self.ranging_surface(cloud, rgb), exposure, self.motion
            )
            if ranged is not None:
                seen.append((math.log(ranged), RANGE_SIGMA))
            if COMPARE:
                # Each scale source beside the one exact depth would give, metres per unit.
                self.depth_scales = dict(floor=below, tof=ranged)
                if truth is not None and truth[2] is not None:
                    unscaled = -cloud.points[:, 2].reshape(cloud.shape).float().cpu().numpy()
                    valid = cloud.valid.reshape(cloud.shape).cpu().numpy()
                    self.depth_scales["exact"] = scale_to(truth[2], np.where(valid, unscaled, 0))
            self.range_readings += ranged is not None
            for value, sigma in seen:
                graph.add_scale(value, sigma)
                self.scale_readings += 1
            if seen:
                graph.solve_scale()
                weights = [sigma**-2 for _, sigma in seen]
                measured_scale = math.exp(
                    sum(w * value for w, (value, _) in zip(weights, seen, strict=True))
                    / sum(weights)
                )
        graph.solve_frame()
        if self.aligner is not None:
            found = self.aligner.align(rgb, graph.pose(node), self.intrinsics)
            if found is not None and found[2] > 0 and graph.add_map(node, *found[:2]):
                graph.solve_frame()
                self.aligned += 1
        poses, dropped = {}, []
        if result.keyframe:
            dropped = graph.promote()
            poses = graph.solve_window()
        scale = 1.0
        if poses or abs(graph.log_scale) > SCALE_STEP:
            scale = graph.fold_scale()
            self.frontend.correct(poses, scale)
        if truth is None or truth[0] is None:
            self.revise(poses)
        if dropped:
            self.retired_keyframes += len(dropped)
            self.frontend.forget(dropped)
        if cloud is not None:
            cloud = cloud._replace(
                points=cloud.points
                * (scale if measured_scale is None or GRAPH_SCALE else measured_scale)
            )
            if truth is not None and truth[2] is not None and not COMPARE:
                cloud = self.with_depth(cloud, truth[2])
            elif self.depth is not None and self.depth_turn():
                cloud = self.consistent(cloud, graph.pose(node), exposure)
                self.from_da3 = True
            elif self.prompted is not None:
                cloud = self.prompted_depth(cloud, rgb, exposure)
            elif self.mono is not None:
                cloud = self.monocular(cloud, rgb, exposure)
            if self.range_field is not None and not (
                truth is not None and truth[2] is not None and not COMPARE
            ):
                cloud = self.bend_to_ranges(cloud, exposure)
            if COMPARE and TRUE_SCALE and truth is not None and truth[2] is not None:
                unscaled = -cloud.points[:, 2].reshape(cloud.shape).float().cpu().numpy()
                valid = cloud.valid.reshape(cloud.shape).cpu().numpy()
                exact = scale_to(truth[2], np.where(valid, unscaled, 0))
                if exact is not None:
                    cloud = cloud._replace(points=cloud.points * exact)
            if COMPARE and truth is not None and truth[2] is not None:
                depth = -cloud.points[:, 2].reshape(cloud.shape).float().cpu().numpy()
                valid = cloud.valid.reshape(cloud.shape).cpu().numpy()
                self.depth_scales["true"] = scale_to(truth[2], np.where(valid, depth, 0))
        if cloud is None or self.range_field is None:
            self.range_trust = None
        self.placed = None
        if cloud is not None and ALIGN and (truth is None or truth[0] is None):
            rows, columns = cloud.shape
            mapping = self.registration.map_from_estimator
            self.placed = to_map(
                self,
                cloud.points.reshape(rows, columns, 3).float().cpu().numpy(),
                cloud.valid.reshape(rows, columns).cpu().numpy(),
                mapping @ graph.pose(node),
                cloud.intrinsics,
            )
            if self.placed is not None and graph.add_map(
                node, np.linalg.inv(mapping) @ self.placed, MAP_SIGMA
            ):
                graph.solve_frame()
                self.aligned += 1
        pose = self._publish(node)
        settled = self.scale_readings >= SCALE_SETTLED or self.tracked > 4 * SCALE_SETTLED
        if SETTLE_RANGES:
            settled = self.range_readings >= SETTLE_RANGES or self.tracked > 4 * SCALE_SETTLED
        exact = truth is not None and truth[2] is not None and not COMPARE
        if cloud is not None and (settled or exact):
            self.place(cloud, pose, now + clock.now() - started, node=node, rgb=rgb)

    def depth_turn(self):
        self.depth_frames = getattr(self, "depth_frames", -1) + 1
        return self.depth_frames % DA3_EVERY == 0

    def consistent(self, cloud, pose, exposure):
        """The cloud with depth seen together with the retained views nearest it. Its scale
        weighs their metric depth with the ToF's ranges and the floor below the robot's camera,
        so the map stays metric; with none of them, MASt3R's."""
        self.depth_scales = {}
        rows, columns = cloud.shape
        image = cloud.colors.reshape(rows, columns, 3).cpu().numpy()
        references = nearest_views(
            self.views, self.registration.map_from_estimator @ pose, REFERENCES
        )
        cameras = None
        if DA3_CAMS:
            cameras = [(self.registration.map_from_estimator @ pose, cloud.intrinsics)]
            cameras += [(view.pose, view.intrinsics) for view in references]
        with self.gpu_chunk("depth"):
            depths = self.depth([image] + [view.image for view in references], cameras)
        depth = depths[0]
        # Learned depth ramps across a silhouette; what lies on the ramp is neither surface.
        span = maximum_filter(depth, size=5) - minimum_filter(depth, size=5)
        depth = np.where(span > SILHOUETTE * depth, 0, depth)
        unit = self.with_depth(cloud, depth)
        robot = self.motion.at(exposure)
        found = dict(
            views=[
                s
                for view, d in zip(references, depths[1:], strict=True)
                if (s := scale_to(view.depth, d)) is not None
            ],
            tof=self.sensors.range_scale(unit, exposure, self.motion),
            floor=floor_scale(unit, robot[:3, :3].T @ UP, float(robot[1, 3])),
        )
        evidence = [(math.log(s), VIEWS_SIGMA) for s in found["views"]]
        evidence += [(math.log(found[k]), s) for k, s in SCALE_SIGMAS.items() if found[k]]
        if not evidence:
            mast3r = -cloud.points[:, 2].reshape(rows, columns).float().cpu().numpy()
            fallback = scale_to(
                np.where(cloud.valid.reshape(rows, columns).cpu().numpy(), mast3r, 0), depths[0]
            )
            evidence = [(math.log(fallback), 1.0)] if fallback else []
        if not evidence:
            return cloud
        weights = [sigma**-2 for _, sigma in evidence]
        scale = math.exp(
            sum(w * v for w, (v, _) in zip(weights, evidence, strict=True)) / sum(weights)
        )
        self.depth_scales = found | dict(scale=scale)
        if COMPARE:
            if found["views"]:
                self.depth_scales["da3_views"] = float(np.median(found["views"]))
            self.depth_scales |= {f"da3_{k}": found[k] for k in ("tof", "floor") if found[k]}
            self.depth_scales["da3_scale"] = scale
            if self.truth is not None and self.truth[2] is not None:
                self.depth_scales["da3_exact"] = scale_to(self.truth[2], depth)
        return self.with_depth(cloud, depth * scale)

    def clear_ranges(self):
        """Every ToF grid since the last frame, up to this frame's capture, clears the space
        before its zones' readings in the fused field (TOF_MAP)."""
        done = getattr(self, "ranges_cleared", -math.inf)
        td = self.graph.td
        for grid in [
            g for g in self.sensors.grids if done < g["time"] <= self.source_captured - td
        ]:
            # Grid times are on the motion's clock, as exposures are: a capture stamp less td.
            camera = self.pose_at(grid["time"] + td)
            self.ranges_cleared = grid["time"]
            if camera is None:
                continue
            distances = np.array(
                [np.nan if d is None else d for d in grid["distances"]], np.float32
            ).reshape(grid["shape"])
            self.volume.clear(
                camera @ grid["mount"],
                grid["field_of_view"],
                distances * (1 - TOF_MARGIN[1]) - TOF_MARGIN[0],
                TOF_MAP_WEIGHT,
                TOF_NEAR,
                TOF_SPREAD,
            )

    def carve(self, camera, intrinsics, shape):
        """Splats in view that the fused field holds surely empty gather contradictions, as
        splats a view sees through do, and retire."""
        ids, _, _, _ = self.centers(camera, intrinsics, shape)
        if not len(ids):
            return
        centres = _torch().as_tensor(self.records[ids, 0, :3], device="cuda")
        hit = ids[self.volume.free(centres).cpu().numpy()]
        if not len(hit):
            return
        self.contradictions[hit] = np.minimum(self.contradictions[hit].astype(int) + 1, 255)
        limit = np.where(self.support[hit] >= 10, 6, 3)
        gone = hit[self.contradictions[hit] >= limit]
        if len(gone):
            self.captures[gone] = self.source_captured
            self.retire(gone)
            self.volume.carved += len(gone)

    def prompted_depth(self, cloud, rgb, exposure):
        """The cloud with dToF-DMDC's metric depth for the frame (at 640 x 360), prompted with
        the ToF zones' readings at their centre pixels."""
        size = (640, 360)
        i = self.intrinsics
        s = size[0] / i.width
        grid = (i.fx * s, i.fy * s, (i.cx + 0.5) * s - 0.5, (i.cy + 0.5) * s - 0.5)
        hits = self.sensors.zone_hits(exposure, self.motion, grid, (size[1], size[0]))
        fov = math.degrees(2 * math.atan(i.width / 2 / i.fx))
        with self.gpu_chunk("depth"):
            depth = self.prompted(rgb, fov, size, hits)
        if COMPARE:
            self.depth_scales["dmdc_zones"] = len(hits)
        if not (depth > 0).any():
            return cloud
        # Even a prompted network ramps across a silhouette over a pixel or two.
        span = maximum_filter(depth, size=3) - minimum_filter(depth, size=3)
        return self.with_depth(cloud, np.where(span > SILHOUETTE * depth, 0, depth))

    def bend_to_ranges(self, cloud, exposure):
        """The cloud with its depth bent to the ToF zones that land on it (rangefield.py)."""
        rows, columns = cloud.shape
        valid = cloud.valid.reshape(rows, columns).cpu().numpy()
        depth = np.where(
            valid, -cloud.points[:, 2].reshape(rows, columns).float().cpu().numpy(), 0
        ).astype(np.float32)
        image = cloud.colors.reshape(rows, columns, 3).cpu().numpy()
        if TOF_ZONES == "nearest":
            footprints = self.sensors.zone_footprints(
                exposure, self.motion, cloud.intrinsics, cloud.shape
            )
            hits = self.range_field.nearest(depth, footprints, cloud.intrinsics)
        else:
            hits = self.sensors.zone_hits(exposure, self.motion, cloud.intrinsics, cloud.shape)
        self.zone_hits = hits
        with self.gpu_chunk("depth"):
            bent, used, self.range_trust = self.range_field.correct(depth, image, hits)
        if COMPARE:
            self.depth_scales["tof_field"] = used
        if not used:
            return cloud
        torch = _torch()
        factor = np.where(depth > 0, bent / np.maximum(depth, 1e-6), 0).astype(np.float32)
        factor = torch.as_tensor(factor.reshape(-1, 1), device=cloud.points.device)
        return cloud._replace(points=cloud.points * factor.to(cloud.points.dtype))

    def ranging_surface(self, cloud, rgb):
        """The cloud the ToF's ranges are read against: MASt3R's own, or (ITO_TOF_VIA=moge)
        MoGe-2's surfaces brought to MASt3R's units by their median ratio over the whole frame."""
        if not TOF_VIA_MOGE or self.mono is None:
            return cloud
        rows, columns = cloud.shape
        valid = cloud.valid.reshape(rows, columns).cpu().numpy()
        mast3r = np.where(
            valid, -cloud.points[:, 2].reshape(rows, columns).float().cpu().numpy(), 0
        )
        i = self.intrinsics
        fov = math.degrees(2 * math.atan(i.width / 2 / i.fx))
        with self.gpu_chunk("depth"):
            depth = self.mono(rgb, fov, (2 * columns, 2 * rows))
        ratio = scale_to(mast3r, resample(depth, mast3r.shape))
        return cloud if ratio is None else self.with_depth(cloud, depth * ratio)

    def monocular(self, cloud, rgb, exposure):
        """The cloud with MoGe-2's depth for the frame, at the scale MASt3R's depth has (the
        median ratio over pixels both have)."""
        rows, columns = cloud.shape
        valid = cloud.valid.reshape(rows, columns).cpu().numpy()
        mast3r = np.where(
            valid, -cloud.points[:, 2].reshape(rows, columns).float().cpu().numpy(), 0
        )
        i = self.intrinsics
        fov = math.degrees(2 * math.atan(i.width / 2 / i.fx))
        with self.gpu_chunk("depth"):
            depth = self.mono(rgb, fov, (2 * columns, 2 * rows))
        both = (mast3r > 0) & (resample(depth, mast3r.shape) > 0)
        if both.sum() < 0.2 * both.size:
            return cloud
        scale = scale_to(mast3r, resample(depth, mast3r.shape))
        if scale is None:
            return cloud
        if MONO_SCALE:
            # The ToF's ranges and the floor below the robot's camera read MoGe-2's scale on
            # its own surfaces; MASt3R's metric depth (itself scaled from the ToF and floor on
            # MASt3R's bent surfaces, about 6% off either way) only weighs in beside them.
            unit = self.with_depth(cloud, depth)
            robot = self.motion.at(exposure)
            found = dict(
                tof=self.sensors.range_scale(unit, exposure, self.motion),
                floor=floor_scale(unit, robot[:3, :3].T @ UP, float(robot[1, 3])),
                mast3r=scale,
            )
            evidence = [
                (math.log(found[k]), sigma)
                for k, sigma in (
                    ("tof", RANGE_SIGMA),
                    ("floor", FLOOR_SIGMA),
                    ("mast3r", MONO_MAST3R),
                )
                if found[k]
            ]
            weights = [sigma**-2 for _, sigma in evidence]
            scale = math.exp(
                sum(w * v for w, (v, _) in zip(weights, evidence, strict=True)) / sum(weights)
            )
            if COMPARE:
                self.depth_scales |= {f"mono_{k}": v for k, v in found.items()}
                if self.truth is not None and self.truth[2] is not None:
                    self.depth_scales["mono_exact"] = scale_to(
                        self.truth[2], resample(depth, mast3r.shape)
                    )
        fit = log_field(depth, mast3r, MONO_GRID) if MONO_FIELD and not MONO_SCALE else None
        depth = depth * (fit[0] if fit is not None else scale)
        if DEPTH == "hybrid":
            # MASt3R's own depth where the two agree (its detail up close, where it is sharper),
            # MoGe-2's where they do not (walls, far surfaces, the bleed around silhouettes).
            mast3r_fine = resample(mast3r, depth.shape)
            agree = (mast3r_fine > 0) & (np.abs(mast3r_fine - depth) < HYBRID_AGREE * depth)
            depth = np.where(agree, mast3r_fine, depth)
        # Even sharp single-image depth ramps over a pixel or two at a silhouette.
        span = maximum_filter(depth, size=3) - minimum_filter(depth, size=3)
        depth = np.where(span > SILHOUETTE * depth, 0, depth)
        if COMPARE:
            self.depth_scales["mono"] = scale
        return self.with_depth(cloud, depth)

    def with_depth(self, cloud, depth):
        """The cloud with this axial depth (metres, covering the whole frame at any size) in
        place of the network's, sampled at the cloud's pixels."""
        height, width = cloud.shape
        fx, fy, cx, cy = cloud.intrinsics
        i = self.intrinsics
        v, u = np.mgrid[:height, :width].astype(np.float64)
        rows, columns = depth.shape
        x = ((u - cx) / fx * i.fx + i.cx + 0.5) * columns / i.width - 0.5
        y = ((v - cy) / fy * i.fy + i.cy + 0.5) * rows / i.height - 0.5
        x = np.clip(np.rint(x).astype(int), 0, columns - 1)
        y = np.clip(np.rint(y).astype(int), 0, rows - 1)
        d = depth[y, x].astype(np.float32)
        points = np.stack(((u - cx) / fx * d, -(v - cy) / fy * d, -d), -1).astype(np.float32)
        torch = _torch()
        device = cloud.points.device
        return cloud._replace(
            points=torch.as_tensor(points.reshape(-1, 3), device=device),
            valid=torch.as_tensor((d > 0).ravel(), device=device),
        )

    def pose_at(self, captured):
        pose = self.graph.pose_at(captured)
        return None if pose is None else self.registration.map_from_estimator @ pose

    def _publish(self, node):
        graph = self.graph
        graph.publish(node)
        pose = graph.pose(node)
        mapping = self.registration.map_from_estimator
        self.camera_pose = (mapping @ pose).astype(np.float32)
        reference = self.truth[3] if self.truth is not None and len(self.truth) > 3 else None
        self.pose_error = None
        if reference is not None:
            # compare: where this frame's estimate put the camera against where it was, in the
            # exact camera's frame (cm: right, up, back) and degrees, map frame and estimator's.
            for name, estimate in (("map", self.camera_pose), ("estimator", pose)):
                delta = np.linalg.inv(reference) @ estimate
                angle = np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
                from scipy.spatial.transform import Rotation

                axis = Rotation.from_matrix(delta[:3, :3]).as_rotvec(degrees=True)
                self.pose_error = (self.pose_error or {}) | {
                    f"{name}_cm": [round(float(v) * 100, 2) for v in delta[:3, 3]],
                    f"{name}_deg": round(float(angle), 2),
                    # about the camera's x (pitch), y (yaw) and z (roll) axes
                    f"{name}_axes_deg": [round(float(v), 2) for v in axis],
                }
        self.correction = (mapping @ graph.correction(node)).astype(np.float32)
        if self.truth is not None and self.truth[0] is not None:
            pose, exposed = self.truth[:2]
            self.camera_pose = pose.astype(np.float32)
            self.correction = (pose @ np.linalg.inv(self.motion.at(exposed))).astype(np.float32)
        self.tracked += 1
        self.report(
            f"Robot-first tracking | {self.tracked} frames, "
            f"{self.matched / self.tracked:.0%} matched, {self.blurred} blurred, "
            f"{self.sensors.ranged} ToF-scaled, "
            f"frame delay {graph.td * 1000:.0f} ms"
        )
        return pose

    def photo(self, rgb, intrinsics, shape):
        """The camera's frame resampled to DETAIL times the network's grid (shape, with
        intrinsics), and that grid's intrinsics."""
        import cv2

        height, width = shape
        rows, columns = round(height * DETAIL), round(width * DETAIL)
        fx, fy, cx, cy = intrinsics
        fine = (fx * DETAIL, fy * DETAIL, (cx + 0.5) * DETAIL - 0.5, (cy + 0.5) * DETAIL - 0.5)
        key = (rows, columns, intrinsics, rgb.shape)
        if getattr(self, "_photo_key", None) != key:
            i = self.intrinsics
            v, u = np.mgrid[:rows, :columns].astype(np.float32)
            self._photo_maps = (
                ((u - fine[2]) / fine[0] * i.fx + i.cx).astype(np.float32),
                ((v - fine[3]) / fine[1] * i.fy + i.cy).astype(np.float32),
            )
            self._photo_key = key
        return cv2.remap(rgb, *self._photo_maps, cv2.INTER_LINEAR), fine

    def dump(self, cloud, raw_depth, points, valid, image, pose, rgb):
        from pathlib import Path

        from .depth import resample

        folder = Path(DUMP)
        folder.mkdir(parents=True, exist_ok=True)
        truth = None
        if self.truth is not None and self.truth[2] is not None:
            truth = resample(self.truth[2], raw_depth.shape).astype(np.float32)
        np.savez_compressed(
            folder / f"{self.source_sequence:06d}.npz",
            image=image,
            camera=rgb if rgb is not None else np.zeros((0,)),
            estimated=raw_depth,
            mapped=np.where(valid, -points[..., 2], 0).astype(np.float32),
            truth=np.zeros((0,)) if truth is None else truth,
            pose=pose,
            intrinsics=np.array(cloud.intrinsics),
            camera_intrinsics=np.array(
                [self.intrinsics.fx, self.intrinsics.fy, self.intrinsics.cx, self.intrinsics.cy]
            ),
            captured=self.source_captured,
            zones=np.array(getattr(self, "zone_hits", []), np.float32).reshape(-1, 3),
        )

    def place(self, cloud, pose, when, *, node, rgb=None):
        height, width = cloud.shape
        points = cloud.points.detach().float().cpu().numpy().reshape(height, width, 3)
        image = cloud.colors.detach().cpu().numpy().reshape(height, width, 3)
        valid = cloud.valid.detach().cpu().numpy().reshape(height, width)
        intrinsics = cloud.intrinsics
        if MARGIN and not (self.truth is not None and self.truth[2] is not None and not COMPARE):
            rows, columns = int(MARGIN * height), int(MARGIN * width)
            valid = valid.copy()
            valid[:rows], valid[height - rows :] = False, False
            valid[:, :columns], valid[:, width - columns :] = False, False
        photo = None
        if DETAIL > 1 and rgb is not None:
            photo = self.photo(rgb, intrinsics, (height, width))
        self.observation_estimate = pose
        self.observation_map_estimate = self.registration.map_from_estimator @ pose
        mapped, scale = self.registration.align(
            image, -points[..., 2], pose, intrinsics, self.views
        )
        if GRAPH_SCALE:
            scale = 1.0
        if self.truth is not None and self.truth[0] is not None:
            # Registration only rescales a true pose's depth; it never moves it.
            mapped = pose
        elif self.placed is not None:
            mapped, scale = self.placed, 1.0
        raw_depth = np.where(valid, -points[..., 2] * scale, 0).astype(np.float32)
        if MAPSCALE and not (self.truth is not None and self.truth[2] is not None and not COMPARE):
            ids, z, u, v = self.centers(mapped, intrinsics, raw_depth.shape)
            if len(ids) > 1000:
                alpha, drawn, _ = self.render(mapped, intrinsics, raw_depth.shape, ids, z, u, v)
                both = (alpha > 0.9) & (drawn > 0) & (raw_depth > 0)
                fit = None
                if MAPFIELD and both.mean() > MAPFIELD_COVERED:
                    fit = log_field(
                        raw_depth,
                        np.where(alpha > 0.9, drawn, 0),
                        MAPFIELD_GRID,
                        prior_weight=MAPFIELD_PRIOR,
                    )
                if fit is not None:
                    field = np.clip(fit[0], *MAPSCALE_RANGE)
                    raw_depth *= field
                    points = points * field[..., None]
                    if COMPARE:
                        self.depth_scales["map"] = float(np.median(field))
                elif not MAPFIELD and both.mean() > MAPSCALE_COVERED:
                    ratio = float(np.median(drawn[both] / raw_depth[both]))
                    if MAPSCALE_RANGE[0] < ratio < MAPSCALE_RANGE[1]:
                        scale *= ratio
                        raw_depth *= ratio
                        if COMPARE:
                            self.depth_scales["map"] = ratio
        if MAP_AGREE and not (self.truth is not None and self.truth[2] is not None and not COMPARE):
            ids, z, u, v = self.centers(mapped, intrinsics, raw_depth.shape)
            if len(ids) > 1000:
                alpha, drawn, width = self.render(mapped, intrinsics, raw_depth.shape, ids, z, u, v)
                known = (alpha > 0.9) & (drawn > 0) & (raw_depth > 0) & (width < MAP_AGREE_WIDTH)
                ratio = np.log(np.maximum(raw_depth, 1e-6) / np.maximum(drawn, 1e-6))
                conflict = known & (np.abs(ratio) > MAP_AGREE)
                valid = valid & ~conflict
                raw_depth = np.where(valid, raw_depth, 0).astype(np.float32)
                self.map_conflicts += int(conflict.sum())
        if self.refine is not None and not (self.truth is not None and self.truth[0] is not None):
            with self.gpu_chunk("map"):
                refined = self.refine(image, raw_depth, mapped, intrinsics)
            if refined is not None:
                mapped = refined
                estimator = np.linalg.inv(self.registration.map_from_estimator) @ refined
                if self.graph.add_map(node, estimator, MAP_SIGMA):
                    self.graph.solve_frame()
        if self.truth is not None and self.truth[2] is not None and not COMPARE:
            scale = 1.0
        elif self.volume is not None:
            depth = np.where(valid, -points[..., 2] * scale, 0).astype(np.float32)
            with self.gpu_chunk("fuse"):
                if TOF_MAP:
                    self.clear_ranges()
                trust = self.range_trust
                if self.range_field is not None and trust is None:
                    trust = np.full(depth.shape, TRUST_UNREAD, np.float32)
                if self.from_da3 and DA3_EVERY > 1:
                    trust = DA3_TRUST * (1 if trust is None else trust)
                fused = self.volume.fuse(depth, mapped, intrinsics, trust)
                if self.volume.frames % 2 == 0:
                    self.carve(mapped, intrinsics, depth.shape)
            fx, fy, cx, cy = intrinsics
            v, u = np.mgrid[:height, :width].astype(np.float32)
            points = np.stack(((u - cx) / fx * fused, -(v - cy) / fy * fused, -fused), -1).astype(
                np.float32
            )
            valid, scale = fused > 0, 1.0
        elif self.consensus is not None:
            depth = np.where(valid, -points[..., 2] * scale, 0).astype(np.float32)
            with self.gpu_chunk("consensus"):
                agreed, _ = self.consensus.fuse(
                    depth, mapped, intrinsics, self.source_captured, self.views
                )
            points = points * (agreed / np.maximum(-points[..., 2], 1e-6))[..., None]
            valid, scale = agreed > 0, 1.0
        if DUMP:
            self.dump(cloud, raw_depth, points * scale, valid, image, mapped, rgb)
        self.registration.remember(
            node, pose, self.registration.map_from_estimator @ pose, self.graph.nodes
        )
        self.observe(
            points * scale,
            image,
            valid,
            mapped,
            intrinsics,
            when,
            node=node,
            registered=self.registration.matched
            or self.placed is not None
            or (self.truth is not None and self.truth[0] is not None),
            photo=photo,
            raw_depth=raw_depth if self.consensus is not None else None,
        )


def floor_scale(cloud, up, height):
    """Metres per front-end unit from the floor in a frame's points, or None if it is not
    in view. up: the world's up in the camera frame; height: the camera's above the floor."""
    if height < FLOOR_HEIGHT:
        return None
    torch = _torch()
    points = cloud.points[cloud.valid][::4]
    up = torch.as_tensor(up, device=points.device, dtype=points.dtype)
    below = -(points @ up)
    lower = below > math.sin(FLOOR_RAY) * torch.linalg.vector_norm(points, dim=1)
    levels = torch.log(below[lower]).cpu().numpy()
    if len(levels) < FLOOR_SHARE * len(points):
        return None
    bottom, top = np.percentile(levels, (40, 99.5))
    counts, edges = np.histogram(levels, np.arange(bottom, top + 0.04, 0.02))
    densest = int(np.argmax(np.convolve(counts, np.ones(5), "same")))
    near = np.abs(levels - edges[densest] - 0.01) < FLOOR_LEVEL
    if near.sum() < FLOOR_SHARE * len(points):
        return None
    return height / math.exp(float(np.median(levels[near])))


def _torch():
    import torch

    return torch


def heading(rotation):
    """Yaw of a world-from-camera rotation's view direction (-Z), positive to the left."""
    return math.atan2(rotation[0, 2], rotation[2, 2])
