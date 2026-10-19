import logging
import math
import os
import time

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from ito import clock

from .align import to_map
from .depth import (
    MapFrontend,
    MonoGeometry,
    MultiViewDepth,
    Pi3Depth,
    PromptedDepth,
    VideoGeometry,
    log_field,
    nearest_views,
    resample,
    scale_to,
)
from .fusion import HEIGHT_SIGMA, UP, PoseGraph
from .gaussian import EXPOSURE_INPUT, GaussianBackend, exposed
from .motion import MotionBuffer
from .sensors import HeadSensors
from .slam_vision import RESTART_CAUSES

__all__ = ["RESTART_CAUSES", "SLAMBackend"]

log = logging.getLogger(__name__)

# A frame smeared further than this by the camera's turn during its exposure still gets a
# pose, from the robot, but neither a visual factor nor splats: no tracker matches through
# it, and its splats would be streaks. Blur is |angular velocity| x exposure x focal length.
BLUR_PIXELS = 8.0
EXPOSURE = 0.010
# ITO_ROLLING_SHUTTER=MS: the camera reads its rows top to bottom over MS milliseconds (an IMX219
# at 1080p30: ~30), so a turning head skews each frame. Every frame is first warped to how it
# would look read all at once at its stamp, by the camera's turn then (the motion buffer):
# tracking, depth and colour all see the unskewed frame.
ROLLING_SHUTTER = float(os.environ.get("ITO_ROLLING_SHUTTER", "0")) / 1000
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
# ITO_SLAM=vision: vision-first tracking (vision_track.py): MASt3R-SLAM's own tracker and keyframe
# graph place every frame; the robot seeds each frame's turn, the ToF/floor set one metric scale and
# the first frame ties the vision world to the robot's. Fusion's robot-first window is not used.
VISION = os.environ.get("ITO_SLAM") == "vision"
# The metric scale (metres per vision unit) is pooled over this many ToF/floor readings, then held:
# a later change would grow or shrink the whole room about its first camera.
VISION_SCALE_READINGS = int(os.environ.get("ITO_VISION_SCALE_READINGS", "40"))
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
# ITO_DEPTH=vigeo maps ViGeo's streaming video depth (depth.py VideoGeometry) through the same
# path as MoGe-2's. ITO_MONO_STEADY=T: the scale is the scale evidence of the last ~T seconds
# pooled (weights decaying with age), not each frame's own: a video network keeps one scale
# from frame to frame, while each frame's ToF or floor reading is a few percent off.
MONO_STEADY = float(os.environ.get("ITO_MONO_STEADY", "0"))
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
# ITO_SLAM=full: upstream MASt3R-SLAM's whole backend (mast3r_global.py): every keyframe in one
# global Sim(3) graph with the robot's factors, retrieval loop closure, relocalisation; the map
# anchored to keyframes and moved with them. Unset: the 30-keyframe pose graph alone, as before.
FULL = os.environ.get("ITO_SLAM") == "full"
FLIP = np.diag([1.0, -1.0, -1.0])  # Ito axes <-> MASt3R's (OpenCV), for the global graph
FLIP4 = np.diag([1.0, -1.0, -1.0, 1.0])
# ITO_HEAD_IMU=1 (stone P3): the head IMU's gyro turn since the last keyframe sets the frame
# tracker's rotation prior (the kinematic one rides up to 2 degrees past the servos' play), and
# with ITO_SLAM=full its turns and tilt join the global graph. Unset: as before.
HEAD_IMU = os.environ.get("ITO_HEAD_IMU", "0") != "0"
MAP_MOVE = (0.002, 0.05, 0.002)  # metres, degrees, log-scale: a keyframe's move the map follows
KEYFRAME_AGE = float(os.environ.get("ITO_SLAM_KEYFRAME_AGE", "20"))  # s, full SLAM
FIELD_MOVE = (0.02, 1.5)  # metres, degrees: keyframe moves that rebuild the fused field
FIELD_INTERVAL = 20.0  # seconds: the least time from one field rebuild to the next


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
        if frontend is None and VISION:
            from .vision_track import VisionTracker

            frontend = VisionTracker(
                intrinsics,
                report,
                precision=precision,
                input_resolution=input_resolution,
                precision_report=precision_report,
            )
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
        self.depth = None
        if DEPTH == "da3":
            self.depth = Pi3Depth() if os.environ.get("ITO_DA3") == "pi3" else MultiViewDepth()
        self.mono = MonoGeometry() if DEPTH in ("moge", "hybrid") or TOF_VIA_MOGE else None
        if DEPTH == "vigeo":
            self.mono = VideoGeometry()
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
        self.global_graph = None  # ITO_SLAM=full, made in bind_admission
        self.empty_window_frames = 0  # frames tracked while the pose graph had no keyframe
        self.robot_camera = None  # the robot's own camera pose at the last frame (timeline)
        self.field_moved = (0.0, 0.0)  # full: keyframe moves since the field was last rebuilt
        self.field_rebuilt = -math.inf  # time.monotonic() of the last rebuild started
        self.keyframe_exposure = {}  # full: keyframe index -> exposure time (pilot clock)
        self.node_keyframe = {}  # full: frame node -> keyframe index its surfaces follow
        self.global_applied = 0
        self.global_change = (0.0, 0.0)
        self.map_poses = {}  # full: keyframe -> its pose (Ito axes) its surfaces were placed by
        self.first_pose = None  # full: keyframe 0's rigid pose when it was made
        self.imu_priors = 0  # ITO_HEAD_IMU: frames tracked with the gyro's rotation prior
        # Full with ITO_FUSE: every fused frame's input, by keyframe, to fuse the field afresh
        # where the global graph moved keyframes (rebuild_volume) instead of keeping its old room.
        self.fused_frames = []
        self.rebuilding = None  # the thread fusing a new field
        self.rebuilt = None  # a new field and how many fused_frames it holds
        self.volume_rebuilds = 0
        self.rebuild_seconds = 0.0
        from .registration import Registration

        self.registration = Registration()
        report("MASt3R ready; waiting for camera and robot motion")

    def bind_admission(self, admission):
        super().bind_admission(admission)
        if hasattr(self.frontend, "bind_admission"):
            self.frontend.bind_admission(admission)
        if FULL and hasattr(self.frontend, "global_graph"):
            from .mast3r_global import GlobalGraph

            self.global_graph = GlobalGraph(
                self.frontend,
                self.report,
                admission.available,
                lambda: bool(admission.stopped.value),
                admission.torch_stream,
            )
            self.frontend.global_graph = self.global_graph

    def world_alignment(self):
        """Full SLAM: the rigid move taking today's world (the gauge follows the keyframe the
        pilot looks from) back to the robot's starting frame, as keyframe 0 was placed in it."""
        if self.global_graph is None or not self.frontend.frames or 0 not in self.map_poses:
            return None
        from .mast3r_global import EDGES, level_correction, sim3_matrix, to_ito

        if self.first_pose is None:
            return None
        now = sim3_matrix(self.frontend.frames[0].T_WC.data.reshape(8).double().cpu().numpy())[0]
        if EDGES == "pose":
            # The pose graph solves keyframe 0's tilt and height too; only its heading and floor
            # position are the gauge's.
            return to_ito(level_correction(to_ito(self.first_pose), now))
        now = to_ito(now)
        now[:3, :3] /= np.cbrt(np.linalg.det(now[:3, :3]))
        return self.first_pose @ np.linalg.inv(now)

    def global_statistics(self):
        if self.global_graph is None:
            return None
        stats = self.global_graph.statistics()
        stats.update(
            applied=self.global_applied,
            change=[round(v, 4) for v in self.global_change],
            frontend_relocalised=getattr(self.frontend, "relocalisations", 0),
            imu=dict(
                gyro=len(self.sensors.gyro), tilts=len(self.sensors.tilts), priors=self.imu_priors
            ),
            reference=self.frontend.reference,
            match=None
            if self.frontend.last_match is None
            else round(float(self.frontend.last_match), 3),
            empty_window_frames=self.empty_window_frames,
            forced_keyframes=self.frontend.forced,
            robot=None
            if self.robot_camera is None
            else [round(float(v), 3) for v in self.robot_camera[:3, 3]]
            + [round(math.degrees(heading(self.robot_camera[:3, :3])), 1)],
            field_rebuilds=self.volume_rebuilds,
            field_rebuild_ms=round(self.rebuild_seconds * 1000),
            fused_frames=len(self.fused_frames),
        )
        return stats

    def apply_global(self):
        """Full SLAM, on the tracking thread: take each finished global solve to the keyframes,
        the pose graph's window and every surface and view anchored to a moved keyframe."""
        from .mast3r_global import corrections, matrix_sim3, to_ito

        torch = _torch()
        import lietorch

        frontend, graph = self.frontend, self.graph
        for result in self.global_graph.poll():
            frames = frontend.frames
            current = {
                k: frames[k].T_WC.data.reshape(8).double().cpu().numpy()
                for k in result["keyframes"]
            }
            changes = corrections(result, current)
            with torch.inference_mode():
                for k, (_, pose) in changes.items():
                    data = torch.as_tensor(matrix_sim3(pose), dtype=torch.float32, device="cuda")
                    frames[k].T_WC = lietorch.Sim3(data)
                    self.global_graph.owned.add(k)
            index = {node: i for i, node in enumerate(frontend.ids)}
            for node in graph.keyframes:
                k = index.get(node)
                if k in changes:
                    pose, _ = frontend.placed(frames[k].T_WC)
                    if pose is not None:
                        graph.nodes[node].pose = pose  # held there by solve_window(fixed)
            if graph.keyframes:
                graph.anchor = graph.nodes[graph.keyframes[0]].pose.copy()
            graph._refresh_published()
            # The map follows a keyframe once its pose has moved past what a surface would show
            # (MAP_MOVE); smaller corrections wait, adding up, so a solve does not resend the map.
            deltas, moved, turned = {}, 0.0, 0.0
            for k, (_, pose) in changes.items():
                # Rigid only: surfaces and fused depth are metric (each frame scaled by its ToF or
                # the floor); a keyframe's Sim3 scale is its pointmap's unit. Following it stretched
                # a keyframe's splats 1.7-2.2x when a solve first set a new keyframe's scale.
                now = _rigid(to_ito(pose))
                before = self.map_poses.get(k)
                if before is None:
                    self.map_poses[k] = now
                    continue
                delta = now @ np.linalg.inv(before)
                scale = float(np.cbrt(np.linalg.det(delta[:3, :3])))
                rotation = delta[:3, :3] / scale
                angle = math.degrees(math.acos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
                shift = float(
                    np.linalg.norm(delta[:3, :3] @ before[:3, 3] + delta[:3, 3] - before[:3, 3])
                )
                moved, turned = max(moved, shift), max(turned, angle)
                if shift > MAP_MOVE[0] or angle > MAP_MOVE[1] or abs(math.log(scale)) > MAP_MOVE[2]:
                    deltas[k] = delta
                    self.map_poses[k] = now
            self.global_change = (moved, turned)
            self.global_applied += 1
            if deltas:
                self.repose(deltas, self.node_keyframe)
                if self.volume is not None:
                    self.move_fused(deltas)
                    # A rebuild takes seconds and holds two fields: rebuilt after nearly every
                    # solve the CUDA peak reached 10.4 GiB (hp-fullptn-loop). Only once keyframes
                    # have moved 2 cm or 1.5 degrees since, and at most every FIELD_INTERVAL.
                    shift, turn = self.field_moved
                    self.field_moved = (shift + moved, turn + turned)
                    far = self.field_moved[0] > FIELD_MOVE[0] or self.field_moved[1] > FIELD_MOVE[1]
                    if far and time.monotonic() - self.field_rebuilt >= FIELD_INTERVAL:
                        self.field_moved = (0.0, 0.0)
                        self.field_rebuilt = time.monotonic()
                        self.rebuild_volume()

    def remember_fused(self, depth, trust, pose, intrinsics, node):
        self.fused_frames.append(
            dict(
                depth=depth.astype(np.float16),
                trust=None if trust is None else np.asarray(trust, np.float16),
                pose=np.array(pose, np.float64),
                intrinsics=intrinsics,
                keyframe=self.node_keyframe.get(node),
            )
        )

    def move_fused(self, deltas):
        from .gaussian import _moved_camera

        for frame in self.fused_frames:
            delta = deltas.get(frame["keyframe"])
            if delta is None:
                continue
            frame["pose"] = _moved_camera(delta, frame["pose"])
            scale = float(np.cbrt(np.linalg.det(delta[:3, :3])))
            if abs(scale - 1) > 1e-6:
                frame["depth"] = (frame["depth"].astype(np.float32) * scale).astype(np.float16)

    def rebuild_volume(self):
        """Fuse every remembered frame at its pose now into a new field, on a thread of its own
        beside tracking; frames fused meanwhile are added before it replaces the old one."""
        import threading

        from .volume import Volume

        if self.rebuilding is not None:
            self.rebuild_again = True
            return
        self.rebuild_again = False
        frames = list(self.fused_frames)
        blocks = self.volume.count
        torch = _torch()
        available = self.admission.available
        stopped = self.global_graph.stopped
        tracking = self.admission.torch_stream

        def run():
            began = time.perf_counter()
            stream = torch.cuda.Stream(priority=0)
            try:
                # no_grad, not inference_mode: tracking updates this field in place afterwards.
                with torch.cuda.stream(stream), torch.no_grad():
                    if tracking is not None:
                        stream.wait_stream(tracking)
                    volume = Volume()
                    _reserve(volume, blocks + blocks // 8 + 4096)
                    for frame in frames:
                        while not available():
                            if stopped():
                                return
                            time.sleep(0.002)
                        _fuse_remembered(volume, frame)
                    stream.synchronize()
                self.rebuilt = (volume, len(frames))
                self.rebuild_seconds = time.perf_counter() - began
            except Exception:
                log.exception("Field rebuild failed")
                self.rebuilding = None

        self.rebuilding = threading.Thread(target=run, name="field-rebuild", daemon=True)
        self.rebuilding.start()

    def swap_volume(self):
        volume, count = self.rebuilt
        for frame in self.fused_frames[count:]:
            _fuse_remembered(volume, frame)
        for name in ("filled", "agreed", "refuted", "carved", "cleared"):
            setattr(volume, name, getattr(self.volume, name))
        self.volume = volume
        self.rebuilt = self.rebuilding = None
        self.volume_rebuilds += 1
        # The old field's blocks go back to the driver, not into this process's cache.
        _torch().cuda.empty_cache()
        if self.rebuild_again:
            self.field_moved = (math.inf, math.inf)  # rebuilt at the next solve the interval allows

    def queue_keyframe(self, node, exposure):
        """Full SLAM: hand the keyframe just kept to the global graph with the robot's factors."""
        frontend = self.frontend
        index = len(frontend.frames) - 1
        self.keyframe_exposure[index] = exposure
        previous = self.keyframe_exposure.get(index - 1)
        robot = self.motion.at(exposure)
        rotation = FLIP @ robot[:3, :3] @ FLIP
        position = FLIP @ robot[:3, 3]
        gravity = (rotation.T @ np.array([0.0, -1.0, 0.0]), self.motion.gravity_sigma(exposure))
        turn = None
        if HEAD_IMU:
            seen = self.sensors.up(exposure)
            if seen is not None:
                gravity = (FLIP @ seen[0], seen[1])  # the head's own tilt, past the gear play
            if previous is not None:
                turned = self.sensors.turn(previous, exposure)
                if turned is not None:
                    turn = (index - 1, FLIP @ turned[0].as_matrix() @ FLIP, turned[1])
        height = (float(position[1]), self.motion.height_sigma(exposure, HEIGHT_SIGMA))
        odometry = None
        if previous is not None and self.motion.covers(previous):
            relative, covariance = self.motion.between(previous, exposure)
            odometry = (
                index - 1,
                FLIP @ relative[:3, :3] @ FLIP,
                FLIP @ relative[:3, 3],
                np.sqrt(np.diag(covariance)),
            )
        from .mast3r_global import sim3_matrix, to_ito

        if index == 0:
            first = to_ito(
                sim3_matrix(frontend.frames[0].T_WC.data.reshape(8).double().cpu().numpy())[0]
            )
            first[:3, :3] /= np.cbrt(np.linalg.det(first[:3, :3]))
            self.first_pose = first
        self.map_poses[index] = _rigid(
            to_ito(
                sim3_matrix(frontend.frames[index].T_WC.data.reshape(8).double().cpu().numpy())[0]
            )
        )
        lost = frontend.lost
        self.global_graph.add_keyframe(
            index,
            tracked_against=None if lost else frontend.reference,
            lost=lost,
            gravity=gravity,
            height=height,
            odometry=odometry,
            turn=turn,
            truth=self._keyframe_truth(),
        )
        self.node_keyframe[node] = index

    def _keyframe_truth(self):
        """The sim's exact pose of the frame just kept (MASt3R axes), if the run has it."""
        truth = self.truth
        if truth is None:
            return None
        exact = truth[3] if len(truth) > 3 and truth[3] is not None else truth[0]
        return None if exact is None else FLIP4 @ np.asarray(exact, float) @ FLIP4

    def add_sensors(self, imu, ranges):
        self.sensors.add(imu, ranges)

    def add_motion(self, rows):
        self.motion.extend(rows)

    def integrate(self, rgb, depth, camera, now, measured=False, *, captured, truth=None):
        """captured: the frame's stamp on the pilot clock, the clock motion samples are on.
        truth: (world-from-camera or None, exposure on the pilot clock, axial depth or None)
        to publish and map this frame by instead of the estimates; the estimator still runs,
        unchanged, beside them."""
        if VISION:
            return self.integrate_vision(rgb, now, captured)
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
        if ROLLING_SHUTTER:
            rgb = self.unroll(rgb, exposure)
        if self.global_graph is not None:
            self.apply_global()
        previous = graph.latest()
        previous_time = graph.nodes[previous].time if previous is not None else None
        propagated = graph.predict(captured) if previous is not None else None
        expired = graph.expire(self.motion.times[0])
        self.retired_keyframes += len(expired)
        self.frontend.forget(expired)
        # Full SLAM: a revisit makes no keyframe, so the window empties as the motion buffer
        # moves on; each frame then follows the one before it by the robot's motion, not raw
        # odometry held to where the window emptied.
        empty = not graph.keyframes and propagated is not None
        hold = empty and (bool(expired) or self.global_graph is not None)
        if hold:
            graph.anchor = propagated.copy()
            self.empty_window_frames += 1
        node = graph.add_frame(captured, pose=propagated if hold else None)
        predicted = graph.pose(node)
        last = graph.latest_keyframe()
        reference_time = graph.nodes[last].time if last is not None else captured
        if last is None and hold and previous_time is not None and self.global_graph is not None:
            reference_time = previous_time
        _, covariance = self.motion.between(reference_time - graph.td, exposure)
        prior_sigma = np.sqrt(np.diag(covariance))
        if HEAD_IMU and last is not None:
            turned = self.sensors.turn(reference_time - graph.td, exposure)
            if turned is not None and turned[1] < prior_sigma[0]:
                predicted = predicted.copy()
                predicted[:3, :3] = graph.nodes[last].pose[:3, :3] @ turned[0].as_matrix()
                prior_sigma = prior_sigma.copy()
                prior_sigma[:3] = turned[1]
                self.imu_priors += 1
        gyro = self.sensors.angular_velocity(exposure)
        if gyro is None:
            gyro = self.motion.angular_velocity(exposure)
        spin = float(np.linalg.norm(gyro))
        if spin * self.exposure * self.focal > BLUR_PIXELS:
            self.blurred += 1
            graph.solve_frame()
            self._publish(node)
            return
        # ...and the next frame that tracks becomes a keyframe, so the window has one again. Full
        # SLAM also keeps one at least every KEYFRAME_AGE seconds while the robot dwells on known
        # views: chained from an old keyframe, frames drift with the head's gear play (2.5-4.5 deg
        # by the end of a close-up replay) and the window empties.
        force = self.global_graph is not None and (
            hold or (last is not None and captured - graph.nodes[last].time > KEYFRAME_AGE)
        )
        result = self.frontend.track(
            rgb, node, predicted, prior_sigma, **({"force": True} if force else {})
        )
        self.robot_camera = self.motion.at(exposure)
        cloud = result.cloud
        if self.global_graph is not None:
            self.global_graph.reference = self.frontend.reference
        if result.relative is not None:
            if self.global_graph is not None and result.reference not in graph.nodes:
                # Tracked against a keyframe the window has let go (a revisit): where the
                # global graph has that keyframe says where this frame is.
                k = self.frontend.reference
                anchor, _ = self.frontend.placed(self.frontend.frames[k].T_WC)
                if anchor is not None:
                    graph.add_map(node, anchor @ result.relative, result.sigma)
            else:
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
            if ranged is not None and self.global_graph is not None and ranged > 0:
                # Metres per unit of the reference keyframe's pointmap, and of this frame's
                # own if it becomes a keyframe: the global graph's tie to the ToF.
                k = self.frontend.reference
                if k is not None and k < len(self.frontend.frames):
                    _, own = self.frontend.placed(self.frontend.frames[k].T_WC)
                    self.global_graph.add_scale(k, math.log(own * ranged))
                if result.keyframe:
                    self.global_graph.add_scale(
                        len(self.frontend.frames) - 1, math.log(self.frontend.last_scale * ranged)
                    )
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
            fixed = ()
            if self.global_graph is not None:
                fixed = {self.frontend.ids[k] for k in self.global_graph.owned}
            poses = graph.solve_window(fixed)
        scale = 1.0
        if poses or abs(graph.log_scale) > SCALE_STEP:
            scale = graph.fold_scale()
            self.frontend.correct(poses, scale)
        if (truth is None or truth[0] is None) and self.global_graph is None:
            self.revise(poses)
        if dropped:
            self.retired_keyframes += len(dropped)
            self.frontend.forget(dropped)
        if self.global_graph is not None:
            if result.keyframe:
                self.queue_keyframe(node, exposure)
            elif self.frontend.reference is not None:
                self.node_keyframe[node] = self.frontend.reference
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

    def unroll(self, rgb, exposure):
        """rgb warped from rolling-shutter rows (row v read at (v / height - 1/2) x ROLLING_SHUTTER
        from the stamp) to one read at the stamp, for the camera's rotation alone."""
        import cv2

        before = self.motion.at(exposure - 0.02)[:3, :3]
        after = self.motion.at(exposure + 0.02)[:3, :3]
        turn = before.T @ after  # camera axes: x right, y up, looking down -z
        angle = math.acos(max(-1.0, min(1.0, (np.trace(turn) - 1) / 2)))
        if angle < 1e-5:
            return rgb
        axis = np.array(
            (turn[2, 1] - turn[1, 2], turn[0, 2] - turn[2, 0], turn[1, 0] - turn[0, 1])
        ) / (2 * math.sin(angle))
        wx, wy, wz = (float(w) for w in axis * angle / 0.04)  # radians per second
        height, width = rgb.shape[:2]
        if getattr(self, "unroll_grid", (None,))[0] != (height, width):
            v, u = np.mgrid[:height, :width].astype(np.float32)
            self.unroll_grid = ((height, width), u, v)
        _, u, v = self.unroll_grid
        f = float(self.intrinsics.fx) * width / self.intrinsics.width
        t = (v / height - 0.5) * ROLLING_SHUTTER
        # Image motion of a turning camera, pixels per second (+u right, +v down).
        du = f * wy - wz * (v - height / 2)
        dv = f * wx + wz * (u - width / 2)
        return cv2.remap(
            np.ascontiguousarray(rgb),
            u + du * t,
            v + dv * t,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

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
        if MONO_STEADY:
            pooled = self.steady(
                evidence if MONO_SCALE else [(math.log(scale), MONO_MAST3R)], self.source_captured
            )
            scale = pooled if pooled is not None else scale
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

    def steady(self, evidence, now):
        """exp of the weighted mean of log-scale evidence (value, sigma) seen over about the
        last MONO_STEADY seconds, or None before any."""
        last = getattr(self, "steady_time", None)
        decay = 1.0 if last is None else math.exp(-max(now - last, 0.0) / MONO_STEADY)
        self.steady_time = now
        total = getattr(self, "steady_weight", 0.0) * decay
        value = getattr(self, "steady_value", 0.0) * decay
        for v, sigma in evidence:
            total += sigma**-2
            value += sigma**-2 * v
        self.steady_weight, self.steady_value = total, value
        return math.exp(value / total) if total > 0 else None

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

    def integrate_vision(self, rgb, now, captured):
        """ITO_SLAM=vision: the frame placed by the images (vision_track.py), in the robot world."""
        started = clock.now()
        self.from_da3 = False
        exposure = captured - self.graph.td
        if not self.motion.covers(exposure):
            if not self.waiting:
                self.report("Vision tracking waiting for the robot's motion")
            self.waiting = True
            return
        self.waiting = False
        if ROLLING_SHUTTER:
            rgb = self.unroll(rgb, exposure)
        robot = self.motion.at(exposure)
        last = getattr(self, "vision_robot", None)
        turn = None if last is None else last[:3, :3].T @ robot[:3, :3]
        self.vision_robot = robot
        pose, unit, cloud, _, tracked = self.frontend.track(rgb, turn)
        if pose is None:
            return
        if getattr(self, "vision_world", None) is None:
            # The vision world's first camera is where the robot says it was.
            self.vision_world = robot @ np.linalg.inv(pose)
            self.vision_metres = []
        metres = self.vision_scale(cloud, exposure)
        world = self.vision_world
        placed = np.eye(4)
        placed[:3, :3] = world[:3, :3] @ pose[:3, :3]
        placed[:3, 3] = world[:3, :3] @ (pose[:3, 3] * (metres or 1.0)) + world[:3, 3]
        self.camera_pose = placed.astype(np.float32)
        self.correction = (placed @ np.linalg.inv(robot)).astype(np.float32)
        published = getattr(self, "vision_published", [])
        published.append((captured, self.correction.astype(np.float64)))
        self.vision_published = published[-600:]
        self.tracked += 1
        self.matched += bool(tracked)
        self.lost = not tracked
        self.report(
            f"Vision tracking | {self.tracked} frames, {self.matched / self.tracked:.0%} tracked, "
            f"{len(self.frontend.keyframes)} keyframes, {self.frontend.solves} solves, "
            f"scale {metres or 0:.3f} m/unit from {len(self.vision_metres)} readings"
        )
        if cloud is None or metres is None:
            return
        cloud = cloud._replace(points=cloud.points * metres)
        if self.range_field is not None:
            cloud = self.bend_to_ranges(cloud, exposure)
        node = getattr(self, "vision_node", 0)
        self.vision_node = node + 1
        self.place(cloud, placed, now + clock.now() - started, node=node, rgb=rgb)

    def vision_scale(self, cloud, exposure):
        """Metres per vision unit: the median of the first VISION_SCALE_READINGS ToF and floor
        readings (ToF weighed in four times), then held; None until there are enough."""
        readings = self.vision_metres
        if cloud is not None and len(readings) < VISION_SCALE_READINGS:
            robot = self.motion.at(exposure)
            ranged = self.sensors.range_scale(cloud, exposure, self.motion)
            if ranged is not None:
                readings.extend([math.log(ranged)] * 4)
                self.range_readings += 1
            below = floor_scale(cloud, robot[:3, :3].T @ UP, float(robot[1, 3]))
            if below is not None:
                readings.append(math.log(below))
            self.scale_readings += (ranged is not None) + (below is not None)
        if len(readings) < min(VISION_SCALE_READINGS, 12):
            return None
        return math.exp(float(np.median(readings)))

    def pose_at(self, captured):
        if VISION:
            published = getattr(self, "vision_published", None)
            if not published:
                return None
            times = [t for t, _ in published]
            i = int(np.clip(np.searchsorted(times, captured), 0, len(times) - 1))
            correction = published[i][1]
            return correction @ self.motion.at(captured - self.graph.td)
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
        if self.global_graph is not None or VISION:
            # The global graph keeps the map consistent; its corrections move the map with the
            # keyframes (apply_global), so frames are placed where the graph has them. Vision-first
            # poses are the images' own: registration would only pull them off.
            mapped, scale = pose, 1.0
        else:
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
                if self.rebuilt is not None:
                    self.swap_volume()
                fused = self.volume.fuse(depth, mapped, intrinsics, trust)
                if self.global_graph is not None:
                    self.remember_fused(depth, trust, mapped, intrinsics, node)
                # A field being rebuilt for moved keyframes holds them where they were: carving
                # by it would retire the surfaces that moved.
                if self.volume.frames % 2 == 0 and self.rebuilding is None:
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
            or self.global_graph is not None
            or (self.truth is not None and self.truth[0] is not None),
            photo=photo,
            raw_depth=raw_depth if self.consensus is not None else None,
            keyframe=self.node_keyframe.get(node) if self.global_graph is not None else None,
        )


def _rigid(pose):
    """A similarity's rigid part: its rotation without the scale."""
    pose = np.array(pose, float)
    pose[:3, :3] /= np.cbrt(np.linalg.det(pose[:3, :3]))
    return pose


def _reserve(volume, capacity):
    """Room for capacity blocks in a new field at once: grown by doubling while a rebuild fused
    hundreds of frames, the copies took the CUDA peak to 10.8 GiB (hp-fullptn3-loop-b)."""
    torch = _torch()
    size = volume.sdf.shape[1]
    volume.sdf = torch.ones((capacity, size), dtype=volume.sdf.dtype, device=volume.device)
    volume.weight = torch.zeros((capacity, size), dtype=volume.weight.dtype, device=volume.device)
    volume.origins = torch.zeros((capacity, 3), dtype=torch.int64, device=volume.device)


def _fuse_remembered(volume, frame):
    depth = frame["depth"].astype(np.float32)
    trust = None if frame["trust"] is None else frame["trust"].astype(np.float32)
    weights = volume.weights(depth, frame["intrinsics"], trust)
    volume.integrate(depth, frame["pose"], frame["intrinsics"], weights)


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
