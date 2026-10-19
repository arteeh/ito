"""Vision-led full MASt3R-SLAM (ITO_SLAM=full ITO_SLAM_LEAD=vision): frames placed by the images,
keyframes by one global graph with loop closure.

Why: robot-first full SLAM (mast3r_frontend.py + fusion.py's window) ties every frame to the one
before it by the robot's motion, whose head gear play (1.5-3 degrees, depending on the head's
history) pulls frames off what the images agree on, and scales each frame's cloud by its own ToF or
floor reading (5-8% apart frame to frame): co-visible frames disagree by 0.3-1.1 degrees (p50) and
surfaces are seeded at depths 5-8% apart, so the map blurs and doubles. Vision-first tracking
(vision_track.py, SillyFalcon) takes each frame's pose AND scale from MASt3R's alignment to its
keyframe and one held metric scale: 0.07-0.16 degrees, as original MASt3R-SLAM.

What this adds to vision-first tracking, from mast3r_global.py: every keyframe stays in one Sim(3)
graph solved on a worker thread by upstream's calibrated dense Gauss-Newton (in chunks, so VRAM
stays bounded however many keyframes), loop edges from upstream's ASMK retrieval (each checked by a
two-keyframe solve), relocalisation of a frame against what retrieval finds when the newest keyframe
fails, and a gauge at the keyframe the pilot tracks against, so a solve moves the room, never the
view. slam.py applies each solve to the keyframes and moves the map's surfaces with them (repose).
"""

from .mast3r_frontend import _finite_pointmaps
from .vision_track import VisionTracker

LOST_KEYFRAME = 2  # frames in a row no keyframe tracks before a new keyframe is started there


class VisionFullTracker(VisionTracker):
    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        _finite_pointmaps()
        self.frames = self.keyframes  # the global graph's name for them
        self.global_graph = None  # set by SLAMBackend.bind_admission
        self.reference = None  # keyframe the last frame was tracked against (the gauge)
        self.parent = None  # for a new keyframe: the keyframe it was tracked against
        self.keyframe_lost = False  # the newest keyframe was started without a match
        self.last_match = None
        self.relocalisations = 0
        self.dead_reckoned = 0
        self.forced = 0

    def _attempt(self, frame, reference, seed, scale):
        """Track frame against keyframe `reference` from the seed pose; (new_keyframe, lost)."""
        if reference != self.reference:
            self.tracker.reset_idx_f2k()
        self.keyframes.reference = reference
        frame.T_WC = self.unplaced(seed, scale)
        try:
            new_keyframe, _, lost = self.tracker.track(frame)
        finally:
            self.keyframes.reference = None
        if not lost and self.placed(frame.T_WC)[0] is None:
            lost = True
        self.last_match = 0.0 if lost else float(self.tracker.match_frac)
        if lost:
            self.tracker.reset_idx_f2k()
        return bool(new_keyframe), bool(lost)

    def track(self, rgb, turn):
        """As VisionTracker.track: (pose, scale, cloud, keyframe, tracked), pose world-from-camera
        (Ito axes, vision world). Tracks against the newest keyframe (upstream); if that fails,
        against the keyframes retrieval finds for the frame; then dead reckoning on the robot's
        turn, and after LOST_KEYFRAME such frames a keyframe of its own the graph relocalises."""
        from mast3r_slam.mast3r_utils import mast3r_inference_mono

        torch = self.torch
        with torch.inference_mode():
            frame = self.image.frame(
                self.frame_id, rgb, self.lietorch.Sim3.Identity(1, device="cuda")
            )
            frame.K = self.K
            self.frame_id += 1
            if not self.keyframes:
                points, confidence = mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(points, confidence)
                self._keep(frame, None, lost=False)
                self.previous = frame.T_WC
                pose, scale = self.placed(frame.T_WC)
                return pose, scale, self.cloud(frame, scale), True, True
            last, scale = self.placed(self.previous)
            seed = last.copy()
            if turn is not None:
                seed[:3, :3] = last[:3, :3] @ turn
            reference = len(self.keyframes) - 1
            new_keyframe, lost = self._attempt(frame, reference, seed, scale)
            if lost and self.global_graph is not None:
                for other in self.global_graph.query(frame):
                    if other == reference or other >= len(self.keyframes):
                        continue
                    new_keyframe, lost = self._attempt(frame, other, seed, scale)
                    if not lost:
                        reference = other
                        self.relocalisations += 1
                        break
            if lost:
                self.lost += 1
                self.dead_reckoned += 1
                frame.T_WC = self.unplaced(seed, scale)
                if self.lost < LOST_KEYFRAME:
                    self.previous = frame.T_WC
                    return seed, scale, None, False, False
                points, confidence = mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(points, confidence)
                self._keep(frame, None, lost=True)
                self.lost = 0
                self.previous = frame.T_WC
                return seed, scale, self.cloud(frame, scale), True, False
            self.lost = 0
            self.reference = reference
            if new_keyframe:
                self._keep(frame, reference, lost=False)
            self.previous = frame.T_WC
            pose, scale = self.placed(frame.T_WC)
            return pose, scale, self.cloud(frame, scale), new_keyframe, True

    def _keep(self, frame, parent, *, lost):
        self.keyframes.append(frame)
        self.parent = parent
        self.keyframe_lost = lost
        self.reference = len(self.keyframes) - 1
        self.tracker.reset_idx_f2k()
        if self.global_graph is not None:
            # The robot's factors stay out: the images fix every relative pose far better
            # (each matched pixel is a measurement); the oldest keyframe holds the gauge.
            self.global_graph.add_keyframe(
                self.reference,
                tracked_against=parent,
                lost=lost,
                gravity=None,
                height=None,
                odometry=None,
            )

    def apply_global(self):
        """On the tracking thread, before a frame: every finished global solve to the keyframes,
        and to the last frame (it follows the keyframe it was tracked against). Returns the
        keyframes that moved."""
        graph = self.global_graph
        if graph is None:
            return set()
        from .mast3r_global import corrections, matrix_sim3, sim3_matrix

        torch, lietorch = self.torch, self.lietorch

        def sim3(matrix):
            data = torch.as_tensor(matrix_sim3(matrix), dtype=torch.float32, device="cuda")
            return lietorch.Sim3(data)

        moved = set()
        for result in graph.poll():
            current = {
                k: self.keyframes[k].T_WC.data.reshape(8).double().cpu().numpy()
                for k in result["keyframes"]
            }
            changes = corrections(result, current)
            with torch.inference_mode():
                for k, (_, pose) in changes.items():
                    self.keyframes[k].T_WC = sim3(pose)
                    graph.owned.add(k)
                if self.reference in changes and self.previous is not None:
                    previous = self.previous.data.reshape(8).double().cpu().numpy()
                    increment = changes[self.reference][0]
                    self.previous = sim3(increment @ sim3_matrix(previous)[0])
            moved |= set(changes)
        graph.reference = self.reference
        return moved

    def link(self, new):
        """The global graph links and solves keyframes (on its worker thread)."""
