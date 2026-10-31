"""Head IMU as a rotation prior for MASt3R-SLAM's tracking (proof of concept, #81 part 1).

Opt-in: ITO_HEAD_IMU=1 uses it, ITO_HEAD_IMU=log only records what it would predict (for
comparison runs); unset, the worker never reads the IMU and SLAM is upstream's.

Upstream's tracker starts each frame from the last frame's pose, and starts matching from the
last frame's pixel correspondences (or, right after a new keyframe, from "every pixel where it
was"). Both assume the camera hardly moved since. In a fast head turn it moved 10-25 degrees
between the frames the worker takes, and tracking fails. The head gyro says how far it turned:
- pose: the frame starts from the last tracked pose turned by the gyro's rotation since then;
- matching: each keyframe pixel's ray, turned by the predicted keyframe-to-frame rotation and
  projected through the camera, is its first guess in the frame (rotation only: parallax and
  translation are left to upstream's iterative matching, which refines from there).
Both are only starting points; upstream's matching and pose solve decide.

Clocks: samples are on the pilot clock (the driver's mapped like any of its stamps); a frame is
stamped with its capture time minus DELAY (ITO_HEAD_IMU_DELAY, seconds: a frame stamped on
arrival rather than exposure arrives that much after it).
"""

import math
import os

import numpy as np

MODE = os.environ.get("ITO_HEAD_IMU", "")
DELAY = float(os.environ.get("ITO_HEAD_IMU_DELAY", "0"))
SIZE = 4096  # samples kept: about a minute at the LSM6DSV16X's 60 Hz as robotd reads it
# The gyro's rotation since the last tracked frame above which matching starts from it rather
# than from the last frame's correspondences (or a new keyframe's identity guess).
SEED_DEG = 2.0
# Relocalising: how long the gyro may dead-reckon the camera's turn since it was last tracked
# (its bias, +-1 deg/s on the LSM6DSV16X, adds up), and how far a keyframe may be turned from
# that prediction to be offered to relocalisation beside retrieval's.
LOST_MAX = 10.0
NEAR_DEG = 35.0
NEAR = 2  # keyframes offered at most
# Held sample rate beyond the last sample before giving up on predicting (seconds).
EXTRAPOLATE = 0.1
OPENCV = np.diag([1.0, -1.0, -1.0])  # Ito camera axes to OpenCV's (a rotation: a vector map)


def enabled():
    return MODE in ("1", "log")


def prior():
    return MODE == "1"


def exp(rotvec):
    """Rodrigues: rotation vector to 3x3."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    k = rotvec / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def angle_deg(rotation):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1) / 2))))


def quaternion_xyzw(m):
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = (s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s)
    q = np.array(q)
    return q / np.linalg.norm(q)


def to_opencv(mount, rates):
    """Gyro rates (N x 3, the sensor's axes) as the camera's, in OpenCV axes. mount: the
    protocol's sensor-to-camera pose as a 4x4 (Ito camera axes)."""
    return np.asarray(rates, np.float64) @ (OPENCV @ np.asarray(mount)[:3, :3]).T


class Gyro:
    """The camera's angular velocity over time (OpenCV camera axes, rad/s), and the rotation it
    turned through between two times."""

    def __init__(self):
        self.t = np.full(SIZE, -np.inf)
        self.w = np.zeros((SIZE, 3))
        self.n = 0

    def add(self, times, rates):
        times, rates = np.asarray(times, np.float64), np.asarray(rates, np.float64)
        if not len(times):
            return
        keep = times > (self.t[(self.n - 1) % SIZE] if self.n else -np.inf)
        for t, w in zip(times[keep], rates[keep], strict=True):
            self.t[self.n % SIZE], self.w[self.n % SIZE] = t, w
            self.n += 1

    def rotation(self, t0, t1):
        """R_{C(t0) C(t1)}: each sample's rate held over half a sample period either side of it
        (best against the sim's truth: 0.4 deg median error on 11 deg turns). None if the
        samples do not reach both ends."""
        if self.n < 2 or t1 <= t0:
            return None if self.n < 2 else np.eye(3)
        count = min(self.n, SIZE)
        order = (np.arange(self.n - count, self.n)) % SIZE
        t, w = self.t[order], self.w[order]
        if t0 < t[0] or t1 > t[-1] + EXTRAPOLATE:
            return None
        edges = np.concatenate(([-np.inf], (t[1:] + t[:-1]) / 2, [np.inf]))
        first = max(0, int(np.searchsorted(edges, t0, "right")) - 1)
        last = min(len(t) - 1, int(np.searchsorted(edges, t1, "right")) - 1)
        R = np.eye(3)
        for i in range(first, last + 1):
            span = min(t1, edges[i + 1]) - max(t0, edges[i])
            if span > 0:
                R = R @ exp(w[i] * span)
        return R


class HeadPrior:
    """What slam.SLAMBackend asks of the head IMU: the frame's starting pose and match guess."""

    def __init__(self, backend):
        self.b = backend
        self.gyro = Gyro()
        self.last = None  # (time, T_WC data) of the last tracked frame
        self.log = None
        timeline = os.environ.get("ITO_TIMELINE")
        if timeline:
            self.log = open(timeline + ".imu.jsonl", "a", buffering=1)  # noqa: SIM115
        h, w = backend.input.height, backend.input.width
        v, u = np.mgrid[:h, :w].reshape(2, -1).astype(np.float64)
        K = backend.input.K.astype(np.float64)
        self.K, self.h, self.w = K, h, w
        self.rays = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)))
        self.seeded = self.frames = 0
        self.row = {}

    def add(self, batch):
        times, rates = batch
        self.gyro.add(times, rates)

    def rotation_of(self, data):
        """The rotation part of an upstream Sim(3) (lietorch data, 1 x 8: t, q xyzw, s)."""
        x, y, z, w = data.reshape(-1)[3:7].tolist()
        n = math.sqrt(x * x + y * y + z * z + w * w)
        x, y, z, w = x / n, y / n, z / n, w / n
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    def start(self, now, T_WC, tracking):
        """Before tracking a frame captured at now: the pose to start from (lietorch Sim3, or
        None to keep upstream's) and the match guess (1 x HW indices, or None)."""
        b, torch, lietorch = self.b, self.b.torch, self.b.lietorch
        now -= DELAY
        self.frames += 1
        self.row = dict(frame=b.i, t=round(now, 4), mode=str(b.states.get_mode()).split(".")[-1])
        if not tracking or self.last is None:
            return None, None
        then, last_data = self.last
        R = self.gyro.rotation(then, now)
        if R is None:
            self.row["gyro"] = "missing"
            return None, None
        turned = angle_deg(R)
        self.row["turn_since_last_deg"] = round(turned, 2)
        self.row["since_last_s"] = round(now - then, 3)
        delta = torch.tensor(
            [[0.0, 0.0, 0.0, *quaternion_xyzw(R).tolist(), 1.0]],
            device=last_data.device,
            dtype=last_data.dtype,
        )
        predicted = lietorch.Sim3(last_data) * lietorch.Sim3(delta)
        keyframes = b.keyframes
        k = len(keyframes) - 1
        R_kf = self.rotation_of(keyframes.T_WC[k].cpu()).T @ self.rotation_of(predicted.data.cpu())
        self.row["turn_from_keyframe_deg"] = round(angle_deg(R_kf), 2)
        self.predicted = self.rotation_of(predicted.data.cpu())
        if not prior():
            return None, None
        guess = None
        if turned > SEED_DEG or (b.tracker.idx_f2k is None and angle_deg(R_kf) > SEED_DEG):
            guess = self.guess(R_kf.T)
            self.seeded += 1
            self.row["seeded"] = True
        T_WC = lietorch.Sim3(predicted.data.clone())
        return T_WC, guess

    def reloc_hint(self, now):
        """For a lost frame captured at now: the camera's pose dead-reckoned by the gyro from the
        last tracked one (lietorch data), and the keyframes nearest it in view direction (and
        place, over their depth). Relocalisation tries those beside retrieval's and starts the
        new keyframe from that pose. None when the gyro cannot say."""
        b, torch, lietorch = self.b, self.b.torch, self.b.lietorch
        now -= DELAY
        if self.last is None or now - self.last[0] > LOST_MAX:
            return None
        then, last_data = self.last
        R = self.gyro.rotation(then, now)
        if R is None:
            return None
        delta = torch.tensor(
            [[0.0, 0.0, 0.0, *quaternion_xyzw(R).tolist(), 1.0]],
            device=last_data.device,
            dtype=last_data.dtype,
        )
        predicted = (lietorch.Sim3(last_data) * lietorch.Sim3(delta)).data
        n = len(b.keyframes)
        poses = b.keyframes.T_WC[:n, 0].cpu().double().numpy()
        p = predicted.reshape(-1).cpu().double().numpy()
        q = poses[:, 3:7] / np.linalg.norm(poses[:, 3:7], axis=1, keepdims=True)
        qp = p[3:7] / np.linalg.norm(p[3:7])
        turned = np.degrees(2 * np.arccos(np.clip(np.abs(q @ qp), 0, 1)))
        depth = b.depth[:n].cpu().double().numpy() * np.maximum(poses[:, 7], 1e-6)
        apart = np.linalg.norm(poses[:, :3] - p[:3], axis=1) / np.maximum(depth, 1e-6)
        score = np.radians(turned) + apart
        near = [int(k) for k in np.argsort(score) if turned[k] < NEAR_DEG][:NEAR]
        self.row["hint"] = near
        self.row["hint_turn_deg"] = [round(float(turned[k]), 1) for k in near]
        self.row["dead_reckoned_s"] = round(now - then, 2)
        return near, predicted.clone()

    def guess(self, R_fk):
        """Each keyframe pixel's ray turned into the frame and projected: its frame pixel."""
        r = R_fk @ self.rays
        z = np.where(r[2] > 1e-3, r[2], np.nan)
        K = self.K
        u = K[0, 0] * r[0] / z + K[0, 2]
        v = K[1, 1] * r[1] / z + K[1, 2]
        # Behind the camera or off the frame: the nearest edge pixel, from which upstream's
        # matching starts and which its occlusion test rejects if it is no match.
        u = np.clip(np.nan_to_num(u, nan=K[0, 2]), 0, self.w - 1)
        v = np.clip(np.nan_to_num(v, nan=K[1, 2]), 0, self.h - 1)
        index = np.rint(v).astype(np.int64) * self.w + np.rint(u).astype(np.int64)
        return self.b.torch.as_tensor(index[None], device=self.b.axes.device)

    def tracked(self, now, frame, outcome):
        """After the frame: remember a tracked one's pose; log what happened."""
        now -= DELAY
        if outcome == "tracked":
            if hasattr(self, "predicted") and "turn_since_last_deg" in self.row:
                error = self.predicted.T @ self.rotation_of(frame.T_WC.data.cpu())
                self.row["gyro_vs_tracked_deg"] = round(angle_deg(error), 2)
            self.last = (now, frame.T_WC.data.clone())
        self.row["outcome"] = outcome
        stats = getattr(self.b, "match_stats", None)
        if stats:
            self.row.update(stats)
            self.b.match_stats = None
        if self.log is not None:
            import json

            self.log.write(json.dumps(self.row) + "\n")
        self.row = {}
        if hasattr(self, "predicted"):
            del self.predicted

    def relocalised(self, now, T_WC_data):
        """A relocalisation: tracking resumes from that keyframe's pose, now."""
        self.last = (now - DELAY, T_WC_data.clone())
