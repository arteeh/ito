"""Register new evidence to retained observations in the persistent map frame."""

import numpy as np

AXES = np.diag([1.0, -1.0, -1.0, 1.0])


class Registration:
    def __init__(self):
        import cv2

        self.cv = cv2
        cv2.setNumThreads(1)
        self.map_from_estimator = np.eye(4)
        self.accepted = 0
        self.frames = {}

    def remember(self, node, estimated, mapped, live_nodes):
        self.frames = {key: value for key, value in self.frames.items() if key in live_nodes}
        self.frames[node] = (estimated.copy(), mapped.copy())

    def revise(self, poses, views):
        for node in reversed(self.frames):
            if node in poses:
                self.map_from_estimator = self.frames[node][1] @ np.linalg.inv(poses[node])
                for key, pose in poses.items():
                    if key in self.frames:
                        self.frames[key] = (pose.copy(), self.frames[key][1])
                return
        for view in reversed(views):
            if view.node in poses:
                self.map_from_estimator = view.pose @ np.linalg.inv(poses[view.node])
                break

    def align(self, image, depth, estimated, intrinsics, views):
        cv = self.cv
        predicted = self.map_from_estimator @ estimated
        if not views:
            return predicted, 1.0
        fx, fy, cx, cy = intrinsics
        calibration = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        candidates = []
        for view in views:
            cosine = view.pose[:3, 2] @ predicted[:3, 2]
            distance = np.linalg.norm(view.pose[:3, 3] - predicted[:3, 3])
            if cosine > 0.9 and distance < 0.5 and view.depth.shape == depth.shape:
                candidates.append((1 - cosine + distance * 0.1, view))
        candidates.sort(key=lambda item: item[0])
        for _, view in candidates[:2]:
            if view.features is None:
                source = cv.cvtColor(view.image, cv.COLOR_RGB2GRAY)
                corners = cv.goodFeaturesToTrack(source, 600, 0.01, 6)
                if corners is None:
                    continue
                xy = corners[:, 0]
                x, y = np.rint(xy).astype(int).T
                d = view.depth[y, x]
                keep = d > 0
                xy, d = xy[keep], d[keep]
                vx, vy, vcx, vcy = view.intrinsics
                local = np.column_stack(((xy[:, 0] - vcx) / vx * d, -(xy[:, 1] - vcy) / vy * d, -d))
                world = local @ view.pose[:3, :3].T + view.pose[:3, 3]
                view.features = source, np.ascontiguousarray(xy, dtype=np.float32), world
            source, xy, world = view.features
            if len(xy) < 40:
                continue
            camera = np.linalg.inv(predicted @ AXES)
            local = world @ camera[:3, :3].T + camera[:3, 3]
            projected = local[:, :2] / np.maximum(local[:, 2:3], 0.01)
            projected = projected * [fx, fy] + [cx, cy]
            flow, ok, _ = cv.calcOpticalFlowPyrLK(
                source,
                gray,
                xy[:, None],
                np.ascontiguousarray(projected[:, None], dtype=np.float32),
                winSize=(21, 21),
                maxLevel=3,
                flags=cv.OPTFLOW_USE_INITIAL_FLOW,
                criteria=(cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 20, 0.01),
            )
            back, back_ok, _ = cv.calcOpticalFlowPyrLK(
                gray,
                source,
                flow,
                None,
                winSize=(21, 21),
                maxLevel=3,
            )
            keep = ok[:, 0].astype(bool) & back_ok[:, 0].astype(bool)
            keep &= np.linalg.norm(back[:, 0] - xy, axis=1) < 0.75
            keep &= local[:, 2] > 0.01
            points, pixels = world[keep], flow[keep, 0]
            if len(points) < 40:
                continue
            rvec = cv.Rodrigues(camera[:3, :3])[0]
            tvec = camera[:3, 3].copy()
            success, rvec, tvec, inliers = cv.solvePnPRansac(
                points,
                pixels,
                calibration,
                None,
                rvec,
                tvec,
                True,
                iterationsCount=60,
                reprojectionError=2,
                confidence=0.99,
                flags=cv.SOLVEPNP_ITERATIVE,
            )
            if not success or inliers is None or len(inliers) < max(40, len(points) * 0.6):
                continue
            idx = inliers[:, 0]
            matched = pixels[idx]
            # A small repeated texture must not anchor the whole room.
            if (
                np.ptp(matched[:, 0]) < depth.shape[1] * 0.35
                or np.ptp(matched[:, 1]) < depth.shape[0] * 0.35
            ):
                continue
            rvec, tvec = cv.solvePnPRefineLM(points[idx], matched, calibration, None, rvec, tvec)
            camera[:3, :3] = cv.Rodrigues(rvec)[0]
            camera[:3, 3] = tvec.ravel()
            pose = np.linalg.inv(camera) @ AXES
            change = np.linalg.inv(predicted) @ pose
            angle = np.arccos(np.clip((np.trace(change[:3, :3]) - 1) / 2, -1, 1))
            if angle > np.radians(1) or np.linalg.norm(change[:3, 3]) > 0.02:
                continue
            x, y = np.rint(matched).astype(int).T
            inside = (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
            x, y = x[inside], y[inside]
            observed = depth[y, x]
            expected = (points[idx][inside] @ camera[:3, :3].T + camera[:3, 3])[:, 2]
            valid = (observed > 0) & (expected > 0)
            scale = float(np.median(expected[valid] / observed[valid])) if valid.any() else 1.0
            scale = np.clip(scale, 0.8, 1.25)
            self.accepted += 1
            return pose, scale
        return predicted, 1.0
