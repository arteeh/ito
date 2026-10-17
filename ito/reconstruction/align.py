"""Place a frame's metric depth onto the map's surfaces: projective point-to-plane ICP against
the map's depth rendered at the predicted pose, the way KinectFusion tracks against its model.
A frame placed by its own surfaces lands on the map's instead of beside it."""

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial.transform import Rotation

COVERED = 0.85  # rendered opacity of a map pixel to align against
SHARED = 0.3  # of the frame the map must cover, and of its points that must fit, to align
STEP = 4  # every STEP-th pixel each way
REACH = (0.10, 0.06, 0.04, 0.03, 0.02)  # metres: residuals beyond these are other surfaces
ITERATIONS = 8
LARGEST = (np.radians(8), 0.15)  # an alignment moving the frame further is a mismatch


def to_map(backend, points, valid, pose, intrinsics):
    """The map-from-camera pose that lays points (H x W x 3, camera frame, metres) onto the
    map's surfaces seen from pose, or None if the map does not show enough of them."""
    height, width = valid.shape
    ids, z, u, v = backend.centers(pose, intrinsics, (height, width))
    if len(ids) < 1000:
        return None
    alpha, depth, _ = backend.render(pose, intrinsics, (height, width), ids, z, u, v)
    covered = (alpha > COVERED) & (depth > 0)
    if covered.mean() < SHARED:
        return None
    fx, fy, cx, cy = intrinsics
    yy, xx = np.mgrid[:height, :width]
    surface = np.stack(((xx - cx) / fx * depth, -(yy - cy) / fy * depth, -depth), -1)
    normals = np.zeros_like(surface)
    normals[1:-1, 1:-1] = np.cross(
        surface[1:-1, 2:] - surface[1:-1, :-2], surface[:-2, 1:-1] - surface[2:, 1:-1]
    )
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= np.maximum(length, 1e-12)
    # Facing the camera, and not across a silhouette.
    normals *= np.where((normals * surface).sum(-1, keepdims=True) > 0, -1, 1)
    span = maximum_filter(depth, size=3) - minimum_filter(depth, size=3)
    usable = minimum_filter(covered, size=3) & (span < 0.05 * depth) & (length[..., 0] > 0)
    mine = points[::STEP, ::STEP][valid[::STEP, ::STEP]].astype(np.float64)
    if len(mine) < 500:
        return None
    rotation, translation = np.eye(3), np.zeros(3)
    fit = np.zeros(len(mine), bool)
    for iteration in range(ITERATIONS):
        x = mine @ rotation.T + translation
        ahead = -x[:, 2]
        safe = np.maximum(ahead, 1e-3)
        pu = np.rint(cx + fx * x[:, 0] / safe).astype(int)
        pv = np.rint(cy - fy * x[:, 1] / safe).astype(int)
        fit = (ahead > 0.05) & (pu >= 0) & (pu < width) & (pv >= 0) & (pv < height)
        pu, pv = np.where(fit, pu, 0), np.where(fit, pv, 0)
        fit &= usable[pv, pu]
        n = normals[pv, pu]
        residual = ((x - surface[pv, pu]) * n).sum(-1)
        reach = REACH[min(iteration, len(REACH) - 1)]
        fit &= np.abs(residual) < reach
        if fit.sum() < SHARED * len(mine):
            return None
        xs, ns, rs = x[fit], n[fit], residual[fit]
        jacobian = np.hstack((np.cross(xs, ns), ns))
        weight = np.minimum(1.0, 0.5 * reach / np.maximum(np.abs(rs), 1e-9))  # Huber
        hessian = (jacobian * weight[:, None]).T @ jacobian
        hessian += np.eye(6) * 1e-3 * np.trace(hessian) / 6
        step = -np.linalg.solve(hessian, (jacobian * weight[:, None]).T @ rs)
        turn = Rotation.from_rotvec(step[:3]).as_matrix()
        rotation, translation = turn @ rotation, turn @ translation + step[3:]
        if np.linalg.norm(step) < 1e-5:
            break
    angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
    if angle > LARGEST[0] or np.linalg.norm(translation) > LARGEST[1]:
        return None
    moved = np.eye(4)
    moved[:3, :3], moved[:3, 3] = rotation, translation
    return pose @ moved
