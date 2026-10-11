"""CPU geometry checks and a jittered RGB-D stream through the production fusion.

uv run python e2e/sharp_fusion.py
"""

import json

import numpy as np

from ito.protocol import Intrinsics
from ito.reconstruction.rgbd import RGBDBackend
from ito.reconstruction.surface import ellipse, full_depth, orientation, patches, surfaces


def main():
    depth = np.arange(24, dtype=np.float32).reshape(4, 6) + 1000
    np.testing.assert_array_equal(full_depth(depth, depth.shape, (1, 1, 0, 0), np), depth)
    enlarged = full_depth(depth, (8, 12), (2, 2, 0, 0), np)
    np.testing.assert_allclose(enlarged[2, 2], depth[1, 1])
    np.testing.assert_allclose(enlarged[3, 3], depth[1:3, 1:3].mean())
    cropped = full_depth(depth, (8, 16), (2, 2, 1, 0), np)
    assert not cropped[:, :2].any()
    step = np.ones((4, 6), np.float32)
    step[:, 3:] = 5
    edge = full_depth(step, (8, 12), (2, 2, 0, 0), np)
    assert set(np.unique(edge)) == {0, 1, 5}, "interpolation bridged a depth edge"
    for normal in ([0, 0, 1], [0, 0, -1], [1, 0, 0], [0.6, 0, 0.8]):
        q = orientation(np.array([normal], float), np)[0]
        w, x, y, z = q
        np.testing.assert_allclose(
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], normal, atol=1e-6
        )

    rng = np.random.default_rng(34)
    derivatives = rng.normal(size=(100, 2, 3)).astype(np.float32)
    tangent, scales = ellipse(derivatives, np)
    normals = np.cross(derivatives[:, 0], derivatives[:, 1])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    q = orientation(normals, np, tangent)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-5)
    w, x, y, z = q.T
    major = np.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), axis=1)
    minor = np.cross(normals, major)
    basis = np.stack((major, minor), axis=1) * scales[..., None]
    np.testing.assert_allclose(
        basis.transpose(0, 2, 1) @ basis,
        derivatives.transpose(0, 2, 1) @ derivatives / 4,
        rtol=1e-4,
        atol=1e-5,
    )

    k = Intrinsics(width=160, height=120, fx=150, fy=150, cx=79.5, cy=59.5)
    backend = RGBDBackend(40_000, k, device="cpu", voxel_size=0.005)
    rgb = np.full((k.height, k.width, 3), 128, np.uint8)
    depth = np.full(rgb.shape[:2], 2, np.float32)
    points = backend.rays * depth[..., None]
    normal, size, valid, axes = surfaces(points, depth > 0, np)
    assert valid.all()
    np.testing.assert_allclose(size, 2 / 150, rtol=1e-5)
    np.testing.assert_allclose(np.abs(normal[..., 2]), 1)
    sample_points, sample_rgb, _, sample_sizes, _, _ = patches(
        points, rgb, normal, size, valid, 0.05, np, axes
    )
    assert len(sample_points) == k.width * k.height // 4
    np.testing.assert_allclose(sample_sizes, 4 / 150, rtol=1e-5)
    edge_rgb = rgb.copy()
    edge_rgb[:, 79:] = 200
    _, sample_rgb, _, sample_sizes, _, _ = patches(
        points, edge_rgb, normal, size, valid, 0.05, np, axes
    )
    assert set(np.unique(sample_rgb)) == {128, 200}, "patch merger blurred a colour edge"
    assert np.min(sample_sizes) < 0.02, "colour edges lost their pixel footprints"
    rng = np.random.default_rng(34)
    counts, errors = [], []
    for frame in range(30):
        camera = np.eye(4, dtype=np.float32)
        camera[:3, 3] = rng.normal(0, 0.002, 3)
        noisy = depth + rng.normal(0, 0.001, depth.shape).astype(np.float32)
        backend.expire(frame * 0.1, np.inf)
        backend.integrate(rgb, noisy, camera, frame * 0.1)
        # Emulate acknowledgment of the production ring's retirement messages.
        backend.retire_revision[backend.retiring] = 0
        active = (backend.keys >= 0) & ~backend.retiring
        counts.append(int(active.sum()))
        errors.append(float(np.mean(np.abs(backend.records[active, 0, 2] + 2))))
    assert np.median(backend.confidence[active]) > 5, "observations did not fuse"
    assert counts[-1] < counts[0] * 1.5, counts
    assert np.mean(errors[-5:]) < np.mean(errors[:5]), errors
    assert np.median(backend.records[active, 1, 2]) < 0.001
    # A plane displaced toward the camera must disappear when the real plane returns.
    backend.integrate(rgb, depth * 0.75, np.eye(4), 4)
    backend.integrate(rgb, depth, np.eye(4), 5)
    active = (backend.keys >= 0) & ~backend.retiring
    assert np.mean(backend.records[active, 0, 2] > -1.8) < 0.01
    print(
        json.dumps(
            dict(
                first_count=counts[0],
                last_count=counts[-1],
                initial_error_m=np.mean(errors[:5]),
                final_error_m=np.mean(errors[-5:]),
            )
        )
    )
    coarse = RGBDBackend(40_000, k, device="cpu")
    for frame in range(30):
        camera = np.eye(4, dtype=np.float32)
        camera[0, 3] = frame * 0.001
        coarse.integrate(rgb, depth, camera, frame * 0.1)
        if frame == 0:
            first = coarse.count
    assert coarse.count < 2 * first, (first, coarse.count)
    print(json.dumps(dict(patch_first_count=first, patch_last_count=coarse.count)))
    print("PASS: resize/crop, depth edges, normals, jitter fusion, patch fusion and carving")


if __name__ == "__main__":
    main()
