"""Pixel footprints and normals, on NumPy or CuPy without per-pixel Python work."""


def surfaces(points, valid, xp):
    # Choose the shorter one-sided difference: never bridge a foreground silhouette.
    dx = xp.zeros_like(points)
    dy = xp.zeros_like(points)
    for axis, target in ((1, dx), (0, dy)):
        forward = xp.roll(points, -1, axis=axis) - points
        backward = points - xp.roll(points, 1, axis=axis)
        good_forward = valid & xp.roll(valid, -1, axis=axis)
        good_backward = valid & xp.roll(valid, 1, axis=axis)
        edge = [slice(None), slice(None)]
        edge[axis] = -1
        good_forward[tuple(edge)] = False
        edge[axis] = 0
        good_backward[tuple(edge)] = False
        a = xp.where(good_forward, xp.sum(forward**2, axis=-1), xp.inf)
        b = xp.where(good_backward, xp.sum(backward**2, axis=-1), xp.inf)
        target[:] = xp.where((a < b)[..., None], forward, backward)
        target[~(good_forward | good_backward)] = 0
    normal = xp.cross(dx, dy)
    length = xp.linalg.norm(normal, axis=-1)
    normal /= xp.maximum(length[..., None], 1e-20)
    # The tangent area is the actual pixel's surface footprint; the Gaussian's
    # two-sigma diameter spans one sample, independent of the association tolerance.
    size = xp.sqrt(length)
    return normal, size, valid & (size > 1e-8) & xp.isfinite(size), xp.stack((dx, dy), axis=-2)


def orientation(normals, xp, tangent=None):
    """Unit wxyz quaternion rotating +Z onto the surface normal."""
    q = xp.stack((1 + normals[:, 2], -normals[:, 1], normals[:, 0], xp.zeros(len(normals))), axis=1)
    opposite = q[:, 0] < 1e-6
    q[opposite] = xp.asarray([0, 1, 0, 0])
    q /= xp.maximum(xp.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    if tangent is not None:
        w, x, y, z = q.T
        start = xp.stack(
            (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), axis=1
        )
        cosine = xp.clip(xp.sum(start * tangent, axis=1), -1, 1)
        sine = xp.sum(xp.cross(start, tangent) * normals, axis=1)
        c = xp.sqrt((1 + cosine) / 2)
        s = xp.copysign(xp.sqrt((1 - cosine) / 2), sine)
        q = xp.stack((w * c - z * s, x * c + y * s, y * c - x * s, z * c + w * s), axis=1)
    return q


def ellipse(axes, xp):
    """Principal tangent and Gaussian scales of the pixel-to-surface Jacobian."""
    dx, dy = axes[:, 0], axes[:, 1]
    a, b, c = xp.sum(dx * dx, axis=1), xp.sum(dx * dy, axis=1), xp.sum(dy * dy, axis=1)
    mid = (a + c) / 2
    radius = xp.sqrt(xp.maximum(0, ((a - c) / 2) ** 2 + b * b))
    major, minor = xp.maximum(mid + radius, 1e-16), xp.maximum(mid - radius, 1e-16)
    weights = xp.stack((b, major - a), axis=1)
    diagonal = xp.abs(b) < 1e-6 * xp.maximum(a, c)
    weights[diagonal, 0] = a[diagonal] >= c[diagonal]
    weights[diagonal, 1] = a[diagonal] < c[diagonal]
    weights /= xp.maximum(xp.linalg.norm(weights, axis=1, keepdims=True), 1e-30)
    tangent = dx * weights[:, :1] + dy * weights[:, 1:]
    tangent /= xp.maximum(xp.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    return tangent, xp.sqrt(xp.stack((major, minor), axis=1)) * 0.5


def full_depth(depth, shape, transform, xp):
    """Undo MASt3R's resize/crop on calibrated depth; do not invent cropped geometry.

    Bilinear interpolation is limited to a continuous surface. At depth edges the
    nearest sample wins, so full-resolution colour never paints a foreground halo.
    """
    sx, sy, crop_x, crop_y = transform
    y, x = xp.mgrid[: shape[0], : shape[1]]
    x = x.astype(xp.float32) / sx - crop_x
    y = y.astype(xp.float32) / sy - crop_y
    height, width = depth.shape
    inside = (x >= 0) & (y >= 0) & (x <= width - 1) & (y <= height - 1)
    x, y = xp.clip(x, 0, width - 1), xp.clip(y, 0, height - 1)
    x0, y0 = x.astype(xp.int32), y.astype(xp.int32)
    x1, y1 = xp.minimum(x0 + 1, width - 1), xp.minimum(y0 + 1, height - 1)
    a, b, c, d = depth[y0, x0], depth[y0, x1], depth[y1, x0], depth[y1, x1]
    lo, hi = (
        xp.minimum(xp.minimum(a, b), xp.minimum(c, d)),
        xp.maximum(xp.maximum(a, b), xp.maximum(c, d)),
    )
    dx, dy = x - x0, y - y0
    smooth = (a * (1 - dx) + b * dx) * (1 - dy) + (c * (1 - dx) + d * dx) * dy
    nearest = depth[xp.rint(y).astype(xp.int32), xp.rint(x).astype(xp.int32)]
    result = xp.where((lo > 0) & (hi - lo < 0.05 * lo), smooth, nearest)
    return xp.where(inside & xp.isfinite(result) & (result > 0), result, 0).astype(xp.float32)


def patches(points, colors, normals, sizes, valid, maximum, xp, axes):
    """Coalesce only flat, similarly coloured 2x2 patches; retain pixel leaves at edges.

    The footprint is the area actually observed by the merged samples, never an
    association/noise radius. A world-space ceiling bounds curvature linearization.
    """
    footprint = sizes.copy()
    selected = [[], [], [], [], []]
    level = 1
    while True:
        height, width = valid.shape
        h, w = height // 2, width // 2
        if not h or not w:
            for out, value in zip(selected, (points, colors, normals, sizes, axes), strict=True):
                out.append(value[valid])
            break

        def children(value, h=h, w=w):
            return xp.stack(
                [
                    value[dy : 2 * h : 2, dx : 2 * w : 2]
                    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))
                ]
            )

        ps, cs, ns, ss, derivatives = map(children, (points, colors, normals, sizes, axes))
        merged_axes = derivatives.mean(0) * 2
        center, color, normal = ps.mean(0), cs.astype(xp.float32).mean(0), ns.mean(0)
        normal /= xp.maximum(xp.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
        size = xp.sqrt(xp.sum(ss**2, axis=0))
        merge = (
            children(valid).all(0)
            & (size <= maximum)
            & ((cs.max(0).astype(xp.float32) - cs.min(0)).max(-1) <= 12)
            & (xp.abs(xp.sum(ns * normal[None], axis=-1)).min(0) > 0.98)
            & (
                xp.abs(xp.sum((ps - center[None]) * normal[None], axis=-1)).max(0)
                < xp.maximum(size * 0.05, 1e-6)
            )
        )
        keep = valid.copy()
        keep[: 2 * h, : 2 * w] &= ~merge.repeat(2, 0).repeat(2, 1)
        for out, value in zip(selected, (points, colors, normals, sizes, axes), strict=True):
            out.append(value[keep])
        if not bool(merge.any()):
            break
        level *= 2
        area = footprint[: h * level, : w * level]
        mask = merge.repeat(level, 0).repeat(level, 1)
        expanded = size.repeat(level, 0).repeat(level, 1)
        area[mask] = expanded[mask]
        points, colors, normals, sizes, valid, axes = (
            center,
            color,
            normal,
            size,
            merge,
            merged_axes,
        )
    return (*(xp.concatenate(values) for values in selected), footprint)
