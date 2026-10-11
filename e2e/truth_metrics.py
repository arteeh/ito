"""Metric coordinates and appearance shared by source-truth end-to-end runs."""

import numpy as np
from scipy.ndimage import laplace
from scipy.spatial.transform import Rotation
from skimage.metrics import structural_similarity


def alignment(estimated, truth):
    a, b = estimated[:, :3, 3], truth[:, :3, 3]
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, singular, vt = np.linalg.svd(bc.T @ ac / len(a))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    variance = np.mean(np.sum(ac * ac, axis=1))
    if variance < 1e-12 or np.linalg.matrix_rank(ac, tol=1e-5) < 2:
        raise ValueError("Camera-centre Sim3 is unobservable; cannot report a fitted metric scale")
    scale = float(singular @ sign / variance)
    translation = b.mean(0) - scale * rotation @ a.mean(0)
    aligned = estimated.copy()
    aligned[:, :3, :3] = rotation @ estimated[:, :3, :3]
    aligned[:, :3, 3] = scale * a @ rotation.T + translation
    ate = np.linalg.norm(aligned[:, :3, 3] - b, axis=1)
    angle = Rotation.from_matrix(aligned[:, :3, :3].transpose(0, 2, 1) @ truth[:, :3, :3])
    de = np.linalg.inv(aligned[:-1]) @ aligned[1:]
    dt = np.linalg.inv(truth[:-1]) @ truth[1:]
    residual = np.linalg.inv(dt) @ de
    report = dict(
        method="Umeyama corresponding camera centres, no surface fitting",
        scale=scale,
        rotation=rotation.tolist(),
        translation_m=translation.tolist(),
        matches=len(a),
        singular_values=singular.tolist(),
        ate_rmse_m=float(np.sqrt(np.mean(ate**2))),
        ate_p95_m=float(np.percentile(ate, 95)),
        orientation_p95_deg=float(np.percentile(angle.magnitude(), 95) * 180 / np.pi),
        rpe_translation_rmse_m=float(np.sqrt(np.mean(np.sum(residual[:, :3, 3] ** 2, axis=1)))),
        rpe_rotation_rmse_deg=float(
            np.sqrt(np.mean(Rotation.from_matrix(residual[:, :3, :3]).magnitude() ** 2))
            * 180
            / np.pi
        ),
        rpe_interval="consecutive successfully tracked source frames",
    )
    return scale, rotation, translation, aligned, report


def transform_records(records, scale, rotation, translation):
    result = records.copy()
    result[:, 0, :3] = scale * result[:, 0, :3] @ rotation.T + translation
    result[:, 1, :3] *= scale
    q = Rotation.from_quat(result[:, 2, [1, 2, 3, 0]])
    result[:, 2] = (Rotation.from_matrix(rotation) * q).as_quat()[:, [3, 0, 1, 2]]
    return np.ascontiguousarray(result)


def unproject(depth, k, camera):
    y, x = np.indices(depth.shape)
    rays = np.stack(((x - k[0, 2]) / k[0, 0], -(y - k[1, 2]) / k[1, 1], -np.ones_like(x)), -1)
    points = (rays * depth[..., None]) @ camera[:3, :3].T + camera[:3, 3]
    return points


def image_metrics(image, truth, alpha):
    a, b = image.astype(float) / 255, truth.astype(float) / 255
    energy = [float(np.mean(laplace(v.mean(2))[1:-1, 1:-1] ** 2)) for v in (a, b)]
    return dict(
        psnr_db=float(-10 * np.log10(max(np.mean((a - b) ** 2), 1e-15))),
        ssim=float(
            structural_similarity(
                a,
                b,
                data_range=1,
                channel_axis=2,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        ),
        coverage=float(np.mean(alpha > 0.5)),
        sharpness_ratio=energy[0] / max(energy[1], 1e-15),
    )
