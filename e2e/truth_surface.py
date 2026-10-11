"""Continuous rendered surfaces, in MuJoCo world metres (never a depth-sample union)."""

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class Surface:
    def __init__(self, model, data, groups=(0, 1, 4, 5)):
        import mujoco

        meshes, self.planes = [], []
        for i in range(model.ngeom):
            if model.geom_group[i] not in groups or model.geom_rgba[i, 3] == 0:
                continue
            rotation = data.geom_xmat[i].reshape(3, 3)
            centre = data.geom_xpos[i]
            kind = model.geom_type[i]
            if kind == mujoco.mjtGeom.mjGEOM_PLANE:
                self.planes.append((centre.copy(), rotation[:, 2].copy()))
                continue
            if kind != mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError(f"Surface must use the rendered mesh: {model.geom(i).name}")
            mesh = model.geom_dataid[i]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start : start + count] @ rotation.T + centre
            start, count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            faces = model.mesh_face[start : start + count]
            meshes.append(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
        self.mesh = trimesh.util.concatenate(meshes)
        self.query = trimesh.proximity.ProximityQuery(self.mesh)

    def distance(self, points):
        points = np.asarray(points, dtype=float)
        result = np.full(len(points), np.inf)
        for start in range(0, len(points), 4096):
            part = points[start : start + 4096]
            distance = self.query.on_surface(part)[1]
            for centre, normal in self.planes:
                distance = np.minimum(distance, np.abs((part - centre) @ normal))
            result[start : start + len(part)] = distance
        return result

    def metrics(self, points):
        distance = self.distance(points)
        return dict(
            points=len(points),
            median_m=float(np.median(distance)),
            p95_m=float(np.percentile(distance, 95)),
            ghost_fraction=float(np.mean(distance > 0.05)),
        )


def wall_histogram(points, *, axis, face, bounds, inward):
    """Signed distances in a finite bare-wall patch; peaks need 5 cm separation.

    Bounds exclude trim, corners and furniture. Both peaks must contain at least 10%
    of the dominant peak's support. This detects separated layers, not broad noise.
    """
    other = [i for i in range(3) if i != axis]
    keep = np.ones(len(points), bool)
    for i, (low, high) in zip(other, bounds, strict=True):
        keep &= (points[:, i] >= low) & (points[:, i] <= high)
    distance = inward * (points[keep, axis] - face)
    edges = np.arange(-0.4, 0.405, 0.005)
    counts, _ = np.histogram(distance, edges)
    smooth = gaussian_filter1d(counts.astype(float), 1)
    peaks, _ = find_peaks(smooth, distance=10, prominence=max(5, smooth.max() * 0.1))
    peaks = peaks[smooth[peaks] >= smooth.max() * 0.1]
    centres = (edges[:-1] + edges[1:]) / 2
    return dict(
        points=int(keep.sum()),
        bin_edges_m=edges.tolist(),
        counts=counts.tolist(),
        peaks_m=centres[peaks].tolist(),
        doubled=len(peaks) >= 2,
    )
