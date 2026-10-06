"""World-space splats. All poses are right-handed: +X right, +Y up, -Z forward.

Packed float32 records are [xyz, opacity], [scale_xyz, 0], [quaternion_wxyz],
then 1/4/9/16 [SH_rgb, 0] vectors. A producer may put these records in shared
memory for file snapshots. Live sources publish SplatUpdate packets instead.
SceneSource.poll() must never wait for reconstruction: None retains the scene.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray
from plyfile import PlyData

if TYPE_CHECKING:
    from ito.reconstruction.ring import SplatUpdate

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class GaussianBuffer:
    records: FloatArray

    def __post_init__(self) -> None:
        a = self.records
        if (
            a.dtype != np.float32
            or not a.flags.c_contiguous
            or a.ndim != 3
            or a.shape[1] not in (4, 7, 12, 19)
            or a.shape[2] != 4
        ):
            raise ValueError("Gaussian records must be contiguous float32 (N, 3 + SH_count, 4)")
        if not np.isfinite(a).all():
            raise ValueError("Gaussian records contain non-finite values")
        if np.any((a[:, 0, 3] < 0) | (a[:, 0, 3] > 1)):
            raise ValueError("Gaussian opacity must be in [0, 1]")
        if np.any(a[:, 1, :3] <= 0):
            raise ValueError("Gaussian scales must be positive")
        if not np.allclose(np.linalg.norm(a[:, 2], axis=1), 1, atol=1e-4):
            raise ValueError("Gaussian rotations must be unit wxyz quaternions")

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def sh_degree(self) -> int:
        return (1, 4, 9, 16).index(self.records.shape[1] - 3)


@dataclass(frozen=True)
class GaussianFrame:
    gaussians: GaussianBuffer
    revision: int
    captured_at: float | None = None  # Pilot monotonic clock, after link clock correction.


class SceneSource(Protocol):
    def poll(self) -> "GaussianFrame | SplatUpdate | None": ...


def load_ply(path: str | Path) -> GaussianBuffer:
    """Read ASCII or binary standard 3DGS PLY; scales are logs, opacity is a logit."""
    try:
        vertices = PlyData.read(str(path))["vertex"].data
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid Gaussian PLY: {exc}") from exc
    names = set(vertices.dtype.names or ())
    required = ["x", "y", "z", "opacity"]
    required += [
        f"{prefix}_{i}" for prefix, n in (("scale", 3), ("rot", 4), ("f_dc", 3)) for i in range(n)
    ]
    missing = set(required) - names
    if missing:
        raise ValueError(f"Not a 3DGS PLY; missing fields: {', '.join(sorted(missing))}")
    rest = {name for name in names if name.startswith("f_rest_")}
    if len(rest) not in (0, 9, 24, 45) or rest != {f"f_rest_{i}" for i in range(len(rest))}:
        raise ValueError("PLY SH coefficients must form a complete degree 0, 1, 2 or 3")
    coefficients = 1 + len(rest) // 3
    records = np.zeros((len(vertices), 3 + coefficients, 4), dtype=np.float32)

    def columns(fields: list[str]) -> FloatArray:
        values = np.stack([vertices[name] for name in fields], axis=1).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("PLY contains non-finite Gaussian attributes")
        return values

    records[:, 0, :3] = columns(["x", "y", "z"])
    logits = columns(["opacity"])[:, 0]
    records[:, 0, 3] = np.exp(-np.logaddexp(0, -logits))
    with np.errstate(over="ignore", under="ignore"):
        records[:, 1, :3] = np.exp(columns([f"scale_{i}" for i in range(3)]))
    rotation = columns([f"rot_{i}" for i in range(4)]).astype(np.float64)
    lengths = np.linalg.norm(rotation, axis=1, keepdims=True)
    if np.any(lengths == 0):
        raise ValueError("PLY contains a zero rotation quaternion")
    records[:, 2] = rotation / lengths
    records[:, 3, :3] = columns([f"f_dc_{i}" for i in range(3)])
    if rest:
        # The PLY stores all red coefficients, then green, then blue.
        records[:, 4:, :3] = (
            columns([f"f_rest_{i}" for i in range(len(rest))])
            .reshape(-1, 3, coefficients - 1)
            .transpose(0, 2, 1)
        )
    return GaussianBuffer(records)
