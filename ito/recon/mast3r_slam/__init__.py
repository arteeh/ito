"""MASt3R-SLAM (Murai, Dexheimer and Davison, CVPR 2025): dense SLAM from the camera alone, its
map drawn as Gaussian splats, restored from main (ito/reconstruction and ito/build.py at
6426a50) beside the ToF mesh methods: its own worker (worker.py), splat renderer (render.py)
and bundle (build.py: MASt3R's weights and retrieval, CC BY-NC-SA 4.0, non-commercial;
MASt3R-SLAM's sources and its CUDA kernels, compiled for this build's Torch).
"""

from ito.recon import missing_files

FOLDER = "mast3r_slam"
WEIGHTS = {}  # not single downloads: build.py makes the whole bundle


def missing(size=None):
    return missing_files(FOLDER, ["manifest.json", "model.safetensors"])


def start(settings, intrinsics, *, capture_delay=0.0):
    """MASt3R-SLAM on the camera frames the pilot hands on (the window's straightened frames,
    strided as ito.tofmesh.worker.frame_size)."""
    from ito.tofmesh.worker import frame_size

    from .worker import Mast3rSlam

    (width, height), step = frame_size(intrinsics.width, intrinsics.height)
    frame = intrinsics.model_copy(
        update=dict(
            width=width,
            height=height,
            fx=intrinsics.fx / step,
            fy=intrinsics.fy / step,
            cx=(intrinsics.cx - step // 2) / step,
            cy=(intrinsics.cy - step // 2) / step,
        )
    )
    return Mast3rSlam(
        frame,
        max_splats=settings.max_splats,
        precision=settings.slam_precision,
        input_resolution=settings.slam_input_resolution,
        keyframes=settings.slam_keyframes,
    )
