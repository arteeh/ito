"""The 3D reconstruction methods a pilot chooses between in Settings, one per session.

Each method lives in its own package here. Importing one is cheap: its __init__ names the model
sizes and the weights the Windows release bundles (WEIGHTS: folder -> (Hugging Face repository,
revision, licence, {file: sha256})); the heavy code (Torch, the network) is in modules only the
session that chose it imports, in its own processes. Nothing of an unchosen method loads.

The depth methods (guessed, pda, da3, prior_da, mapanything) all feed the ToF mesh
(ito.tofmesh): the robot's ToF ranges fused into a mesh, painted with its camera frames. Guessed
depth places the rest of the camera's view by the renderer's guess; the others by a depth map a
network makes from each camera frame (ito.tofmesh.depth runs it). MASt3R-SLAM is a world of its
own: its own poses, Gaussian splats and renderer.

A session's handle (start()) is what the pilot calls: submit(image, camera, captured_at) a
camera frame, motion(), ranges(), imu(), poll() the newest scene update for the windows'
renderer (`scene` names which), view(now), status(), statistics(), alive(), close().
"""

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Method:
    label: str
    summary: str
    sizes: tuple = ()  # (value, label) of the model sizes offered, the setting's choices
    scene: str = "mesh"  # what the windows draw: "mesh" (ito.tofmesh) or "splats"
    poses: bool = False  # tracks the camera itself, and places the pilot's eye (handle.view)


METHODS = {
    "guessed": Method(
        "Guessed depth",
        "ToF mesh; beyond the ToF's view the renderer guesses depth from what it measured.",
    ),
    "pda": Method(
        "Prompt Depth Anything",
        "ToF mesh; a depth map from each camera frame, prompted with the ToF's ranges.",
        (("small", "Small"), ("large", "Large")),
    ),
    "da3": Method(
        "Depth Anything 3",
        "ToF mesh; a depth map from each camera frame, scaled by the ToF's ranges.",
        (
            ("small", "Small"),
            ("base", "Base"),
            ("large", "Large"),
            ("giant", "Giant"),
            ("metric", "Metric"),
        ),
    ),
    "prior_da": Method(
        "Prior Depth Anything",
        "ToF mesh; the ToF's ranges completed into a depth map by each camera frame.",
        (("small", "Small"), ("base", "Base"), ("large", "Large")),
    ),
    "mapanything": Method(
        "MapAnything",
        "ToF mesh; a metric depth map from each camera frame and its lens, checked by the ToF.",
    ),
    "mast3r_slam": Method(
        "MASt3R-SLAM",
        "Gaussian splats; its own camera poses and map from the camera alone (~11 GB VRAM).",
        scene="splats",
        poses=True,
    ),
}


def package(name):
    """The method's package (light: labels, weights, missing())."""
    if name not in METHODS:
        raise ValueError(f"Unknown reconstruction method {name!r}")
    return importlib.import_module(f"{__name__}.{name}")


def models():
    """Where the release keeps the models: its models folder, beside its runtime."""
    return Path(sys.prefix).parent / "models"


def missing_files(folder, files):
    """None if every file is in the models folder's `folder`, else the error to show."""
    for file in files:
        path = models() / folder / file
        if not path.is_file():
            return f"Model not found: {path}"
    return None


def missing(name, size=None):
    """Why the method cannot start (a model missing from this build), or None."""
    return package(name).missing(size)


def weights():
    """Every method's bundled weights: folder -> (Hugging Face repository, revision, licence,
    {file: sha256})."""
    found = {}
    for name in METHODS:
        found.update(package(name).WEIGHTS)
    return found


def renderer(scene, device):
    """The windows' renderer for a Method.scene, on a vk.Device."""
    if scene == "mesh":
        from ito.tofmesh.render import MeshRenderer

        return MeshRenderer(device)
    from ito.recon.mast3r_slam.render import SplatRenderer

    return SplatRenderer(device)


def start(settings, intrinsics, *, capture_delay=0.0):
    """The session's reconstruction, as the app settings chose it."""
    name = settings.reconstruction
    return package(name).start(settings, intrinsics, capture_delay=capture_delay)
