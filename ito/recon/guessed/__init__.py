"""Guessed depth: the ToF mesh alone; beyond the ToF's view the renderer guesses where the
camera frame's pixels are from the surfaces the ToF measured and the floor (ito.tofmesh.render).
No model: nothing loads beyond the mesh worker."""

WEIGHTS = {}


def missing(size=None):
    return None


def start(settings, intrinsics, *, capture_delay=0.0):
    from ito.tofmesh.worker import TofMesh

    return TofMesh(intrinsics, settings.mesh_memory, capture_delay=capture_delay)
