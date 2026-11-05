"""Depth Anything 3 (ByteDance Seed, 2025): a depth map from each camera frame alone.

DA3Metric-Large's maps are metric (Apache-2.0); the ToF only corrects their scale.
"""

from ito.recon import missing_files, models

# Size -> the model's folder: its name under huggingface.co/depth-anything.
SIZES = {"metric": "DA3METRIC-LARGE"}
WEIGHTS = {
    "DA3METRIC-LARGE": (
        "depth-anything/DA3METRIC-LARGE",
        "4010e39f3634a45bc60553321fb49fb760bd594e",
        "Apache-2.0",
        {
            "README.md": "4fdd25492cf6401d1459483fd829f6abb399bda8218e1a3ef5ad4a66e9aa5a2a",
            "config.json": "a336f3e76fe375aaae17a9aed9130c9f2aa061535d317ec57dcb2f1f02e1dd53",
            "model.safetensors": "bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776",
        },
    ),
}


def folder(name):
    return models() / name


def missing(size):
    return missing_files(SIZES[size], ["model.safetensors"])


def start(settings, intrinsics, *, capture_delay=0.0):
    from ito.tofmesh.worker import TofMesh

    return TofMesh.for_model("da3", settings.da3_size, settings, intrinsics, capture_delay)
