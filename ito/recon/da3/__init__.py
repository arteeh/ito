"""Depth Anything 3 (ByteDance Seed, 2025): a depth map from each camera frame alone.

DA3Metric-Large's maps are metric (Apache-2.0); the ToF only corrects their scale. The main
series (Small and Base, Apache-2.0; Large 1.1, Apache-2.0; Giant 1.1, CC-BY-NC-4.0) give depth
up to a scale, which the ToF's ranges fix in every map (a map without enough of them is skipped).
"""

from ito.recon import missing_files, models

# Size -> the model's folder: its name under huggingface.co/depth-anything.
SIZES = {
    "small": "DA3-SMALL",
    "base": "DA3-BASE",
    "large": "DA3-LARGE-1.1",
    "giant": "DA3-GIANT-1.1",
    "metric": "DA3METRIC-LARGE",
}
WEIGHTS = {
    "DA3-SMALL": (
        "depth-anything/DA3-SMALL",
        "e08cab65ca0ec38e7826075418411ab90cab4da3",
        "Apache-2.0",
        {
            "README.md": "0a711be055e4eb5293e31586e3176c09ce12eb2e79593f6b96a41f7fe5b415d0",
            "config.json": "a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2",
            "model.safetensors": "364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf",
        },
    ),
    "DA3-BASE": (
        "depth-anything/DA3-BASE",
        "f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
        "Apache-2.0",
        {
            "README.md": "5388563071b4361b11d6d811b354d454f26aed2adba634fcf4ee3775c8eda480",
            "config.json": "5e34115ebc17bd2d8d43033c5f72e9446ac8833fd61d3fa160b7e67e0bb5b7b5",
            "model.safetensors": "e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5",
        },
    ),
    "DA3-LARGE-1.1": (
        "depth-anything/DA3-LARGE-1.1",
        "0e109ae307c5982f319a67cf6f9f99ccdc0ec97c",
        "Apache-2.0",
        {
            "README.md": "2d323de54b5e27e73762486d1b6e509ab8ac32b6fcd6fbf87c17661671a6a1f0",
            "config.json": "744dcaf53859490ed92fc6cb98d68d3daf624b8c54533aaf604bdb53f06321f5",
            "model.safetensors": "739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64",
        },
    ),
    "DA3-GIANT-1.1": (
        "depth-anything/DA3-GIANT-1.1",
        "72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19",
        "CC-BY-NC-4.0",
        {
            "README.md": "b939f8754d5147d42e926f20073ed59912f751eeb7c727cd4697b0f771c1c487",
            "config.json": "74626a50d6dee2a11820291a4305c1a34aa5adc4f7260908bbdbc9a939ba8e93",
            "model.safetensors": "1e47a08338ca73a6d6a21d37fd060b26b993b672bc6ddf6295fe474df2592001",
        },
    ),
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
    return missing_files(SIZES[size], ["config.json", "model.safetensors"])


def start(settings, intrinsics, *, capture_delay=0.0):
    from ito.tofmesh.worker import TofMesh

    return TofMesh.for_model("da3", settings.da3_size, settings, intrinsics, capture_delay)
