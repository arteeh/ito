"""Prior Depth Anything ("Depth Anything with Any Prior", Wang et al. 2025): the ToF's real
ranges, a few dozen points in the camera frame, completed into a dense metric depth map with
the help of a monocular estimate from the frame (Depth Anything V2) and a network conditioned
on both.

Code Apache-2.0 (ito._vendor.prior_da); the weights come from its authors' Hugging Face
repository (Apache-2.0 on its card; its Depth Anything V2 Base and Large backbones are
CC-BY-NC-4.0 at their source, non-commercial).
"""

from ito.recon import missing_files

FOLDER = "prior-depth-anything"
# Size -> (frozen Depth Anything V2, conditioned model, its version): the conditioned model
# comes in Small and Base (Large is not released); Large pairs it with the large backbone.
SIZES = {
    "small": ("vits", "vits", "1.0"),
    "base": ("vitb", "vitb", "1.1"),
    "large": ("vitl", "vitb", "1.1"),
}
WEIGHTS = {
    FOLDER: (
        "Rain729/Prior-Depth-Anything",
        "25b3bfcf85209fb015e3b15259f4df8b6f13bb16",
        "Apache-2.0 (card); Depth Anything V2 Base/Large CC-BY-NC-4.0",
        {
            "README.md": "4bcf87ecfbbb8e07a01b21415a970c8b53a5283bf6872b657040d3f45c9241f7",
            "depth_anything_v2_vits.pth": (
                "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
            ),
            "depth_anything_v2_vitb.pth": (
                "0d2b7002e62d39d655571c371333340bd88f67ab95050c03591555aa05645328"
            ),
            "depth_anything_v2_vitl.pth": (
                "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345"
            ),
            "prior_depth_anything_vits.pth": (
                "b359539a8c02d9426c48f92f4dd472e81c99e1a87c6030b7ec26427a8263da0b"
            ),
            "prior_depth_anything_vitb_1_1.pth": (
                "bf827963b9e39c1bd3f89d8d93a88979e25483c52c19f33ad13d3ec834a47c42"
            ),
        },
    ),
}


def files(size):
    frozen, conditioned, version = SIZES[size]
    suffix = "_1_1" if version == "1.1" else ""
    return [f"depth_anything_v2_{frozen}.pth", f"prior_depth_anything_{conditioned}{suffix}.pth"]


def missing(size):
    return missing_files(FOLDER, files(size))


def start(settings, intrinsics, *, capture_delay=0.0):
    from ito.tofmesh.worker import TofMesh

    return TofMesh.for_model(
        "prior_da", settings.prior_da_size, settings, intrinsics, capture_delay
    )
