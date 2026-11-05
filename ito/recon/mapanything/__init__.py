"""MapAnything (Meta, 2025): metric 3D geometry from images and whatever else is known of them;
here each camera frame with its lens (intrinsics), for a metric depth map the ToF then checks
and anchors as any model's. One size (the 1.2B-parameter release).

Code Apache-2.0 (ito._vendor.mapanything, with UniCeption, BSD-3-Clause, and DINOv2,
Apache-2.0); the weights are facebook/map-anything, CC-BY-NC-4.0 (non-commercial; Meta also
publishes an Apache-2.0 variant, facebook/map-anything-apache).
"""

from ito.recon import missing_files

FOLDER = "map-anything"
WEIGHTS = {
    FOLDER: (
        "facebook/map-anything",
        "a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa",
        "CC-BY-NC-4.0",
        {
            "README.md": "496b2fef45f7c971629747de9ee533b7be32ff00e2afedab09e2f2d60a53e495",
            "config.json": "65701d09d99ed37a21d295f0d138978b3d584ab3bccdbcb4a2853da212b676c5",
            "model.safetensors": "981f060c64664dff3272b5f5a823d350abe71a2f144444db4cfc325f3ed5a3a0",
        },
    ),
}


def missing(size=None):
    return missing_files(FOLDER, ["config.json", "model.safetensors"])


def start(settings, intrinsics, *, capture_delay=0.0):
    from ito.tofmesh.worker import TofMesh

    return TofMesh.for_model("mapanything", None, settings, intrinsics, capture_delay)
