"""Prompt Depth Anything (PromptDA, CVPR 2025): a metric depth map from each camera frame and a
coarse depth prompt, the ToF's ranges as the camera sees them (model.prompt).

Its models are Apache-2.0 (Hugging Face transformers' port, huggingface.co/depth-anything).
"""

from ito.recon import missing_files

# Size -> the model's folder: its name under huggingface.co/depth-anything.
SIZES = {"large": "prompt-depth-anything-vitl-hf", "small": "prompt-depth-anything-vits-hf"}
WEIGHTS = {
    "prompt-depth-anything-vitl-hf": (
        "depth-anything/prompt-depth-anything-vitl-hf",
        "abfaa0958bbe7382633e90ffb6373cfee61239ed",
        "Apache-2.0",
        {
            "README.md": "6c57baeb946d68c2fa0cf1ff053dee695b257da194ba2cae17c80e312ff148b2",
            "config.json": "b63450bf142e9170988051293a29b89c4bc6950123e5c37df7ba12c8d4c825ec",
            "model.safetensors": "310a5a399691a1f56d1bc994a7463d626aef33d99d0949b954858434d0de3160",
        },
    ),
    "prompt-depth-anything-vits-hf": (
        "depth-anything/prompt-depth-anything-vits-hf",
        "6f3768a5383d95a904ec00d2a7ca3bd186f8797e",
        "Apache-2.0",
        {
            "README.md": "747b00fd961d4052ca056cb42af0bcc531f487834fe0513f74ffc447df78f9f4",
            "config.json": "2aef1ceef354f6149f1cf6c3a50abb00ad94ba3118fc88d65923b82643a34261",
            "model.safetensors": "b8dcec4ab96b47771addbfec2b7b410d801f472366b52fb5f20e2599e8d8c015",
        },
    ),
}


def missing(size):
    return missing_files(SIZES[size], ["config.json", "model.safetensors"])


def start(settings, intrinsics, *, capture_delay=0.0):
    from ito.tofmesh.worker import TofMesh

    return TofMesh.for_model("pda", settings.pda_size, settings, intrinsics, capture_delay)
