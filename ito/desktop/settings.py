import logging
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# MASt3R-SLAM's map size: keyframe slots, all allocated at start. 512 is upstream's; 128 (3.1 GB at
# 100 % input) leaves room for MASt3R and the scene on a 12-16 GB card.
KEYFRAMES = (64, 128, 192, 256, 384, 512)
# MASt3R-SLAM's settings per card VRAM, measured with the sim, the pilot's view drawing and the
# rest of Ceres's 12 GB held by another process (#83). Smaller cards first give up precision,
# then map size and input resolution.
PRESETS = {
    "10 GB+": dict(slam_precision="full", slam_input_resolution=100, slam_keyframes=128),
    "8 GB": dict(slam_precision="half", slam_input_resolution=75, slam_keyframes=128),
    "6 GB": dict(slam_precision="half", slam_input_resolution=50, slam_keyframes=64),
}


class AppSettings(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")
    diagnostic_logging: bool = False
    # ToF frames the mesh averages (tofmesh.volume): fewer follow the latest view.
    mesh_memory: int = Field(default=10, ge=2, le=240)
    # The 3D reconstruction method (ito.recon.METHODS), for the whole of a session, and the model
    # size of those that offer several.
    reconstruction: Literal["guessed", "pda", "da3", "prior_da", "mapanything", "mast3r_slam"] = (
        "guessed"
    )
    pda_size: Literal["small", "large"] = "large"
    da3_size: Literal["small", "base", "large", "giant", "metric"] = "metric"
    prior_da_size: Literal["small", "base", "large"] = "base"
    # A depth model's map in the ToF mesh: how much it counts, as a percentage of a ToF reading,
    # inside and outside the ToF's view (tofmesh.depth.DepthMap).
    depth_inside: int = Field(default=25, ge=0, le=100)
    depth_outside: int = Field(default=100, ge=0, le=100)
    # The depth model's precision (as MASt3R-SLAM's, below): Half runs its network in 16 bits.
    depth_precision: Literal["full", "half"] = "half"
    # MASt3R-SLAM (ito.recon.mast3r_slam): its network's precision, the share of its 512-pixel
    # input it is given, its map's keyframe slots, and the splats drawn.
    slam_precision: Literal["full", "half"] = "full"
    slam_input_resolution: Literal[25, 50, 75, 100] = 100
    slam_keyframes: Literal[KEYFRAMES] = 128
    max_splats: int = Field(default=1_000_000, ge=1, le=(1 << 31) - 1)


def settings_path():
    root = os.environ.get("APPDATA") if os.name == "nt" else os.environ.get("XDG_CONFIG_HOME")
    return (Path(root) if root else Path.home() / ".config") / "ito" / "settings.json"


def load():
    try:
        return AppSettings.model_validate_json(settings_path().read_text())
    except FileNotFoundError:
        return AppSettings()
    except (OSError, ValueError) as exc:
        logging.getLogger(__name__).warning("Cannot load app settings: %s; using defaults", exc)
        return AppSettings()


def save(settings):
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(dir=path.parent, prefix=".settings-")
    temporary = Path(filename)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(settings.model_dump_json() + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
