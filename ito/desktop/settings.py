import logging
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
