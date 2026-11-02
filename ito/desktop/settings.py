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
    mesh_memory: int = Field(default=10, ge=2, le=40)
    # Mono depth estimation where the ToF does not reach: the renderer's guess (tofmesh.render),
    # or Depth Anything 3 Metric depth maps (tofmesh.pda; stored model name is legacy),
    # counting these percentages of a ToF reading inside and outside the ToF's view.
    mono_depth: Literal["guessed", "pda"] = "guessed"
    pda_model: Literal["large", "small"] = "large"
    pda_inside: int = Field(default=25, ge=0, le=100)
    pda_outside: int = Field(default=100, ge=0, le=100)


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
