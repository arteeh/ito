"""App-wide choices shared by desktop and headset menus."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AppSettings(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    precision: Literal["full", "half"] = "full"
    input_resolution: Literal[25, 50, 75, 100] = 100
    max_splats: int = Field(default=1_000_000, ge=1, le=(1 << 31) - 1)
    diagnostic_logging: bool = False


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


def load_budget(default=None):
    return load().max_splats


def save_budget(value):
    save(AppSettings.model_validate(load().model_dump() | {"max_splats": value}))
