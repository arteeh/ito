import logging
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class AppSettings(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")
    diagnostic_logging: bool = False
    # ToF frames the mesh averages (tofmesh.volume): fewer follow the latest view.
    mesh_memory: int = Field(default=10, ge=2, le=40)


_current = None


def settings_path():
    root = os.environ.get("APPDATA") if os.name == "nt" else os.environ.get("XDG_CONFIG_HOME")
    return (Path(root) if root else Path.home() / ".config") / "ito" / "settings.json"


def load():
    global _current
    try:
        _current = AppSettings.model_validate_json(settings_path().read_text())
    except FileNotFoundError:
        _current = AppSettings()
    except (OSError, ValueError) as exc:
        logging.getLogger(__name__).warning("Cannot load app settings: %s; using defaults", exc)
        _current = AppSettings()
    return _current


def current():
    """The settings last loaded or saved: a change on the settings screen applies at once."""
    return _current or load()


def save(settings):
    global _current
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
    _current = settings
