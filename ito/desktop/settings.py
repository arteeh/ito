"""Pilot preferences, shared by desktop and headset panels."""

import json
import os
from pathlib import Path


def settings_path():
    root = os.environ.get("APPDATA") if os.name == "nt" else os.environ.get("XDG_CONFIG_HOME")
    return (Path(root) if root else Path.home() / ".config") / "ito" / "settings.json"


def load_budget(default):
    try:
        value = json.loads(settings_path().read_text())["max_splats"]
    except FileNotFoundError:
        return default
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Invalid Ito settings: max_splats must be an integer") from exc
    if type(value) is not int or not 1 <= value <= 4_194_304:
        raise ValueError("Invalid Ito settings: max_splats must be between 1 and 4,194,304")
    return value


def save_budget(value):
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"max_splats": value}) + "\n")
    temporary.replace(path)
