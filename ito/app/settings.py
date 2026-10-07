"""Atomic preferences keyed by driver address and robot name."""

import hashlib
import json
import logging
from typing import Literal

from pydantic import Field, model_validator

from ito.desktop.settings import settings_path
from ito.protocol import Model, Name

RECENT = 6


class Settings(Model):
    reconstruction: Literal["auto", "rgbd", "slam", "video"] = "auto"
    max_splats: int = Field(default=16384, ge=1, le=4_194_304)
    fov: float = Field(default=70, ge=30, le=120)
    sensitivity: float = Field(default=0.0025, ge=0.0001, le=0.02)
    invert_y: bool = False
    move_x: Name = "move_x"
    move_y: Name = "move_y"

    @model_validator(mode="after")
    def distinct_axes(self):
        if self.move_x == self.move_y:
            raise ValueError("Movement axes must have distinct names")
        return self


def robot_path(address, name):
    address = address.removeprefix("http://").rstrip("/")
    key = hashlib.sha256(f"{address}\n{name}".encode()).hexdigest()
    return settings_path().parent / "robots" / f"{key}.json"


def load(address, name, defaults):
    path = robot_path(address, name)
    try:
        return Settings.model_validate_json(path.read_text())
    except FileNotFoundError:
        return defaults
    except (OSError, ValueError) as exc:
        logging.getLogger(__name__).warning("Cannot load %s: %s; using defaults", path, exc)
        return defaults


def save(address, name, settings):
    path = robot_path(address, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(settings.model_dump_json() + "\n")
    temporary.replace(path)


def recent_path():
    return settings_path().parent / "recent.json"


def recent():
    """Most recent first: [{"address": ..., "name": ...}]."""
    try:
        entries = json.loads(recent_path().read_text())
        return [
            {"address": str(e["address"]), "name": str(e["name"])}
            for e in entries
            if isinstance(e, dict)
        ][:RECENT]
    except FileNotFoundError:
        return []
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logging.getLogger(__name__).warning("Cannot load recent robots: %s", exc)
        return []


def remember(address, name):
    entries = [e for e in recent() if e["address"] != address]
    path = recent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps([{"address": address, "name": name}, *entries][:RECENT]))
    temporary.replace(path)
