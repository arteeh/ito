"""Atomic preferences keyed by driver address and robot name."""

import hashlib
import json
import logging
import os
import tempfile
from typing import Literal

from pydantic import Field, model_validator

from ito.desktop.settings import settings_path
from ito.link.pairing import normalize
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
    """Most recent first: [{"address": ..., "name": ..., "code": pairing code or None}]."""
    try:
        entries = json.loads(recent_path().read_text())
        return [
            {
                "address": str(e["address"]),
                "name": str(e["name"]),
                "code": normalize(e["code"]) if isinstance(e.get("code"), str) else None,
            }
            for e in entries
            if isinstance(e, dict)
        ][:RECENT]
    except FileNotFoundError:
        return []
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logging.getLogger(__name__).warning("Cannot load recent robots: %s", exc)
        return []


def canonical(address):
    return address.removeprefix("http://").rstrip("/")


def code(address):
    """The pairing code that last let this pilot in at address."""
    return next(
        (e["code"] for e in recent() if canonical(e["address"]) == canonical(address)), None
    )


def remember(address, name, code):
    entries = [e for e in recent() if canonical(e["address"]) != canonical(address)]
    _write_recent([{"address": address, "name": name, "code": code}, *entries])


def forget_code(address):
    """A refused code is asked for again instead of being retried."""
    entries = recent()
    if any(canonical(e["address"]) == canonical(address) and e["code"] for e in entries):
        try:
            _write_recent(
                [
                    e | {"code": None} if canonical(e["address"]) == canonical(address) else e
                    for e in entries
                ]
            )
        except OSError as exc:
            logging.getLogger(__name__).warning("Cannot forget pairing code: %s", exc)


def _write_recent(entries):
    path = recent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(dir=path.parent, prefix=".recent-")
    temporary = type(path)(filename)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(entries[:RECENT]))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
