"""Robot comfort preferences and remembered pairing credentials."""

import hashlib
import json
import logging
import os
import tempfile
from typing import Literal

from pydantic import Field, model_validator

from ito.desktop.settings import AppSettings, settings_path
from ito.protocol import Credential, Model, Name

RECENT = 6


class Settings(AppSettings, Model):
    reconstruction: Literal["auto", "rgbd", "slam", "video"] = "auto"
    # SLAM tracking: the robot's own camera motion fused with vision, or vision alone (A/B).
    tracking: Literal["robot", "vision"] = "robot"
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
        values = json.loads(path.read_text())
        if not isinstance(values, dict):
            raise ValueError("Robot settings must be an object")
        for key in AppSettings.model_fields:
            values.pop(key, None)
        return Settings.model_validate(defaults.model_dump() | values)
    except FileNotFoundError:
        return defaults
    except (OSError, ValueError) as exc:
        logging.getLogger(__name__).warning("Cannot load %s: %s; using defaults", path, exc)
        return defaults


def save(address, name, settings):
    path = robot_path(address, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(settings.model_dump_json(exclude=set(AppSettings.model_fields)) + "\n")
    temporary.replace(path)


def recent_path():
    return settings_path().parent / "recent.json"


def _credential(value):
    try:
        return Credential(pilot=value["pilot"], secret=value["secret"])
    except (TypeError, KeyError, ValueError):
        return None


def recent():
    """Most recent first: [{"address": ..., "name": ..., "credential": Credential or None}]."""
    try:
        entries = json.loads(recent_path().read_text())
        return [
            {
                "address": str(e["address"]),
                "name": str(e["name"]),
                "credential": _credential(e.get("credential")),
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


def credential(address):
    """The secret the robot at address gave this pilot when it paired."""
    return next(
        (e["credential"] for e in recent() if canonical(e["address"]) == canonical(address)),
        None,
    )


def remember(address, name, credential):
    entries = [e for e in recent() if canonical(e["address"]) != canonical(address)]
    _write_recent([{"address": address, "name": name, "credential": credential}, *entries])


def forget_credential(address):
    """A refused credential is replaced by pairing again instead of being retried."""
    entries = recent()
    if any(canonical(e["address"]) == canonical(address) and e["credential"] for e in entries):
        try:
            _write_recent(
                [
                    e | {"credential": None} if canonical(e["address"]) == canonical(address) else e
                    for e in entries
                ]
            )
        except OSError as exc:
            logging.getLogger(__name__).warning("Cannot forget pairing credential: %s", exc)


def _write_recent(entries):
    path = recent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(dir=path.parent, prefix=".recent-")
    temporary = type(path)(filename)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(
                json.dumps(
                    [
                        e
                        | {
                            "credential": e["credential"].model_dump(include={"pilot", "secret"})
                            if e["credential"]
                            else None
                        }
                        for e in entries[:RECENT]
                    ]
                )
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def runtime_path():
    return settings_path().parent / "openxr-runtime.json"


def xr_runtime():
    """The OpenXR runtime manifest the pilot last flew with, or None."""
    try:
        return str(json.loads(runtime_path().read_text())["manifest"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def remember_xr_runtime(manifest):
    try:
        path = runtime_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"manifest": manifest}) + "\n")
    except OSError as exc:
        logging.getLogger(__name__).warning("Cannot save the OpenXR runtime: %s", exc)
