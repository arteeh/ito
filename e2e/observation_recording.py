"""Explicit offline migration of v3 recordings; the live v4 wire never accepts v3."""

import json

from ito.protocol import RobotDescription


def robot_description(data):
    if isinstance(data, str):
        data = json.loads(data)
    if data.get("version") == 3:
        data["version"] = 4
        if data.get("motion_noise") is not None:
            # This was v3's declared meaning, confined to archived recordings.
            data["motion_camera"] = data["cameras"][0]["name"]
    return RobotDescription.model_validate_json(json.dumps(data))


def motion_scale(row):
    if "uncertainty_scale" in row:
        return float(row["uncertainty_scale"])
    return 10.0 if row["state"] in {"held", "fallen"} else 1.0
