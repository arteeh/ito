"""Which reconstruction this copy of Ito carries.

"full" (the default) is MASt3R-SLAM / posed RGB-D with the splat renderer. "tofmesh" is the
ToF mesh only (ito/tofmesh: ToF depth fused into a mesh placed by the robot's own pose stream,
camera frames as texture): no SLAM path at runtime and no models bundled. A Windows release
built with `release/windows.py --edition tofmesh` ships an EDITION file inside the ito package;
from source, ITO_EDITION=tofmesh selects it.
"""

import os
from importlib.resources import files

EDITIONS = ("full", "tofmesh")


def _name():
    name = os.environ.get("ITO_EDITION", "").strip().lower()
    if not name:
        try:
            name = files("ito").joinpath("EDITION").read_text(encoding="utf-8").strip().lower()
        except (FileNotFoundError, OSError):
            name = "full"
    if name not in EDITIONS:
        raise RuntimeError(f"Unknown Ito edition {name!r}; expected one of {', '.join(EDITIONS)}")
    return name


NAME = _name()
TOFMESH = NAME == "tofmesh"
