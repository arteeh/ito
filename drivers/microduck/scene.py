"""A furnished room around Pollen's unchanged robot, contact floor and pose keyframes."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).with_name("assets")
# Contact primitives move to a group that neither the viewer nor the head camera draws.
CONTACT_GROUP = "3"


def furnish(source: Path) -> ET.ElementTree:
    scene = ET.parse(source)
    room = ET.parse(ASSETS / "room.xml")
    asset = scene.find("asset")
    # The upstream sky and checker material describe the outdoor training plane.
    for element in list(asset):
        if element.tag == "texture" and element.get("type") == "skybox":
            asset.remove(element)
        elif element.get("name") == "groundplane":
            asset.remove(element)
    for section in room.getroot():
        target = scene.find(section.tag)
        if target is None:
            target = ET.SubElement(scene.getroot(), section.tag)
        for element in section:
            if element.tag == "texture":
                for attribute, filename in list(element.attrib.items()):
                    if attribute.startswith("file"):
                        element.set(attribute, str((ASSETS / filename).resolve()))
            target.append(element)
    # A non-uniform texture (the framed print) spans each face once instead of tiling by size.
    stretched = {m.get("name") for m in asset.iter("material") if m.get("texuniform") == "false"}
    for body in scene.getroot().iter("body"):
        if body.get("name", "").startswith("room_"):
            _draw_as_meshes(body, asset, stretched)
    floor = scene.find("worldbody/geom[@name='floor']")
    floor.set("material", "room_oak")
    floor.set("size", "2 1.8 0.05")
    floor.set("pos", "0.7 0 0")
    ET.SubElement(scene.getroot(), "statistic", center="0.7 0 0.5", extent="4")
    visual = scene.find("visual")
    visual.find("headlight").set("diffuse", "0.45 0.43 0.39")
    visual.find("headlight").set("ambient", "0.35 0.35 0.35")
    visual.find("global").set("azimuth", "45")
    visual.find("global").set("elevation", "-42")
    # MuJoCo scales clipping by model extent: retain millimetre-scale near vision in a room.
    ET.SubElement(visual, "map", znear="0.0003", zfar="10")
    light = scene.find("worldbody/light")
    light.set("pos", "0.3 -0.5 2.1")
    light.set("dir", "0.2 0.3 -1")
    light.set("directional", "false")
    light.set("diffuse", "0.65 0.62 0.56")
    return scene


def _draw_as_meshes(body: ET.Element, asset: ET.Element, stretched: set) -> None:
    """Draw the furniture as meshes; keep its exact primitives, hidden, for contact.

    WSL's GPU OpenGL (Mesa d3d12) draws MuJoCo's built-in primitives about ten times slower than
    the same shapes as meshes: a room of them held the viewer near one frame per second.
    """
    for geom in list(body.findall("geom")):
        kind = geom.get("type", "sphere")
        size = [float(v) for v in geom.get("size", "0").split()]
        drawn = ET.Element(
            "geom", {k: v for k, v in geom.attrib.items() if k not in {"size", "fromto"}}
        )
        if kind == "box":
            vertices, uv, faces = _box(*size, stretch=geom.get("material") in stretched)
        elif kind == "cylinder":
            radius, half = size[0], size[1]
            vertices, uv, faces = _lathe(
                [(0, -half), (radius, -half), (radius, half), (0, half)]
            )
        elif kind in {"ellipsoid", "sphere"}:
            radii = np.array(size * 3 if kind == "sphere" else size)
            vertices, uv, faces = _lathe(
                [(math.sin(t), -math.cos(t)) for t in np.linspace(0, math.pi, 13)]
            )
            vertices *= radii
        elif kind == "capsule":
            ends = np.array([float(v) for v in geom.get("fromto").split()]).reshape(2, 3)
            axis = ends[1] - ends[0]
            half = float(np.linalg.norm(axis)) / 2
            radius = size[0]
            arc = np.linspace(0, math.pi / 2, 5)
            profile = [(radius * math.sin(t), -half - radius * math.cos(t)) for t in arc]
            profile += [(radius * math.cos(t), half + radius * math.sin(t)) for t in arc]
            vertices, uv, faces = _lathe(profile)
            drawn.set("pos", _text(ends.mean(axis=0)))
            drawn.set("quat", _text(_turn_z_to(axis / (2 * half))))
        else:
            continue
        name = geom.get("name")
        ET.SubElement(
            asset,
            "mesh",
            name=f"{name}_mesh",
            vertex=_text(vertices),
            texcoord=_text(uv),
            face=" ".join(map(str, faces.ravel())),
        )
        drawn.attrib.update(
            name=f"{name}_drawn",
            type="mesh",
            mesh=f"{name}_mesh",
            contype="0",
            conaffinity="0",
            density="0",
        )
        body.insert(list(body).index(geom) + 1, drawn)
        if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
            body.remove(geom)  # Decoration: the primitive has no contact left to provide.
        else:
            geom.set("group", CONTACT_GROUP)


def _box(x, y, z, stretch=False):
    half = (x, y, z)
    scale = (lambda p, a: 0.5 - p / (2 * half[a])) if stretch else (lambda p, a: p)
    vertices, uv, faces = [], [], []
    for axis in range(3):
        u, v = (a for a in range(3) if a != axis)
        for sign in (-1, 1):
            start = len(vertices)
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = [0.0, 0.0, 0.0]
                p[axis], p[u], p[v] = sign * half[axis], su * half[u], sv * half[v]
                vertices.append(p)
                uv.append((scale(p[u], u), scale(p[v], v)))
            faces += [(start, start + 1, start + 2), (start, start + 2, start + 3)]
    return _outward(np.array(vertices), np.array(uv), np.array(faces))


def _lathe(profile, segments=24):
    """Revolve (radius, z) points about z, with UVs in metres like MuJoCo's uniform textures."""
    vertices, uv = [], []
    for radius, z in profile:
        for i in range(segments + 1):
            phi = 2 * math.pi * i / segments
            vertices.append((radius * math.cos(phi), radius * math.sin(phi), z))
            uv.append((max(radius, 0.05) * phi, z))
    faces = []
    for j in range(len(profile) - 1):
        for i in range(segments):
            a, b = j * (segments + 1) + i, (j + 1) * (segments + 1) + i
            faces += [(a, a + 1, b + 1), (a, b + 1, b)]
    return _outward(np.array(vertices, float), np.array(uv, float), np.array(faces))


def _outward(vertices, uv, faces):
    """Drop degenerate triangles and wind the rest counter-clockwise seen from outside."""
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    normal = np.cross(b - a, c - a)
    keep = np.linalg.norm(normal, axis=1) > 1e-12
    faces, normal, centre = faces[keep], normal[keep], ((a + b + c) / 3)[keep]
    inward = np.einsum("ij,ij->i", normal, centre) < 0
    faces[inward] = faces[inward][:, ::-1]
    return vertices, uv, faces


def _turn_z_to(direction):
    w = 1.0 + direction[2]
    if w < 1e-9:
        return np.array((0.0, 1.0, 0.0, 0.0))
    q = np.array((w, -direction[1], direction[0], 0.0))
    return q / np.linalg.norm(q)


def _text(values) -> str:
    return " ".join(f"{v:.6g}" for v in np.ravel(values))
