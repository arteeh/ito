"""A furnished room around Pollen's unchanged robot, contact floor and pose keyframes."""

import xml.etree.ElementTree as ET
from pathlib import Path

ASSETS = Path(__file__).with_name("assets")


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
