import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

HINGE_TYPES = ("revolute", "continuous")
# URDF: a joint without an <axis> turns about its frame's +X.
DEFAULT_AXIS = "1 0 0"
# A nib closer than this to the hinge axis has no direction to turn toward (m).
MIN_ARM_M = 1e-6


@dataclass(frozen=True)
class Nib:
    """A target frame's origin in the frame of the hinge it hangs from (m)."""

    hinge: str
    axis: np.ndarray
    position: np.ndarray


@dataclass(frozen=True)
class WristTurn:
    """How far the last joint turns (rad) and the nib gap left afterwards (m)."""

    angle: float
    miss_m: float


def _floats(text: str) -> np.ndarray:
    return np.array([float(value) for value in text.split()])


def _origin(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    xyz = _floats(origin.get("xyz", "0 0 0")) if origin is not None else np.zeros(3)
    rpy = _floats(origin.get("rpy", "0 0 0")) if origin is not None else np.zeros(3)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def nib(robot: ET.Element, frame: str) -> Nib:
    """Where frame sits relative to the first moving joint above it."""
    joints = {joint.find("child").get("link"): joint for joint in robot.iter("joint")}
    if frame not in {link.get("name") for link in robot.iter("link")}:
        raise ValueError(f"the URDF has no link {frame}")
    transform = np.eye(4)
    link = frame
    while (joint := joints.get(link)) is not None:
        kind = joint.get("type")
        if kind in HINGE_TYPES:
            axis_element = joint.find("axis")
            axis = _floats(
                axis_element.get("xyz", DEFAULT_AXIS)
                if axis_element is not None
                else DEFAULT_AXIS
            )
            return Nib(joint.get("name"), axis / np.linalg.norm(axis), transform[:3, 3])
        if kind != "fixed":
            raise ValueError(f"{frame} hangs from the {kind} joint {joint.get('name')}")
        transform = _origin(joint) @ transform
        link = joint.find("parent").get("link")
    raise ValueError(f"{frame} hangs from no hinge")


def wrist_turn(urdf_xml: bytes, old: str, new: str) -> WristTurn:
    """The turn of the hinge both frames hang from that puts new's nib where old's is."""
    robot = ET.fromstring(urdf_xml)
    old_nib, new_nib = nib(robot, old), nib(robot, new)
    if old_nib.hinge != new_nib.hinge:
        raise ValueError(
            f"{old} hangs from {old_nib.hinge} but {new} from {new_nib.hinge}; "
            "one hinge cannot bring them together"
        )
    axis = old_nib.axis
    target = old_nib.position - axis * (axis @ old_nib.position)
    start = new_nib.position - axis * (axis @ new_nib.position)
    if min(np.linalg.norm(target), np.linalg.norm(start)) < MIN_ARM_M:
        angle = 0.0
    else:
        angle = math.atan2(axis @ np.cross(start, target), start @ target)
    turned = Rotation.from_rotvec(axis * angle).apply(new_nib.position)
    return WristTurn(angle, float(np.linalg.norm(turned - old_nib.position)))
