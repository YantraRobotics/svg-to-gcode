import math
import xml.etree.ElementTree as ET

import pytest
from scipy.spatial.transform import Rotation

from svg_to_gcode import wrist


@pytest.mark.parametrize(
    ("old", "new", "degrees"),
    [
        ("target_frame_1", "target_frame_2", 120.0),
        ("target_frame_1", "target_frame_3", -120.0),
        ("target_frame_2", "target_frame_3", 120.0),
    ],
)
def test_the_wrist_turn_puts_the_new_nib_where_the_old_one_was(
    three_pen_urdf, old, new, degrees
):
    turn = wrist.wrist_turn(three_pen_urdf, old, new)

    assert abs(math.degrees(turn.angle)) == pytest.approx(abs(degrees), abs=0.01)
    assert turn.miss_m == pytest.approx(0.0, abs=1e-5)
    robot = ET.fromstring(three_pen_urdf)
    old_nib, new_nib = wrist.nib(robot, old), wrist.nib(robot, new)
    turned = Rotation.from_rotvec(new_nib.axis * turn.angle).apply(new_nib.position)
    assert turned == pytest.approx(old_nib.position, abs=1e-5)


def test_the_wrist_turn_opposes_the_way_back(three_pen_urdf):
    there = wrist.wrist_turn(three_pen_urdf, "target_frame_1", "target_frame_2")
    back = wrist.wrist_turn(three_pen_urdf, "target_frame_2", "target_frame_1")
    assert back.angle == pytest.approx(-there.angle)


def test_nibs_sit_along_the_hinge_axis_beyond_the_flange(three_pen_urdf):
    pen = wrist.nib(ET.fromstring(three_pen_urdf), "target_frame_1")
    assert pen.hinge == "joint6_ee_rotate"
    assert pen.axis @ pen.position == pytest.approx(0.036 + 0.277987)
    radial = pen.position - pen.axis * (pen.axis @ pen.position)
    assert math.hypot(*radial) == pytest.approx(0.137979)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        ("target_frame_9", "no link target_frame_9"),
        ("wrist_adapter_link", "hangs from no hinge"),
    ],
)
def test_frames_without_a_hinge_are_refused(three_pen_urdf, frame, message):
    with pytest.raises(ValueError, match=message):
        wrist.wrist_turn(three_pen_urdf, "target_frame_1", frame)


def test_a_frame_on_a_sliding_joint_is_refused(three_pen_urdf):
    with pytest.raises(ValueError, match="prismatic joint finger_joint"):
        wrist.wrist_turn(three_pen_urdf, "target_frame_1", "finger")
