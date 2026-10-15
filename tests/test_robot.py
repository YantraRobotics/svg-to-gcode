import math
import time
from contextlib import closing
from itertools import pairwise

import pytest
import zmq

from motion import rotation_error
from svg_to_gcode import pens, robot, tilt

TOOL_TABLE = "; T1 yellow #FFF200\n; T4 black #231F20\n"
BLACK = pens.Pen("black", (0x23, 0x1F, 0x20))
RAPID_MM_MIN = 3000.0
FEED_MM_MIN = 1500.0
ORIGIN = (0.0, 0.0, 0.0)
RECEIVE_TIMEOUT_S = 2.0
PEN_OFFSET_MM = 40.0
# The tool's +X straight down: a quarter turn about base Y.
VERTICAL = tilt.rotate((0.0, 0.0, 0.0, 1.0), (0.0, math.pi / 2, 0.0))
TILT_RAD = math.radians(15.0)
ZIGZAG = (
    "G0 Z5\nT1 M6\nG0 X0 Y0\nG1 Z0 F600\nG1 X10 F1500\nG1 X0 Y1\nG1 X10 Y2\nG0 Z5\n"
)
TWO_PENS = (
    "G0 Z5\nT1 M6\nG0 X2 Y0\nG1 Z0 F600\nG1 X10 F1500\nG0 Z5\n"
    "T4 M6\nG0 X20 Y20\nG1 Z0 F600\nG1 X30 F1500\nG0 Z5\n"
)
SHEET_SIZE = (250.0, 200.0)
# Pen-tip corners as an arm with an inexact model reports them: not quite square.
MEASURED_CORNERS = [
    (-156.6, -324.8, -60.2),
    (-308.2, -141.4, -59.2),
    (-466.2, -303.6, -60.4),
    (-296.4, -480.0, -58.5),
]


def parse(body: str) -> list[robot.Step]:
    return robot.parse_gcode(TOOL_TABLE + body, ORIGIN, RAPID_MM_MIN)


@pytest.mark.parametrize("change", ["T4 M6", "M6 T4", "T4\nG0 X1\nM6"])
def test_m6_swaps_in_the_selected_pen(change):
    steps = parse(f"G0 Z5\n{change}\nG0 X10\n")
    (pen_change,) = [step for step in steps if isinstance(step, robot.PenChange)]
    assert (pen_change.tool, pen_change.pen) == (4, BLACK)
    assert pen_change.position[2] == 5.0


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("M6\nG0 X1\n", "line 3: M6 needs a tool"),
        ("T2 M6\nG0 X1\n", "line 3: T2 has no"),
        ("T1 T4 M6\n", "only one tool"),
        ("T1 M6 X5\n", "M6 takes no words"),
        ("TX M6\n", "expected a tool number"),
        ("; T1 cyan #00AEEF\nG0 X1\n", "line 3: T1 is listed twice"),
        ("; T7 grey #GG0000\nG0 X1\n", "line 3: colour must look like"),
    ],
)
def test_bad_tool_changes_are_rejected(body, message):
    with pytest.raises(ValueError, match=message):
        parse(body)


def test_pen_changes_split_chains_and_rise_to_the_change_height():
    steps = parse("G0 Z5\nT1 M6\nG0 X2\nG1 Z0 F600\nG1 X10 F1500\nG1 Y10\nG0 Z5\n")
    segments = robot.chains(steps)
    sizes = ["change" if isinstance(s, robot.PenChange) else len(s) for s in segments]
    assert sizes == [1, "change", 1, 1, 2, 1]

    placement = robot.Placement((100.0, 0.0, 20.0), 0.0, 0.0)
    low, high, _ = robot.summarize(segments, placement)
    assert (low[2], high[2]) == (20.0, 20.0 + robot.PEN_CHANGE_HEIGHT_MM)


def corners_of(
    placement: robot.Placement, size: tuple[float, float]
) -> list[robot.Vec3]:
    width, height = size
    rectangle = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    return [
        tuple(axis * robot.MM_PER_M for axis in placement.to_base((x, y, 0.0)))
        for x, y in rectangle
    ]


def test_corner_placement_lands_on_every_measured_corner():
    placement = robot.corner_placement(MEASURED_CORNERS, z_offset=0.0)
    width, height = placement.size
    rectangle = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    for (x, y), corner in zip(rectangle, MEASURED_CORNERS):
        base = placement.to_base((x, y, 0.0))
        assert tuple(axis * robot.MM_PER_M for axis in base) == pytest.approx(corner)


@pytest.mark.parametrize(
    "point", [(120.0, 100.0, 0.0), (-80.0, 300.0, 45.0)], ids=["on", "beside"]
)
def test_corner_placement_finds_base_points_on_the_paper(point):
    placement = robot.corner_placement(MEASURED_CORNERS, z_offset=-0.5)
    assert placement.to_paper(placement.to_base(point)) == pytest.approx(point)


def test_square_corners_place_like_origin_and_yaw():
    rigid = robot.Placement((-125.0, -460.0, 30.6), math.radians(20.0), -0.5)
    level = robot.Placement(rigid.origin, rigid.yaw, 0.0)
    placement = robot.corner_placement(corners_of(level, SHEET_SIZE), z_offset=-0.5)
    point = (40.0, 170.0, 3.0)
    assert placement.to_base(point) == pytest.approx(rigid.to_base(point))
    assert placement.yaw == pytest.approx(rigid.yaw)


@pytest.mark.parametrize(
    "order", [(0, 1, 3, 2), (0, 3, 2, 1)], ids=["crossed", "clockwise"]
)
def test_corners_out_of_order_are_refused(order):
    corners = [MEASURED_CORNERS[index] for index in order]
    with pytest.raises(ValueError, match="list them bottom-left, bottom-right"):
        robot.corner_placement(corners, z_offset=0.0)


def test_a_mistyped_corner_height_is_refused():
    corners = list(MEASURED_CORNERS)
    x, y, z = corners[2]
    corners[2] = (x, y, z - 20.0)
    with pytest.raises(ValueError, match=r"22\.9 mm out of one plane"):
        robot.corner_placement(corners, z_offset=0.0)


def path_of(moves: list[robot.Move]) -> list[tuple[robot.Vec3, robot.Vec3]]:
    return [(move.start, move.end) for move in moves]


def test_one_pen_draws_only_its_strokes_from_above():
    start = (50.0, 50.0, 40.0)
    steps = robot.parse_gcode(TOOL_TABLE + TWO_PENS, start, RAPID_MM_MIN)
    moves = robot.one_pen(steps, 4, start, RAPID_MM_MIN)
    assert path_of(moves) == [
        ((50.0, 50.0, 40.0), (20.0, 20.0, 40.0)),
        ((20.0, 20.0, 40.0), (20.0, 20.0, 5.0)),
        ((20.0, 20.0, 5.0), (20.0, 20.0, 0.0)),
        ((20.0, 20.0, 0.0), (30.0, 20.0, 0.0)),
        ((30.0, 20.0, 0.0), (30.0, 20.0, 5.0)),
        ((30.0, 20.0, 5.0), (30.0, 20.0, robot.PEN_CHANGE_HEIGHT_MM)),
    ]
    assert [move.rapid for move in moves] == [True, True, False, False, True, True]


def test_one_pen_rises_before_crossing_from_a_low_start():
    start = (0.0, 0.0, 0.0)
    steps = robot.parse_gcode(TOOL_TABLE + TWO_PENS, start, RAPID_MM_MIN)
    approach = robot.one_pen(steps, 1, start, RAPID_MM_MIN)[:2]
    assert path_of(approach) == [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 5.0)),
        ((0.0, 0.0, 5.0), (2.0, 0.0, 5.0)),
    ]


def test_one_pen_refuses_a_tool_with_nothing_to_draw():
    steps = parse(TWO_PENS)
    with pytest.raises(ValueError, match="T2 draws nothing"):
        robot.one_pen(steps, 2, ORIGIN, RAPID_MM_MIN)


def test_gcode_beyond_the_corners_is_refused():
    robot.check_on_paper(TOOL_TABLE + TWO_PENS, (30.0, 20.0))
    with pytest.raises(ValueError, match=r"X30 Y20, off the 25\.0 x 100\.0 mm paper"):
        robot.check_on_paper(TOOL_TABLE + TWO_PENS, (25.0, 100.0))


class FaultedShell:
    def __init__(self):
        self.events: list[str] = []

    def telemetry(self, timeout_s: float = 1.0) -> dict:
        return {"controller_state": robot.CONTROLLER_ERROR, "fault": "estop latched"}

    def send_event(self, name: str) -> bool:
        self.events.append(name)
        return True


def test_draw_refuses_a_controller_in_error_instead_of_clearing_it():
    shell = FaultedShell()
    placement = robot.Placement(ORIGIN, 0.0, 0.0)
    with pytest.raises(robot.DrawError, match="estop latched"):
        robot.draw(
            shell, TOOL_TABLE + ZIGZAG, placement, PEN_OFFSET_MM, RAPID_MM_MIN, 0.0, 0.0
        )
    assert shell.events == []


def receive(listener: zmq.Socket, announcer: robot.PenAnnouncer) -> dict:
    """The first message, announcing until the subscription has reached the publisher."""
    deadline = time.monotonic() + RECEIVE_TIMEOUT_S
    while not listener.poll(10):
        assert time.monotonic() < deadline, "no pen status arrived"
        announcer.announce(BLACK)
    message = listener.recv_json()
    while listener.poll(10):
        listener.recv()
    return message


def test_pen_announcer_publishes_changes_and_repeats_them():
    endpoint = "inproc://pen-status-test"
    with closing(robot.PenAnnouncer(endpoint)) as announcer:
        listener = zmq.Context.instance().socket(zmq.SUB)
        listener.setsockopt(zmq.SUBSCRIBE, b"")
        listener.connect(endpoint)
        try:
            announcer.tick()
            assert receive(listener, announcer) == {
                "name": "black",
                "rgb": [35, 31, 32],
            }
            announcer.tick()
            assert not listener.poll(50)
            announcer.sent_at -= robot.PEN_STATUS_PERIOD_S
            announcer.tick()
            assert listener.poll(int(RECEIVE_TIMEOUT_S * 1000))
        finally:
            listener.close(linger=0)


class FakeGripperShell:
    """Telemetry from a gripper whose fingers stop on a pen when one is held."""

    def __init__(self, present: bool, pen_inserted: bool):
        self.pen_inserted = pen_inserted
        self.commands: list[float] = []
        self.state = {
            "gripper_present": int(present),
            "gripper_activated": int(present),
            "gripper_moving": 0,
            "gripper_object": 1,
            "gripper_position": 0.4,
            "gripper_target": robot.GRIPPER_CLOSED,
        }

    def telemetry(self, timeout_s: float = 1.0) -> dict:
        return dict(self.state)

    def send_gripper_command(
        self, position: float, speed: float = 1.0, force: float = 0.25
    ) -> bool:
        self.commands.append(position)
        gripped = position == robot.GRIPPER_CLOSED and self.pen_inserted
        self.state.update(
            gripper_target=position,
            gripper_object=int(gripped),
            gripper_position=0.4 if gripped else position,
        )
        return True


def test_swap_without_a_gripper_only_waits():
    shell = FakeGripperShell(present=False, pen_inserted=True)
    robot._swap(shell, BLACK, swap_wait=0.0)
    assert shell.commands == []


def test_swap_opens_the_gripper_then_grips_the_new_pen():
    shell = FakeGripperShell(present=True, pen_inserted=True)
    robot._swap(shell, BLACK, swap_wait=0.0)
    assert shell.commands == [robot.GRIPPER_OPEN, robot.GRIPPER_CLOSED]


def test_swap_fails_when_the_gripper_closes_on_nothing():
    shell = FakeGripperShell(present=True, pen_inserted=False)
    with pytest.raises(robot.DrawError, match="closed on nothing"):
        robot._swap(shell, BLACK, swap_wait=0.0)


@pytest.mark.parametrize("heading_deg", [0, 60, 135, 180, 270])
@pytest.mark.parametrize("tilt_deg", [0, 15, 30])
def test_tilted_pen_leans_forward_and_trails_its_tip(heading_deg, tilt_deg):
    heading, lean = math.radians(heading_deg), math.radians(tilt_deg)
    axis = tilt.tool_axis(tilt.tilted(VERTICAL, heading, lean))
    expected = (
        -math.sin(lean) * math.cos(heading),
        -math.sin(lean) * math.sin(heading),
        -math.cos(lean),
    )
    assert axis == pytest.approx(expected, abs=1e-12)


def test_vertical_from_takes_the_smallest_turn():
    start = tilt.rotate(tilt.rotate(VERTICAL, (0.0, 0.0, 0.7)), (0.12, -0.05, 0.0))
    upright = tilt.vertical_from(start)

    assert tilt.tool_axis(upright) == pytest.approx(tilt.DOWN, abs=1e-12)
    turn = rotation_error(start, upright)
    assert math.hypot(*turn) == pytest.approx(math.hypot(0.12, 0.05))
    assert turn[2] == pytest.approx(0.0, abs=1e-12)


def test_vertical_from_refuses_a_steep_start():
    with pytest.raises(ValueError, match="23 degrees off vertical"):
        tilt.vertical_from(tilt.rotate(VERTICAL, (0.4, 0.0, 0.0)))


def test_tcp_velocity_keeps_the_tip_on_course_while_turning():
    quat = tilt.tilted(VERTICAL, 0.3, TILT_RAD)
    tcp = (0.4, -0.1, 0.3)
    tip = robot.pen_tip((*tcp, *quat), PEN_OFFSET_MM)
    tip_velocity = (0.025, -0.01, 0.002)
    spin = (0.2, -0.4, 0.9)
    tip_offset = tuple(t - p for t, p in zip(tip, tcp))
    linear = tilt.tcp_velocity(tip_velocity, spin, tip_offset)

    dt = 1e-6
    moved_tcp = tuple(p + v * dt for p, v in zip(tcp, linear))
    turned = tilt.rotate(quat, tuple(w * dt for w in spin))
    moved_tip = robot.pen_tip((*moved_tcp, *turned), PEN_OFFSET_MM)
    velocity = tuple((b - a) / dt for a, b in zip(tip, moved_tip))
    assert velocity == pytest.approx(tip_velocity, abs=1e-6)


def test_lookahead_heading_turns_before_the_corner():
    points = [(0.0, 0.0), (0.01, 0.0), (0.01, 0.01)]
    distances = [0.0, 0.01, 0.02]

    def heading_deg(travelled: float) -> float:
        heading = tilt.lookahead_heading(points, distances, travelled, 0.002)
        return math.degrees(heading)

    assert heading_deg(0.0) == pytest.approx(0.0)
    assert heading_deg(0.009) == pytest.approx(45.0)
    assert heading_deg(0.0195) == pytest.approx(90.0)
    assert heading_deg(0.02) == pytest.approx(90.0)


def drawing_through(*points: tuple[float, float]) -> list[robot.Move]:
    return [
        robot.Move((*a, 0.0), (*b, 0.0), FEED_MM_MIN, rapid=False)
        for a, b in pairwise(points)
    ]


def test_drawing_splits_only_at_turns_sharper_than_the_limit():
    chain = drawing_through((0, 0), (10, 0), (10, 10), (0, 0), (-5, 1))
    assert [len(piece) for piece in robot.split_at_sharp_turns(chain)] == [2, 2]


def test_a_rounded_corner_splits_once_at_its_sharpest_vertex():
    # Three 50 degree turns 0.2 mm apart, as tracing leaves a sharp point.
    points = [(0.0, 0.0), (10.0, 0.0)]
    heading = 0.0
    for _ in range(3):
        heading += math.radians(50.0)
        x, y = points[-1]
        points.append((x + 0.2 * math.cos(heading), y + 0.2 * math.sin(heading)))
    x, y = points[-1]
    points.append((x + 10.0 * math.cos(heading), y + 10.0 * math.sin(heading)))

    pieces = robot.split_at_sharp_turns(drawing_through(*points))
    assert [len(piece) for piece in pieces] == [2, 3]


def test_tilting_hops_at_each_sharp_turn():
    steps = parse(ZIGZAG)
    upright, no_hops = robot.plan_motion(steps, 0.0)
    segments, hops = robot.plan_motion(steps, 15.0)

    assert (no_hops, hops) == (0, 2)
    assert len(segments) == len(upright) + 3 * hops
    first = next(
        index
        for index, segment in enumerate(segments)
        if isinstance(segment, list) and segment[0].is_drawing
    )
    (up,), (down,) = segments[first + 1], segments[first + 2]
    assert (up.start, up.end) == ((10.0, 0.0, 0.0), (10.0, 0.0, robot.HOP_MM))
    assert (down.start, down.end) == ((10.0, 0.0, robot.HOP_MM), (10.0, 0.0, 0.0))


def test_the_pen_leans_toward_the_next_stroke_and_stands_for_pen_changes():
    steps = parse("G0 Z5\nT1 M6\nG0 X2 Y0\nG1 Z0 F600\nG1 X2 Y10 F1500\nG0 Z5\n")
    segments, _ = robot.plan_motion(steps, 15.0)
    # Paper +Y points along base -X when the paper is turned a quarter turn.
    placement = robot.Placement((300.0, 0.0, 0.0), math.radians(90.0), 0.0)
    goals = robot.orientations(segments, placement, VERTICAL, TILT_RAD)

    leaning = tilt.tool_axis(tilt.tilted(VERTICAL, math.pi, TILT_RAD))
    upright = tilt.tool_axis(VERTICAL)
    axes = [tilt.tool_axis(goal(0.0)) for goal in goals]
    expected = [upright, upright, leaning, leaning, leaning, upright]
    assert axes == [pytest.approx(axis, abs=1e-12) for axis in expected]
    assert leaning[0] > 0


class FakeClock:
    """Stands in for the time module: sleeping moves the clock on."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeArm:
    """A tool that moves exactly as twisted until the deadman stops it.

    One telemetry read stalls the clock, as a starved process would.
    """

    def __init__(self, clock: FakeClock, pose, stall_on_read: int, stall_s: float):
        self.clock = clock
        self.position = pose[:3]
        self.quat = pose[3:]
        self.twist = robot.ZERO_TWIST
        self.twisted_at = self.updated = clock.now
        self.reads = 0
        self.stall_on_read = stall_on_read
        self.stall_s = stall_s

    def _advance(self) -> None:
        moving_until = min(self.clock.now, self.twisted_at + robot.DEADMAN_S)
        dt = max(0.0, moving_until - self.updated)
        self.position = tuple(p + v * dt for p, v in zip(self.position, self.twist))
        self.quat = tilt.rotate(self.quat, tuple(w * dt for w in self.twist[3:]))
        self.updated = self.clock.now

    def telemetry(self, timeout_s: float = 1.0) -> dict:
        self.reads += 1
        if self.reads == self.stall_on_read:
            self.clock.now += self.stall_s
        self._advance()
        return {
            "tcp_pose": (*self.position, *self.quat),
            "controller_state": robot.RUNNING,
            "motion_active": 1,
            "fault": "",
        }

    def send_twist(self, twist, quiet: bool = False) -> bool:
        self._advance()
        self.twist = twist
        self.twisted_at = self.clock.now
        return True


class HoldableArm(FakeArm):
    """A FakeArm whose controller accepts HOLD and reports it reached."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events: list[str] = []

    def send_event(self, name: str) -> bool:
        self.events.append(name)
        return True

    def wait_state(
        self, state_name: str, timeout_s: float, require_motion: bool = False
    ) -> bool:
        return state_name == "power_on"


class PaperArm(FakeArm):
    """A FakeArm whose tip a surface stops at floor_z (base frame, m)."""

    def __init__(self, *args, floor_z: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.floor_z = floor_z

    def _advance(self) -> None:
        super()._advance()
        x, y, z = self.position
        self.position = (x, y, max(z, self.floor_z))


class SilentAnnouncer:
    def __init__(self):
        self.pens: list[pens.Pen] = []

    def announce(self, pen: pens.Pen) -> None:
        self.pens.append(pen)

    def tick(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("stall_s", "goal"),
    [
        (0.0, VERTICAL),
        (0.8, VERTICAL),
        (0.0, tilt.tilted(VERTICAL, 0.0, TILT_RAD)),
    ],
    ids=["steady", "stalled", "tilting"],
)
def test_follow_keeps_the_tip_on_the_stroke(monkeypatch, stall_s, goal):
    clock = FakeClock()
    monkeypatch.setattr(robot, "time", clock)
    placement = robot.Placement((300.0, 0.0, 0.0), 0.0, 0.0)
    tcp = (0.3, 0.0, PEN_OFFSET_MM / robot.MM_PER_M)
    arm = FakeArm(clock, (*tcp, *VERTICAL), stall_on_read=30, stall_s=stall_s)
    plotter = robot.Plotter(arm, placement, PEN_OFFSET_MM, SilentAnnouncer())
    stroke = [robot.Move(ORIGIN, (40.0, 0.0, 0.0), FEED_MM_MIN, rapid=False)]

    worst = plotter.follow(stroke, lambda _travelled: goal)

    assert worst < 1.0
    assert plotter.tip() == pytest.approx((40.0, 0.0, 0.0), abs=0.5)
    assert math.hypot(*rotation_error(arm.quat, goal)) < robot.SETTLE_TOLERANCE_RAD


def test_a_stroke_settles_on_paper_that_sits_higher_than_mapped(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(robot, "time", clock)
    placement = robot.Placement((300.0, 0.0, 0.0), 0.0, 0.0)
    arm = PaperArm(
        clock, (0.3, 0.0, 0.0, *VERTICAL), stall_on_read=0, stall_s=0.0, floor_z=0.001
    )
    plotter = robot.Plotter(arm, placement, 0.0, SilentAnnouncer())
    stroke = [robot.Move(ORIGIN, (40.0, 0.0, 0.0), FEED_MM_MIN, rapid=False)]

    plotter.follow(stroke, lambda _travelled: VERTICAL)

    assert plotter.tip() == pytest.approx((40.0, 0.0, 1.0), abs=0.5)


def test_a_tip_held_off_its_target_above_the_paper_warns_and_moves_on(
    monkeypatch, caplog
):
    clock = FakeClock()
    monkeypatch.setattr(robot, "time", clock)
    placement = robot.Placement((300.0, 0.0, 0.0), 0.0, 0.0)
    arm = PaperArm(
        clock, (0.3, 0.0, 0.01, *VERTICAL), stall_on_read=0, stall_s=0.0, floor_z=0.007
    )
    plotter = robot.Plotter(arm, placement, 0.0, SilentAnnouncer())
    lower = [robot.Move((0.0, 0.0, 10.0), (0.0, 0.0, 5.0), RAPID_MM_MIN, rapid=True)]

    plotter.follow(lower, lambda _travelled: VERTICAL)

    assert "0.0 mm off across the paper, +2.0 mm in height" in caplog.text
    assert clock.now < 2 * robot.SETTLE_TIMEOUT_S


def test_one_pen_run_with_the_tcp_at_the_tip_ends_raised_and_held(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(robot, "time", clock)
    monkeypatch.setattr(robot, "ensure_running_mode", lambda shell, mode: True)
    announcer = SilentAnnouncer()
    monkeypatch.setattr(robot, "PenAnnouncer", lambda: announcer)
    placement = robot.corner_placement(MEASURED_CORNERS, z_offset=0.0)
    start = placement.to_base((0.0, 0.0, 30.0))
    arm = HoldableArm(clock, (*start, *VERTICAL), stall_on_read=0, stall_s=0.0)

    worst = robot.draw(
        arm, TOOL_TABLE + TWO_PENS, placement, 0.0, RAPID_MM_MIN, 0.0, None, 4
    )

    assert worst < 1.0
    assert (arm.events, announcer.pens) == (["hold"], [BLACK])
    tip = placement.to_paper(arm.position)
    assert tip == pytest.approx((30.0, 20.0, robot.PEN_CHANGE_HEIGHT_MM), abs=0.5)


PEN_FRAMES = {"yellow": "target_frame_1", "black": "target_frame_2"}


def test_pen_frames_parse_by_pen_name():
    text = " yellow=target_frame_1 , black=target_frame_2,"
    assert robot.parse_pen_frames(text) == PEN_FRAMES
    assert robot.parse_pen_frames("") == {}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("yellow", "NAME=FRAME"),
        ("yellow=tool0", "must start with 'target_frame'"),
        ("black=target_frame,black=target_frame_1", "black pen is given a frame twice"),
    ],
)
def test_bad_pen_frames_are_rejected(text, message):
    with pytest.raises(ValueError, match=message):
        robot.parse_pen_frames(text)


def test_pen_runs_start_above_each_pens_first_stroke():
    runs = robot.pen_runs(parse(TWO_PENS), PEN_FRAMES)
    assert [(run.tool, run.frame) for run in runs] == [
        (1, "target_frame_1"),
        (4, "target_frame_2"),
    ]
    assert [run.moves[0].start for run in runs] == [(2.0, 0.0, 5.0), (20.0, 20.0, 5.0)]
    (only,) = robot.pen_runs(parse(TWO_PENS), PEN_FRAMES, tool=4)
    assert only.pen == BLACK


def test_pen_runs_need_a_frame_for_every_pen_drawn():
    with pytest.raises(ValueError, match=r"black pen \(T4\) has no target frame"):
        robot.pen_runs(parse(TWO_PENS), {"yellow": "target_frame_1"})
    robot.pen_runs(parse(TWO_PENS), {"black": "target_frame_2"}, tool=4)


class FramedArm(FakeArm):
    """A FakeArm with mounted pens: switching frames moves the reported tip."""

    def __init__(self, *args, offsets: dict[str, robot.Vec3], **kwargs):
        super().__init__(*args, **kwargs)
        self.offsets = offsets
        self.current = next(iter(offsets))
        self.switches: list[str] = []
        self.switched_from: list[robot.Vec3] = []

    def telemetry(self, timeout_s: float = 1.0) -> dict:
        telemetry = super().telemetry(timeout_s)
        moving = self.twist != robot.ZERO_TWIST
        moving = moving and self.clock.now < self.twisted_at + robot.DEADMAN_S
        return {**telemetry, "qdot_measured": (0.1 if moving else 0.0,) * 6}

    def get_target_frame(self) -> dict:
        return {"current": self.current, "eligible": sorted(self.offsets)}

    def set_target_frame(self, name: str) -> bool:
        if self.telemetry()["qdot_measured"][0]:
            return False
        self.switched_from.append(self.position)
        old, new = self.offsets[self.current], self.offsets[name]
        self.position = tuple(p - o + n for p, o, n in zip(self.position, old, new))
        self.current = name
        self.switches.append(name)
        return True


def test_mounted_pens_switch_frames_and_rise_after_each_pen(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(robot, "time", clock)
    monkeypatch.setattr(robot, "ensure_running_mode", lambda shell, mode: True)
    announcer = SilentAnnouncer()
    monkeypatch.setattr(robot, "PenAnnouncer", lambda: announcer)
    placement = robot.Placement((300.0, 0.0, 0.0), 0.0, 0.0)
    offsets = {
        "target_frame": (0.0, 0.0, 0.0),
        "target_frame_1": (0.03, 0.0, 0.0),
        "target_frame_2": (0.0, 0.03, -0.01),
    }
    start = placement.to_base((0.0, 0.0, 30.0))
    arm = FramedArm(
        clock, (*start, *VERTICAL), stall_on_read=0, stall_s=0.0, offsets=offsets
    )

    worst = robot.draw(
        arm,
        TOOL_TABLE + TWO_PENS,
        placement,
        0.0,
        RAPID_MM_MIN,
        0.0,
        None,
        pen_frames=PEN_FRAMES,
    )

    assert worst < 1.0
    assert arm.switches == ["target_frame_1", "target_frame_2"]
    assert announcer.pens == [pens.Pen("yellow", (0xFF, 0xF2, 0x00)), BLACK]
    # Settling at the last stroke and again at the top each leave up to the tolerance.
    within = 2 * robot.SETTLE_TOLERANCE_MM
    yellow_done = placement.to_paper(arm.switched_from[1])
    assert yellow_done == pytest.approx(
        (10.0, 0.0, 5.0 + robot.FRAME_LIFT_MM), abs=within
    )
    end = placement.to_paper(arm.position)
    assert end == pytest.approx((30.0, 20.0, 5.0 + robot.FRAME_LIFT_MM), abs=within)


def test_mounted_pens_need_frames_the_controller_serves(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(robot, "time", clock)
    arm = FramedArm(
        clock,
        (0.3, 0.0, 0.03, *VERTICAL),
        stall_on_read=0,
        stall_s=0.0,
        offsets={"target_frame": (0.0, 0.0, 0.0), "target_frame_1": (0.0, 0.0, 0.0)},
    )
    arm.telemetry = lambda timeout_s=1.0: {"controller_state": robot.RUNNING}
    with pytest.raises(robot.DrawError, match="no target frame target_frame_2"):
        robot.draw(
            arm,
            TOOL_TABLE + TWO_PENS,
            robot.Placement(ORIGIN, 0.0, 0.0),
            0.0,
            RAPID_MM_MIN,
            0.0,
            None,
            pen_frames=PEN_FRAMES,
        )


def test_jog_moves_the_tip_straight_down(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(robot, "time", clock)
    monkeypatch.setattr(robot, "ensure_running_mode", lambda shell, mode: True)
    monkeypatch.setattr(robot, "PenAnnouncer", SilentAnnouncer)
    arm = FakeArm(clock, (0.3, -0.2, 0.05, *VERTICAL), stall_on_read=0, stall_s=0.0)

    stopped = robot.jog(arm, -5.0)

    assert stopped == pytest.approx(
        (300.0, -200.0, 45.0), abs=robot.SETTLE_TOLERANCE_MM
    )
    assert arm.position[:2] == pytest.approx((0.3, -0.2), abs=1e-3)


@pytest.mark.parametrize("height_mm", [0.0, -robot.MAX_JOG_MM - 1])
def test_jog_refuses_no_step_or_a_long_one(height_mm):
    with pytest.raises(ValueError, match="a jog must be nonzero"):
        robot.jog(FaultedShell(), height_mm)


def test_swap_needs_a_terminal_or_a_wait(monkeypatch):
    def no_terminal(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", no_terminal)
    with pytest.raises(robot.DrawError, match="--swap-wait"):
        robot._swap(FakeGripperShell(present=False, pen_inserted=True), BLACK, None)
