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


class SilentAnnouncer:
    def tick(self) -> None:
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


def test_swap_needs_a_terminal_or_a_wait(monkeypatch):
    def no_terminal(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", no_terminal)
    with pytest.raises(robot.DrawError, match="--swap-wait"):
        robot._swap(FakeGripperShell(present=False, pen_inserted=True), BLACK, None)
