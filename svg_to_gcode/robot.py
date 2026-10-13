import logging
import math
import re
import tempfile
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from itertools import accumulate, pairwise
from pathlib import Path

import zmq

from motion import (
    LOOP_RATE_HZ,
    STALL_TELEMETRY_S,
    clamp_norm,
    ensure_running_mode,
    pose_servo_step,
    rotation_error,
)
from robot_interface import CONTROLLER_STATES, Shell, encode_get_model, parse_model
from svg_to_gcode import tilt
from svg_to_gcode.pens import Pen, parse_hex

log = logging.getLogger(__name__)

type Vec3 = tuple[float, float, float]
type Quat = tuple[float, float, float, float]
# The tool orientation to aim for, given how far along a chain the reference is (m).
type Orientation = Callable[[float], Quat]

TIP_LINK = "target_frame"
MM_PER_M = 1000.0
SECONDS_PER_MINUTE = 60.0
MAX_FEED_MM_MIN = 6000.0
MIN_Z_MM = -2.0
KP_LIN = 5.0
KP_ANG = 5.0
MAX_CORRECTION_M_S = 0.05
# The real arm stalls at the few mm/s the correction asks for near a target, so
# a settle that runs out of time moves on; MAX_TRACKING_ERROR_MM still aborts.
SETTLE_TOLERANCE_MM = 1.0
SETTLE_TOLERANCE_RAD = 0.02
SETTLE_CYCLES = 5
SETTLE_TIMEOUT_S = 2.0
# PROTOCOL.md: a twist decays to zero 200 ms after the last one arrives.
DEADMAN_S = 0.2
# The real arm falls several mm behind on tight zigzags at drawing feeds and
# coasts for DEADMAN_S whenever the network stalls.
MAX_TRACKING_ERROR_MM = 25.0
ABORT_LIFT_MM = 10.0
ABORT_LIFT_FEED_MM_MIN = 600.0
PEN_CHANGE_HEIGHT_MM = 80.0
HOLD_TIMEOUT_S = 10.0
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0
# Half speed and a light grip: the fingers close beside the operator's hand.
GRIPPER_SPEED = 0.5
GRIPPER_FORCE = 0.25
GRIPPER_TOLERANCE = 0.02
GRIPPER_TIMEOUT_S = 5.0
TELEMETRY_POLL_S = 0.1
MAX_TILT_DEG = 30.0
SHARP_TURN_DEG = 90.0
HOP_MM = 1.5
HOP_FEED_MM_MIN = 600.0
TILT_LOOKAHEAD_MM = 2.0
# motion.MAX_ANG_VEL (0.5 rad/s) would make every pivot crawl.
MAX_TILT_RATE_RAD_S = 1.0
PEN_STATUS_ENDPOINT = f"ipc://{Path(tempfile.gettempdir()) / 'svg-to-gcode-pen'}"
PEN_STATUS_PERIOD_S = 1.0
TOOL_TABLE_ENTRY = re.compile(r"T(\d+)\s+(\S+)\s+(#\w+)", re.IGNORECASE)
STATE_CODES = {name: code for code, name in CONTROLLER_STATES.items()}
RUNNING = STATE_CODES["running"]
CONTROLLER_ERROR = STATE_CODES["error"]
ZERO_TWIST = (0.0,) * 6
CORNER_NAMES = ("bottom-left", "bottom-right", "top-right", "top-left")
RIGHT_ANGLE_DEG = 90.0
# The arm's reported frame skews measured corners by a few degrees; more than
# this means a corner is mistyped or listed out of order.
MAX_CORNER_SKEW_DEG = 15.0
# How far the corner heights may be from lying in one plane before a height
# is taken to be mistyped.
MAX_CORNER_TWIST_MM = 5.0
MIN_PAPER_MM = 10.0
INVERSE_ITERATIONS = 20
INVERSE_TOLERANCE_MM = 1e-6


class DrawError(Exception):
    pass


@dataclass(frozen=True)
class Move:
    """A straight move in paper coordinates (mm), at feed mm/min."""

    start: Vec3
    end: Vec3
    feed: float
    rapid: bool

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)

    @property
    def is_drawing(self) -> bool:
        return not self.rapid and self.start[2] == self.end[2]


@dataclass(frozen=True)
class PenChange:
    """An M6: swap in the tool's pen, programmed at position (paper mm)."""

    tool: int
    pen: Pen
    position: Vec3


type Step = Move | PenChange
# A run of moves followed in one go, or a pen change.
type Segment = list[Move] | PenChange


@dataclass(frozen=True)
class Placement:
    """Paper pose in the robot base frame.

    origin is the paper's bottom-left corner on its surface (mm), yaw the angle
    of the paper's X edge from base X (rad), z_offset a height trim (mm).
    """

    origin: Vec3
    yaw: float
    z_offset: float

    def to_base(self, point: Vec3) -> Vec3:
        """Paper mm to base-frame meters."""
        x, y, z = point
        origin_x, origin_y, origin_z = self.origin
        cos_yaw, sin_yaw = math.cos(self.yaw), math.sin(self.yaw)
        return (
            (origin_x + cos_yaw * x - sin_yaw * y) / MM_PER_M,
            (origin_y + sin_yaw * x + cos_yaw * y) / MM_PER_M,
            (origin_z + self.z_offset + z) / MM_PER_M,
        )

    def to_paper(self, point: Vec3) -> Vec3:
        """Base-frame meters to paper mm."""
        origin_x, origin_y, origin_z = self.origin
        dx = point[0] * MM_PER_M - origin_x
        dy = point[1] * MM_PER_M - origin_y
        cos_yaw, sin_yaw = math.cos(self.yaw), math.sin(self.yaw)
        return (
            cos_yaw * dx + sin_yaw * dy,
            -sin_yaw * dx + cos_yaw * dy,
            point[2] * MM_PER_M - origin_z - self.z_offset,
        )


@dataclass(frozen=True)
class CornerPlacement(Placement):
    """Paper stretched between its corners as the arm reports them.

    corners are on the paper's surface (base frame mm, CORNER_NAMES order) and
    size its width and height (mm). Points between them, heights included, are
    blended bilinearly, so the pen lands on every measured corner however the
    arm's reported frame is skewed. origin and yaw follow the bottom edge.
    """

    corners: tuple[Vec3, Vec3, Vec3, Vec3]
    size: tuple[float, float]

    def to_base(self, point: Vec3) -> Vec3:
        """Paper mm to base-frame meters."""
        x, y, z = point
        width, height = self.size
        base_x, base_y, surface = self._blend(x / width, y / height)
        return (
            base_x / MM_PER_M,
            base_y / MM_PER_M,
            (surface + self.z_offset + z) / MM_PER_M,
        )

    def to_paper(self, point: Vec3) -> Vec3:
        """Base-frame meters to paper mm, inverting the blend by Newton's method."""
        target_x, target_y = point[0] * MM_PER_M, point[1] * MM_PER_M
        s = t = 0.5
        for _ in range(INVERSE_ITERATIONS):
            blend_x, blend_y, surface = self._blend(s, t)
            miss_x, miss_y = target_x - blend_x, target_y - blend_y
            if math.hypot(miss_x, miss_y) < INVERSE_TOLERANCE_MM:
                width, height = self.size
                return (
                    s * width,
                    t * height,
                    point[2] * MM_PER_M - surface - self.z_offset,
                )
            (along_s_x, along_s_y), (along_t_x, along_t_y) = self._edges(s, t)
            determinant = along_s_x * along_t_y - along_t_x * along_s_y
            s += (miss_x * along_t_y - along_t_x * miss_y) / determinant
            t += (along_s_x * miss_y - miss_x * along_s_y) / determinant
        raise DrawError(f"could not find base point {point} on the paper")

    def _blend(self, s: float, t: float) -> Vec3:
        """The point s of the way along the paper and t of the way up it (mm)."""
        bottom_left, bottom_right, top_right, top_left = self.corners
        return tuple(
            (1 - s) * (1 - t) * bl + s * (1 - t) * br + s * t * tr + (1 - s) * t * tl
            for bl, br, tr, tl in zip(bottom_left, bottom_right, top_right, top_left)
        )

    def _edges(
        self, s: float, t: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """How the blended XY moves per unit of s and per unit of t."""
        bottom_left, bottom_right, top_right, top_left = self.corners
        along_s = [
            (1 - t) * (br - bl) + t * (tr - tl)
            for bl, br, tr, tl in zip(bottom_left, bottom_right, top_right, top_left)
        ]
        along_t = [
            (1 - s) * (tl - bl) + s * (tr - br)
            for bl, br, tr, tl in zip(bottom_left, bottom_right, top_right, top_left)
        ]
        return (along_s[0], along_s[1]), (along_t[0], along_t[1])


def corner_placement(corners: list[Vec3], z_offset: float) -> CornerPlacement:
    """The paper between its corners (base frame mm, CORNER_NAMES order), as wide
    and tall as its opposite edges average."""
    bottom_left, bottom_right, top_right, top_left = corners
    width = (_across(bottom_left, bottom_right) + _across(top_left, top_right)) / 2
    height = (_across(bottom_left, top_left) + _across(bottom_right, top_right)) / 2
    if min(width, height) < MIN_PAPER_MM:
        raise ValueError(f"the corners span only {width:.1f} x {height:.1f} mm")
    angles = corner_angles(corners)
    if any(abs(angle - RIGHT_ANGLE_DEG) > MAX_CORNER_SKEW_DEG for angle in angles):
        shown = ", ".join(f"{angle:.0f}" for angle in angles)
        raise ValueError(
            f"the corners meet at {shown} degrees, more than "
            f"{MAX_CORNER_SKEW_DEG:g} off square; list them {', '.join(CORNER_NAMES)}, "
            "anticlockwise seen from above, and check each"
        )
    twist = abs(bottom_left[2] + top_right[2] - bottom_right[2] - top_left[2])
    if twist > MAX_CORNER_TWIST_MM:
        raise ValueError(
            f"the corner heights are {twist:.1f} mm out of one plane (limit "
            f"{MAX_CORNER_TWIST_MM:g} mm); check their Z"
        )
    yaw = math.atan2(bottom_right[1] - bottom_left[1], bottom_right[0] - bottom_left[0])
    return CornerPlacement(bottom_left, yaw, z_offset, tuple(corners), (width, height))


def corner_angles(corners: list[Vec3]) -> list[float]:
    """The angle inside each corner (degrees) going round them; above 180 where
    they run clockwise seen from above, or cross."""
    angles = []
    for previous, corner, following in zip(
        corners[-1:] + corners[:-1], corners, corners[1:] + corners[:1]
    ):
        in_x, in_y = corner[0] - previous[0], corner[1] - previous[1]
        out_x, out_y = following[0] - corner[0], following[1] - corner[1]
        turn = math.atan2(in_x * out_y - in_y * out_x, in_x * out_x + in_y * out_y)
        angles.append(2 * RIGHT_ANGLE_DEG - math.degrees(turn))
    return angles


def _across(a: Vec3, b: Vec3) -> float:
    """Horizontal distance between two base-frame points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


class PenAnnouncer:
    """Publishes the pen in the gripper for the viewer: on each change, then every second."""

    def __init__(self, endpoint: str = PEN_STATUS_ENDPOINT):
        self.socket = zmq.Context.instance().socket(zmq.PUB)
        self.socket.bind(endpoint)
        self.pen: Pen | None = None
        self.sent_at = -math.inf

    def announce(self, pen: Pen) -> None:
        self.pen = pen
        self._send()

    def tick(self) -> None:
        due = time.monotonic() - self.sent_at >= PEN_STATUS_PERIOD_S
        if self.pen is not None and due:
            self._send()

    def close(self) -> None:
        self.socket.close(linger=0)

    def _send(self) -> None:
        self.socket.send_json({"name": self.pen.name, "rgb": list(self.pen.rgb)})
        self.sent_at = time.monotonic()


def parse_gcode(text: str, start: Vec3, rapid_feed: float) -> list[Step]:
    """Moves and pen changes from the G-code gcode.py writes, beginning at start (paper mm).

    Pens come from '; T<n> NAME #RRGGBB' comment lines, and M6 swaps in the pen of
    the last T word, which may share its line.
    """
    tools = tool_table(text)
    position = start
    feed = None
    tool = None
    steps: list[Step] = []
    for number, line in enumerate(text.splitlines(), start=1):
        words = line.split(";", 1)[0].upper().split()
        try:
            selections = [word for word in words if word.startswith("T")]
            words = [word for word in words if not word.startswith("T")]
            if selections:
                tool = _tool_word(selections, tools)
            if not words:
                continue
            command = _command_word(words[0])
            if command == "M2":
                break
            if command == "M6":
                if len(words) > 1:
                    raise ValueError("M6 takes no words besides T")
                if tool is None:
                    raise ValueError("M6 needs a tool selected with a T word first")
                steps.append(PenChange(tool, tools[tool], position))
                continue
            if command in ("G21", "G90"):
                if len(words) > 1:
                    raise ValueError(f"{command} takes no parameters")
                continue
            if command not in ("G0", "G1"):
                raise ValueError(f"unsupported command {command}")
            target, feed = _motion_words(words[1:], position, feed)
            if command == "G1" and feed is None:
                raise ValueError("G1 needs a feed rate (F) first")
        except ValueError as exc:
            raise ValueError(f"line {number}: {exc}") from None
        if target != position:
            rapid = command == "G0"
            steps.append(Move(position, target, rapid_feed if rapid else feed, rapid))
            position = target
    if not any(isinstance(step, Move) for step in steps):
        raise ValueError("the G-code has no moves")
    return steps


def steps_to_draw(
    text: str, start: Vec3, rapid_feed: float, tool: int | None
) -> list[Step]:
    """The G-code's steps from start, or only tool's when one is given (one_pen)."""
    steps = parse_gcode(text, start, rapid_feed)
    return steps if tool is None else one_pen(steps, tool, start, rapid_feed)


def one_pen(steps: list[Step], tool: int, start: Vec3, rapid_feed: float) -> list[Step]:
    """Only tool's moves, for its pen already in the gripper: no pen change.

    From start the pen stays at least as high as it began until it is above the
    first stroke, and after the last stroke it rises to PEN_CHANGE_HEIGHT_MM for
    the operator to swap pens.
    """
    swaps = sum(isinstance(step, PenChange) and step.tool == tool for step in steps)
    if swaps > 1:
        raise ValueError(f"T{tool} is swapped in more than once")
    active = None
    moves: list[Move] = []
    for step in steps:
        if isinstance(step, PenChange):
            active = step.tool
        elif active == tool:
            moves.append(step)
    first = next((index for index, move in enumerate(moves) if not move.rapid), None)
    if first is None:
        raise ValueError(f"T{tool} draws nothing in this G-code")
    moves = moves[first:]
    above_first, last = moves[0].start, moves[-1].end
    height = max(start[2], above_first[2])
    path = [start, (*start[:2], height), (*above_first[:2], height), above_first]
    raised = (*last[:2], max(last[2], PEN_CHANGE_HEIGHT_MM))
    approach = [Move(a, b, rapid_feed, rapid=True) for a, b in pairwise(path) if a != b]
    departure = [Move(last, raised, rapid_feed, rapid=True)] if raised != last else []
    return [*approach, *moves, *departure]


def check_on_paper(text: str, size: tuple[float, float]) -> None:
    """Refuse G-code that goes beyond the paper its corners span."""
    width, height = size
    for step in parse_gcode(text, (0.0, 0.0, 0.0), MAX_FEED_MM_MIN):
        if not isinstance(step, Move):
            continue
        x, y, _ = step.end
        if not (0.0 <= x <= width and 0.0 <= y <= height):
            raise ValueError(
                f"the G-code goes to X{x:g} Y{y:g}, off the {width:.1f} x "
                f"{height:.1f} mm paper the corners span; make it with --paper "
                f"{math.floor(width)}x{math.floor(height)}"
            )


def chains(steps: list[Step]) -> list[Segment]:
    """Runs of drawing moves blend through their corners; anything else stands alone."""
    segments: list[Segment] = []
    for step in steps:
        previous = segments[-1] if segments else None
        if isinstance(step, PenChange):
            segments.append(step)
        elif step.is_drawing and isinstance(previous, list) and previous[-1].is_drawing:
            previous.append(step)
        else:
            segments.append([step])
    return segments


def plan_motion(steps: list[Step], tilt_deg: float) -> tuple[list[Segment], int]:
    """The chains and pen changes to run, and how many hops a tilted pen needs.

    A tilted pen cannot swing round a sharp turn on the paper, so the drawing is
    cut there and the pen hops: it lifts HOP_MM, re-tilts, and comes back down.
    """
    segments: list[Segment] = []
    hops = 0
    for segment in chains(steps):
        if not tilt_deg or isinstance(segment, PenChange) or not segment[0].is_drawing:
            segments.append(segment)
            continue
        first, *rest = split_at_sharp_turns(segment)
        segments.append(first)
        for piece in rest:
            corner = piece[0].start
            above = (*corner[:2], corner[2] + HOP_MM)
            segments.append([Move(corner, above, HOP_FEED_MM_MIN, rapid=True)])
            segments.append([Move(above, corner, HOP_FEED_MM_MIN, rapid=False)])
            segments.append(piece)
            hops += 1
    return segments, hops


def split_at_sharp_turns(chain: list[Move]) -> list[list[Move]]:
    """The chain cut at each corner that turns by more than SHARP_TURN_DEG.

    Turns are measured across TILT_LOOKAHEAD_MM, the distance the tilt looks
    ahead, so a traced corner rounded off by a few short segments still counts.
    """
    points = [chain[0].start[:2], *(move.end[:2] for move in chain)]
    distances = [0.0, *accumulate(math.dist(a, b) for a, b in pairwise(points))]
    corners = tilt.sharp_corners(
        points, distances, TILT_LOOKAHEAD_MM, math.radians(SHARP_TURN_DEG)
    )
    bounds = [0, *corners, len(chain)]
    return [chain[start:end] for start, end in pairwise(bounds)]


def summarize(
    segments: list[Segment], placement: Placement
) -> tuple[Vec3, Vec3, float]:
    """Pen-tip bounds in the base frame (mm), pen changes included, and the motion time (s)."""
    points = []
    seconds = 0.0
    for segment in segments:
        if isinstance(segment, PenChange):
            x, y, z = segment.position
            points.append((x, y, max(z, PEN_CHANGE_HEIGHT_MM)))
            continue
        for move in segment:
            points += [move.start, move.end]
            seconds += move.length / (move.feed / SECONDS_PER_MINUTE)
    base = [placement.to_base(point) for point in points]
    low = tuple(min(p[axis] for p in base) * MM_PER_M for axis in range(3))
    high = tuple(max(p[axis] for p in base) * MM_PER_M for axis in range(3))
    return low, high, seconds


def check_not_in_error(shell: Shell) -> None:
    """Refuse a controller in error instead of letting ensure_running_mode clear it.

    After a real E-stop the drives report zero joint angles until the arm is
    re-referenced (PROTOCOL.md), so twists would move it unpredictably.
    """
    telemetry = shell.telemetry(timeout_s=1.0)
    if telemetry is None:
        raise DrawError("no telemetry; is the controller serving?")
    if telemetry["controller_state"] == CONTROLLER_ERROR:
        raise DrawError(
            f"the controller is in error ({telemetry['fault'] or 'no fault text'}); "
            "recover it yourself, re-referencing the arm after an E-stop, then draw again"
        )


def check_tip_link(shell: Shell) -> None:
    """Refuse a controller whose TCP is not the Hand-E target_frame."""
    shell.req.send(encode_get_model())
    try:
        model = parse_model(shell.req.recv())
    except zmq.Again:
        raise DrawError("no reply to GET_MODEL; is the controller serving?") from None
    if model is None:
        raise DrawError("the controller did not answer GET_MODEL with its model")
    if model["tip_link"] != TIP_LINK:
        raise DrawError(
            f"the controller's tip link is {model['tip_link']!r}, but the pen "
            f"offset is measured along {TIP_LINK!r} of URDF_hand_e.urdf"
        )


def pen_tip_position(shell: Shell, pen_offset: float) -> Vec3:
    """Current pen tip in the base frame (mm)."""
    telemetry = shell.telemetry(timeout_s=1.0)
    if telemetry is None:
        raise DrawError("no telemetry; is the controller serving?")
    tip = pen_tip(telemetry["tcp_pose"], pen_offset)
    return tip[0] * MM_PER_M, tip[1] * MM_PER_M, tip[2] * MM_PER_M


def pen_tip(tcp_pose, pen_offset: float) -> Vec3:
    """Pen tip (m) from a tcp_pose: pen_offset mm beyond target_frame along its +X."""
    axis = tilt.tool_axis(tuple(tcp_pose[3:7]))
    offset = pen_offset / MM_PER_M
    return tuple(p + offset * a for p, a in zip(tcp_pose[0:3], axis))


def draw(
    shell: Shell,
    text: str,
    placement: Placement,
    pen_offset: float,
    rapid_feed: float,
    tilt_deg: float,
    swap_wait: float | None,
    tool: int | None = None,
) -> float:
    """Draw the G-code; returns the largest pen-tip error while drawing (mm).

    The pen leans tilt_deg back from its direction of travel; 0 holds the start
    orientation. Pen changes wait swap_wait seconds, or for Enter when it is None.
    With tool, only that tool's strokes are drawn (one_pen) and the arm is held
    afterwards for the operator to swap pens.
    """
    steps_to_draw(text, (0.0, 0.0, 0.0), rapid_feed, tool)
    check_not_in_error(shell)
    # With no offset the controller's TCP is the pen tip, whatever link it is on.
    if pen_offset:
        check_tip_link(shell)
    if not ensure_running_mode(shell, "world_twist"):
        raise DrawError("could not bring the controller to running in world_twist")
    telemetry = _telemetry(shell)
    start_quat = tuple(telemetry["tcp_pose"][3:7])
    upright = tilt.vertical_from(start_quat) if tilt_deg else start_quat
    start = placement.to_paper(pen_tip(telemetry["tcp_pose"], pen_offset))
    segments, _ = plan_motion(steps_to_draw(text, start, rapid_feed, tool), tilt_deg)
    goals = orientations(segments, placement, upright, math.radians(tilt_deg))
    worst = 0.0
    with closing(PenAnnouncer()) as announcer:
        plotter = Plotter(shell, placement, pen_offset, announcer)
        if tool is not None:
            announcer.announce(tool_table(text)[tool])
        try:
            for segment, goal in zip(segments, goals):
                if isinstance(segment, PenChange):
                    plotter.change_pen(segment, goal, rapid_feed, swap_wait)
                    continue
                error = plotter.follow(segment, goal)
                if segment[0].is_drawing:
                    worst = max(worst, error)
            if tool is not None:
                plotter.hold()
        except DrawError, KeyboardInterrupt:
            plotter.lift()
            raise
        finally:
            plotter.stop()
    return worst


def orientations(
    segments: list[Segment], placement: Placement, upright: Quat, tilt_rad: float
) -> list[Orientation]:
    """How the pen leans in each segment.

    While drawing it trails its direction of travel. Between strokes it already
    leans the way the next stroke starts, and it stands upright for pen changes
    and after the last stroke.
    """
    goals: list[Orientation] = []
    upcoming = _fixed(upright)
    for segment in reversed(segments):
        if isinstance(segment, PenChange):
            goals.append(_fixed(upright))
            upcoming = _fixed(upright)
        elif segment[0].is_drawing:
            lean = _trailing(segment, placement, upright, tilt_rad)
            goals.append(lean)
            upcoming = _fixed(lean(0.0))
        else:
            goals.append(upcoming)
    return goals[::-1]


class Plotter:
    """Moves the pen tip along paper paths by streaming world twists."""

    def __init__(
        self,
        shell: Shell,
        placement: Placement,
        pen_offset: float,
        announcer: PenAnnouncer,
    ):
        self.shell = shell
        self.placement = placement
        self.pen_offset = pen_offset
        self.announcer = announcer

    def follow(self, chain: list[Move], goal: Orientation) -> float:
        """Track the chain at its feeds and turn the pen toward goal, then settle.

        Returns the largest pen-tip error (mm).
        """
        points = [self.placement.to_base(chain[0].start)]
        points += [self.placement.to_base(move.end) for move in chain]
        durations = [m.length / (m.feed / SECONDS_PER_MINUTE) for m in chain]
        ends = list(accumulate(durations))
        lengths = [move.length / MM_PER_M for move in chain]
        distances = list(accumulate(lengths))
        on_paper = chain[-1].end[2] <= 0.0
        period = 1.0 / LOOP_RATE_HZ
        # The reference turns toward the goal no faster than the tilt rate.
        reference_quat = tuple(_telemetry(self.shell)["tcp_pose"][3:7])
        started = previous = next_tick = time.monotonic()
        index = settled = 0
        worst = 0.0
        while True:
            next_tick += period
            telemetry = _telemetry(self.shell)
            self.announcer.tick()
            now = time.monotonic()
            # When this loop stalls, the arm stops at the deadman, so the
            # reference waits out the rest of the stall instead of running ahead.
            started += max(0.0, now - previous - DEADMAN_S)
            previous = now
            elapsed = now - started
            while index < len(chain) and elapsed >= ends[index]:
                index += 1
            if index < len(chain):
                begin, end = points[index], points[index + 1]
                remaining = (ends[index] - elapsed) / durations[index]
                reference = tuple(b - (b - a) * remaining for a, b in zip(begin, end))
                feedforward = tuple(
                    (b - a) / durations[index] for a, b in zip(begin, end)
                )
                travelled = distances[index] - lengths[index] * remaining
            else:
                reference, feedforward = points[-1], (0.0, 0.0, 0.0)
                travelled = distances[-1]

            target_quat = goal(travelled)
            turn = clamp_norm(
                rotation_error(reference_quat, target_quat),
                MAX_TILT_RATE_RAD_S * period,
            )
            reference_quat = tilt.rotate(reference_quat, turn)

            tcp_pose = telemetry["tcp_pose"]
            tcp_quat = tuple(tcp_pose[3:7])
            tip = pen_tip(tcp_pose, self.pen_offset)
            correction, error, _ = pose_servo_step(
                (*tip, *tcp_quat),
                reference,
                reference_quat,
                KP_LIN,
                KP_ANG,
                MAX_CORRECTION_M_S,
                MAX_TILT_RATE_RAD_S,
            )
            error_mm = error * MM_PER_M
            if error_mm > MAX_TRACKING_ERROR_MM:
                raise DrawError(
                    f"pen tip is {error_mm:.1f} mm off its path (limit "
                    f"{MAX_TRACKING_ERROR_MM:g} mm); check --origin or --corners, "
                    "and --pen-offset"
                )
            worst = max(worst, error_mm)
            if index == len(chain):
                unturned = math.hypot(*rotation_error(tcp_quat, target_quat))
                across = math.dist(tip[:2], reference[:2]) * MM_PER_M
                height = (tip[2] - reference[2]) * MM_PER_M
                # Paper that sits a little higher than its corners suggest stops the
                # pen short of its target, so on the paper only the position across
                # it has to settle.
                miss = across if on_paper else error_mm
                still = miss < SETTLE_TOLERANCE_MM and unturned < SETTLE_TOLERANCE_RAD
                settled = settled + 1 if still else 0
                if settled >= SETTLE_CYCLES:
                    return worst
                if elapsed - ends[-1] > SETTLE_TIMEOUT_S:
                    log.warning(
                        "pen tip did not settle within %g s (%.1f mm off across the "
                        "paper, %+.1f mm in height, wrist %.1f deg off); going on",
                        SETTLE_TIMEOUT_S,
                        across,
                        height,
                        math.degrees(unturned),
                    )
                    return worst

            spin = clamp_norm(
                tuple(t / period + c for t, c in zip(turn, correction[3:])),
                MAX_TILT_RATE_RAD_S,
            )
            tip_velocity = tuple(f + c for f, c in zip(feedforward, correction[:3]))
            tip_offset = tuple(t - p for t, p in zip(tip, tcp_pose[0:3]))
            linear = tilt.tcp_velocity(tip_velocity, spin, tip_offset)
            if not self.shell.send_twist((*linear, *spin), quiet=True):
                raise DrawError("the controller rejected the twist command")
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)

    def change_pen(
        self,
        change: PenChange,
        upright: Orientation,
        rapid_feed: float,
        swap_wait: float | None,
    ) -> None:
        """Raise the pen, hold the arm while the pen is swapped, and lower it back."""
        left_from = self.tip()
        raised = (*left_from[:2], max(left_from[2], PEN_CHANGE_HEIGHT_MM))
        self.follow([Move(left_from, raised, rapid_feed, rapid=True)], upright)
        self.hold()
        _swap(self.shell, change.pen, swap_wait)
        self.announcer.announce(change.pen)
        if not ensure_running_mode(self.shell, "world_twist"):
            raise DrawError("could not resume world_twist after the pen change")
        self.follow([Move(raised, left_from, rapid_feed, rapid=True)], upright)

    def hold(self) -> None:
        """Stop, and have the controller brake the arm at standstill."""
        self.stop()
        held = self.shell.send_event("hold")
        if not held or not self.shell.wait_state("power_on", HOLD_TIMEOUT_S):
            raise DrawError("the controller did not hold the arm")

    def tip(self) -> Vec3:
        """The pen tip now, in paper mm."""
        telemetry = _telemetry(self.shell)
        return self.placement.to_paper(pen_tip(telemetry["tcp_pose"], self.pen_offset))

    def lift(self) -> None:
        """Raise the pen off the paper after an abort, keeping its orientation."""
        try:
            telemetry = _telemetry(self.shell)
            tcp_pose = telemetry["tcp_pose"]
            tip = self.placement.to_paper(pen_tip(tcp_pose, self.pen_offset))
            above = (*tip[:2], tip[2] + ABORT_LIFT_MM)
            lift = Move(tip, above, ABORT_LIFT_FEED_MM_MIN, rapid=True)
            self.follow([lift], _fixed(tuple(tcp_pose[3:7])))
        except (DrawError, zmq.ZMQError) as exc:
            log.warning("could not lift the pen: %s", exc)

    def stop(self) -> None:
        try:
            self.shell.send_twist(ZERO_TWIST, quiet=True)
        except zmq.ZMQError as exc:
            log.warning(
                "could not send a zero twist (%s); the controller deadman stops "
                "the arm within 200 ms",
                exc,
            )


def _fixed(quat: Quat) -> Orientation:
    return lambda _travelled: quat


def _trailing(
    chain: list[Move], placement: Placement, upright: Quat, tilt_rad: float
) -> Orientation:
    """The pen leaning tilt_rad back from where the chain heads TILT_LOOKAHEAD_MM on."""
    points = [placement.to_base(chain[0].start)[:2]]
    points += [placement.to_base(move.end)[:2] for move in chain]
    distances = [0.0, *accumulate(math.dist(a, b) for a, b in pairwise(points))]
    lookahead = TILT_LOOKAHEAD_MM / MM_PER_M

    def lean(travelled: float) -> Quat:
        heading = tilt.lookahead_heading(points, distances, travelled, lookahead)
        return tilt.tilted(upright, heading, tilt_rad)

    return lean


def tool_table(text: str) -> dict[int, Pen]:
    """Pens by tool number from '; T<n> NAME #RRGGBB' comment lines."""
    tools = {}
    for number, line in enumerate(text.splitlines(), start=1):
        code, _, comment = line.partition(";")
        entry = TOOL_TABLE_ENTRY.fullmatch(comment.strip())
        if code.strip() or entry is None:
            continue
        tool = int(entry[1])
        try:
            if tool in tools:
                raise ValueError(f"T{tool} is listed twice in the tool table")
            tools[tool] = Pen(entry[2], parse_hex(entry[3]))
        except ValueError as exc:
            raise ValueError(f"line {number}: {exc}") from None
    return tools


def _tool_word(words: list[str], tools: dict[int, Pen]) -> int:
    if len(words) > 1:
        raise ValueError("a line can select only one tool")
    number = words[0][1:]
    if not number.isdigit():
        raise ValueError(f"expected a tool number, got {words[0]}")
    tool = int(number)
    if tool not in tools:
        raise ValueError(f"T{tool} has no '; T{tool} NAME #RRGGBB' tool table line")
    return tool


def _command_word(word: str) -> str:
    letter, number = word[0], word[1:]
    if letter not in "GM" or not number.isdigit():
        raise ValueError(f"expected a G or M command, got {word}")
    return f"{letter}{int(number)}"


def _motion_words(
    words: list[str], position: Vec3, feed: float | None
) -> tuple[Vec3, float | None]:
    axes = dict(zip("XYZ", position))
    for word in words:
        letter, value = word[0], _number(word[1:])
        if letter == "F":
            if not 0 < value <= MAX_FEED_MM_MIN:
                raise ValueError(f"{word} is outside (0, {MAX_FEED_MM_MIN:g}] mm/min")
            feed = value
        elif letter in axes:
            if letter == "Z" and value < MIN_Z_MM:
                raise ValueError(f"{word} is below the {MIN_Z_MM:g} mm floor")
            axes[letter] = value
        else:
            raise ValueError(f"unsupported word {word}")
    return (axes["X"], axes["Y"], axes["Z"]), feed


def _number(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} is not a finite number")
    return value


def _swap(shell: Shell, pen: Pen, swap_wait: float | None) -> None:
    """Let the operator swap pens, through the gripper when one is ready."""
    telemetry = shell.telemetry()
    if telemetry and telemetry["gripper_present"] and telemetry["gripper_activated"]:
        _pause("hold the pen", "open the gripper", swap_wait)
        _move_fingers(shell, GRIPPER_OPEN)
        _pause(f"insert the {pen.name} pen at the same stick-out", "grip it", swap_wait)
        if not _move_fingers(shell, GRIPPER_CLOSED)["gripper_object"]:
            raise DrawError(f"the gripper closed on nothing; is the {pen.name} pen in?")
    else:
        _pause(f"swap in the {pen.name} pen at the same stick-out", "go on", swap_wait)


def _pause(action: str, then: str, swap_wait: float | None) -> None:
    """Wait swap_wait seconds, or for Enter when it is None."""
    if swap_wait is not None:
        print(f"{action}; going on in {swap_wait:g} s")
        time.sleep(swap_wait)
        return
    try:
        input(f"{action}, then press Enter to {then}: ")
    except EOFError:
        raise DrawError(
            "no terminal to confirm the pen change; pass --swap-wait"
        ) from None


def _move_fingers(shell: Shell, position: float) -> dict:
    """Command the gripper and return the telemetry once its fingers come to rest."""
    if not shell.send_gripper_command(position, GRIPPER_SPEED, GRIPPER_FORCE):
        raise DrawError("the gripper refused the command")
    deadline = time.monotonic() + GRIPPER_TIMEOUT_S
    while time.monotonic() < deadline:
        telemetry = shell.telemetry(TELEMETRY_POLL_S)
        if telemetry is None or telemetry["gripper_moving"]:
            continue
        commanded = abs(telemetry["gripper_target"] - position) <= GRIPPER_TOLERANCE
        arrived = abs(telemetry["gripper_position"] - position) <= GRIPPER_TOLERANCE
        if commanded and (arrived or telemetry["gripper_object"]):
            return telemetry
    raise DrawError(f"the gripper did not come to rest within {GRIPPER_TIMEOUT_S:g} s")


def _telemetry(shell: Shell) -> dict:
    deadline = time.monotonic() + STALL_TELEMETRY_S
    while (telemetry := shell.telemetry(timeout_s=TELEMETRY_POLL_S)) is None:
        if time.monotonic() > deadline:
            raise DrawError("telemetry stalled")
    healthy = telemetry["controller_state"] == RUNNING and telemetry["motion_active"]
    if not healthy or telemetry["fault"]:
        raise DrawError(f"controller left motion: {shell.status_line(telemetry)}")
    return telemetry
