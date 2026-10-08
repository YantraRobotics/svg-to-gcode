import math
from bisect import bisect_left, bisect_right
from collections.abc import Sequence

from motion import quat_mul

type Vec2 = tuple[float, float]
type Vec3 = tuple[float, float, float]
type Quat = tuple[float, float, float, float]

DOWN: Vec3 = (0.0, 0.0, -1.0)
MAX_START_TILT_DEG = 20.0
# Rotations below this many radians are no rotation at all.
MIN_ROTATION = 1e-12
# A chord shorter than this fraction of the look-ahead has no reliable direction.
MIN_CHORD_FRACTION = 1e-3


def tool_axis(quat: Quat) -> Vec3:
    """The tool's +X axis in the base frame: along the pen, toward its tip."""
    qx, qy, qz, qw = quat
    return (
        1.0 - 2.0 * (qy * qy + qz * qz),
        2.0 * (qx * qy + qz * qw),
        2.0 * (qx * qz - qy * qw),
    )


def rotate(quat: Quat, rotation: Vec3) -> Quat:
    """quat turned by a base-frame rotation vector (axis times angle in rad)."""
    angle = math.hypot(*rotation)
    if angle < MIN_ROTATION:
        return quat
    scale = math.sin(angle / 2) / angle
    turn = (*(component * scale for component in rotation), math.cos(angle / 2))
    return quat_mul(turn, quat)


def vertical_from(start: Quat) -> Quat:
    """start turned the least amount that points the pen straight down.

    Raises ValueError when start is more than MAX_START_TILT_DEG off vertical.
    """
    axis = tool_axis(start)
    across = _cross(axis, DOWN)
    sin_angle = math.hypot(*across)
    angle = math.atan2(sin_angle, _dot(axis, DOWN))
    if angle > math.radians(MAX_START_TILT_DEG):
        raise ValueError(
            f"the pen starts {math.degrees(angle):.0f} degrees off vertical; point "
            f"it straight down (within {MAX_START_TILT_DEG:g} degrees) to tilt it"
        )
    if sin_angle < MIN_ROTATION:
        return start
    return rotate(start, tuple(c * angle / sin_angle for c in across))


def tilted(vertical: Quat, heading: float, tilt: float) -> Quat:
    """vertical leaned tilt rad so the pen's top points along heading and its tip trails.

    The turn is about the horizontal axis (-sin h, cos h, 0), which puts the pen
    axis at (-sin t cos h, -sin t sin h, -cos t).
    """
    return rotate(vertical, (-math.sin(heading) * tilt, math.cos(heading) * tilt, 0.0))


def tcp_velocity(tip_velocity: Vec3, angular_velocity: Vec3, tip_offset: Vec3) -> Vec3:
    """The tool-origin velocity that moves the tip at tip_velocity while the tool turns.

    A world twist's linear part moves the tool origin, and turning swings the tip,
    tip_offset away from it, by angular_velocity x tip_offset.
    """
    swing = _cross(angular_velocity, tip_offset)
    return tuple(v - s for v, s in zip(tip_velocity, swing))


def turn_angle(first: Vec2, second: Vec2) -> float:
    """Angle (rad, 0 to pi) between two directions."""
    cross = first[0] * second[1] - first[1] * second[0]
    dot = first[0] * second[0] + first[1] * second[1]
    return math.atan2(abs(cross), dot)


def lookahead_heading(
    points: Sequence[Vec2],
    distances: Sequence[float],
    travelled: float,
    lookahead: float,
) -> float:
    """Heading (rad from +X) from travelled along a polyline to lookahead further on.

    distances[i] is the arc length at points[i]. Where the chord is too short to
    point anywhere, as at the end, the heading of the segment underfoot is used.
    """
    here = _point_along(points, distances, travelled)
    ahead = _point_along(points, distances, min(travelled + lookahead, distances[-1]))
    dx, dy = ahead[0] - here[0], ahead[1] - here[1]
    if math.hypot(dx, dy) <= MIN_CHORD_FRACTION * lookahead:
        index = _segment_at(distances, travelled)
        dx = points[index + 1][0] - points[index][0]
        dy = points[index + 1][1] - points[index][1]
    return math.atan2(dy, dx)


def sharp_corners(
    points: Sequence[Vec2], distances: Sequence[float], window: float, limit: float
) -> list[int]:
    """Indices of the vertices where a polyline turns by more than limit rad.

    The turn at a vertex is between the chords from window back and to window
    ahead along the polyline, so a corner rounded off within window still counts.
    Each corner is reported once, at its sharpest vertex.
    """
    turns = [
        _turn_across(points, distances, index, window)
        for index in range(1, len(points) - 1)
    ]
    turns = [0.0, *turns, 0.0]
    corners = []
    for index in range(1, len(points) - 1):
        if turns[index] <= limit:
            continue
        low = bisect_left(distances, distances[index] - window)
        high = bisect_right(distances, distances[index] + window)
        sharpest = max(range(low, high), key=lambda other: (turns[other], -other))
        if sharpest == index:
            corners.append(index)
    return corners


def _turn_across(
    points: Sequence[Vec2], distances: Sequence[float], index: int, window: float
) -> float:
    here = points[index]
    behind = _point_along(points, distances, max(distances[index] - window, 0.0))
    ahead = _point_along(
        points, distances, min(distances[index] + window, distances[-1])
    )
    incoming = (here[0] - behind[0], here[1] - behind[1])
    outgoing = (ahead[0] - here[0], ahead[1] - here[1])
    shortest = min(math.hypot(*incoming), math.hypot(*outgoing))
    if shortest <= MIN_CHORD_FRACTION * window:
        return 0.0
    return turn_angle(incoming, outgoing)


def _segment_at(distances: Sequence[float], travelled: float) -> int:
    return min(max(bisect_right(distances, travelled) - 1, 0), len(distances) - 2)


def _point_along(
    points: Sequence[Vec2], distances: Sequence[float], travelled: float
) -> Vec2:
    index = _segment_at(distances, travelled)
    (x0, y0), (x1, y1) = points[index], points[index + 1]
    fraction = (travelled - distances[index]) / (
        distances[index + 1] - distances[index]
    )
    return x0 + (x1 - x0) * fraction, y0 + (y1 - y0) * fraction


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Vec3, b: Vec3) -> float:
    return sum(x * y for x, y in zip(a, b))
