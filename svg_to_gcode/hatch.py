import math
from collections.abc import Sequence

import numpy as np

type Point = tuple[float, float]
type Stroke = list[Point]
type Ring = Sequence[Point] | np.ndarray

# Spans shorter than this are tangent touches with no area to ink.
MIN_SPAN = 1e-6


def ring_area(ring: Ring) -> float:
    """Area enclosed by a ring of points, closed implicitly."""
    points = np.asarray(ring, dtype=float).reshape(-1, 2)
    x, y = points[:, 0], points[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def hatch(
    rings: list[Ring], spacing: float, angle_deg: float, evenodd: bool
) -> list[Stroke]:
    """Parallel lines filling the rings as SVG fills them, each ring closed implicitly.

    The lines run at angle_deg from +X, spacing apart on a grid through the origin,
    and every line runs the same way. A point is inside where the rings wind
    around it a non-zero number of times, or an odd number with evenodd.
    """
    if spacing <= 0:
        raise ValueError(f"hatch spacing must be positive, got {spacing:g}")
    angle = math.radians(angle_deg)
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    to_hatch_frame = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    starts, ends = [], []
    for ring in rings:
        points = np.asarray(ring, dtype=float).reshape(-1, 2) @ to_hatch_frame
        starts.append(points)
        ends.append(np.roll(points, -1, axis=0))
    if not starts:
        return []
    start, end = np.concatenate(starts), np.concatenate(ends)
    sloped = start[:, 1] != end[:, 1]
    start, end = start[sloped], end[sloped]
    if not len(start):
        return []
    winding = np.where(end[:, 1] > start[:, 1], 1, -1)
    low = np.minimum(start[:, 1], end[:, 1])
    high = np.maximum(start[:, 1], end[:, 1])

    strokes = []
    first_line = math.ceil(low.min() / spacing)
    last_line = math.floor(high.max() / spacing)
    for line in range(first_line, last_line + 1):
        v = line * spacing
        # Half-open so a line through a vertex counts the crossing once.
        crossing = (low <= v) & (v < high)
        if not crossing.any():
            continue
        a, b = start[crossing], end[crossing]
        u = a[:, 0] + (v - a[:, 1]) / (b[:, 1] - a[:, 1]) * (b[:, 0] - a[:, 0])
        order = np.argsort(u, kind="stable")
        u = u[order]
        turns = np.cumsum(winding[crossing][order])
        inside = turns % 2 == 1 if evenodd else turns != 0
        was_inside = np.concatenate([[False], inside[:-1]])
        for enter, leave in zip(u[inside & ~was_inside], u[~inside & was_inside]):
            if leave - enter >= MIN_SPAN:
                strokes.append(_from_hatch_frame(enter, leave, v, to_hatch_frame))
    return strokes


def _from_hatch_frame(
    enter: float, leave: float, v: float, to_hatch_frame: np.ndarray
) -> Stroke:
    span = np.array([[enter, v], [leave, v]]) @ to_hatch_frame.T
    return [(float(x), float(y)) for x, y in span]
