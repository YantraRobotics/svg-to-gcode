import math

import numpy as np
from PIL import Image
from scipy.spatial import KDTree

PIXEL_CENTER = 0.5
# Points are relaxed over a grid this many times finer than the field per side.
RELAX_SUBDIVISIONS = 4
RELAX_ITERATIONS = 30
# 2-opt only tries joining each point to its nearest few, which finds nearly all
# of the gain at a fraction of the cost of trying every pair.
TOUR_NEIGHBOURS = 8
MAX_TOUR_PASSES = 50
MIN_TWO_OPT_POINTS = 4
MIN_GAIN_PX = 1e-9
# Smoothed, a tour through evenly spread points at density rho runs about
# TOUR_STEP / sqrt(rho) per point, so it draws TOUR_STEP * sqrt(rho) of line per unit
# area (measured on even fields: the tour itself runs 1.01, smoothing cuts corners).
TOUR_STEP = 0.84
# Only steps this many times longer than the tour's median step count as jumps
# worth lifting the pen over; shorter ones are drawn.
JUMP_RATIO = 3.0


def draw(
    field: np.ndarray, breaks: int, rng: np.random.Generator, line_px: float = 1.0
) -> list[np.ndarray]:
    """At most breaks + 1 paths of points in px whose curves (curve_data) ink a field.

    With lines line_px wide, the line covers about the share of each area that the
    field asks for: it wanders through points scattered at the density that makes
    a tour through them that long, and is cut open at its longest jumps.
    """
    density = (np.clip(field, 0.0, 1.0) / (line_px * TOUR_STEP)) ** 2
    count = round(float(density.sum()))
    if count < 2:
        return []
    points = scatter(density, count, rng)
    return split(points[tour(points)], breaks)


def scatter(density: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """count points (x, y) in px, spread evenly but more densely where density is higher.

    Weighted Lloyd relaxation: starting from grid cells drawn in proportion to the
    density, each point moves to the density-weighted centroid of the cells nearest
    to it.
    """
    height, width = density.shape
    fine_size = (width * RELAX_SUBDIVISIONS, height * RELAX_SUBDIVISIONS)
    upsampled = Image.fromarray(density.astype(np.float32)).resize(
        fine_size, Image.Resampling.BILINEAR
    )
    fine = np.asarray(upsampled, dtype=float)
    rows, columns = np.nonzero(fine > 0)
    weights = fine[rows, columns]
    cells = (np.column_stack([columns, rows]) + PIXEL_CENTER) / RELAX_SUBDIVISIONS
    start = rng.choice(len(cells), size=count, replace=False, p=weights / weights.sum())
    jitter = rng.uniform(-PIXEL_CENTER, PIXEL_CENTER, (count, 2)) / RELAX_SUBDIVISIONS
    points = cells[start] + jitter
    for _ in range(RELAX_ITERATIONS):
        _, nearest = KDTree(points).query(cells, workers=-1)
        mass = np.bincount(nearest, weights, minlength=count)
        held = mass > 0
        for axis in range(2):
            moment = np.bincount(nearest, weights * cells[:, axis], minlength=count)
            points[held, axis] = moment[held] / mass[held]
    return points


def tour(points: np.ndarray) -> np.ndarray:
    """Indices of a short closed tour through every point: nearest neighbour, then 2-opt."""
    order = _nearest_neighbour_tour(points)
    if len(order) >= MIN_TWO_OPT_POINTS:
        _two_opt(points, order)
    return order


def split(path: np.ndarray, breaks: int) -> list[np.ndarray]:
    """A closed tour cut open at its longest jumps into at most breaks + 1 paths."""
    steps = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
    longest = np.argsort(steps)[::-1][: breaks + 1]
    cuts = np.sort(longest[steps[longest] > JUMP_RATIO * np.median(steps)])
    if len(cuts) == 0:
        cuts = np.array([np.argmax(steps)])
    opened = np.roll(path, -(cuts[0] + 1), axis=0)
    pieces = np.split(opened, cuts[1:] - cuts[0])
    return [piece for piece in pieces if len(piece) > 1]


def curve_data(paths: list[np.ndarray]) -> str:
    """SVG path data with one smooth subpath per path.

    Each subpath is the uniform cubic B-spline with the path's points as control
    points, written as Bezier segments and clamped to begin and end on the path's
    end points, so it bends round each point instead of turning sharply at it.
    """
    subpaths = []
    for path in paths:
        controls = np.concatenate([path[:1], path[:1], path, path[-1:], path[-1:]])
        before, start, end, after = (
            controls[:-3],
            controls[1:-2],
            controls[2:-1],
            controls[3:],
        )
        begin = (before[0] + 4 * start[0] + end[0]) / 6
        first = (2 * start + end) / 3
        second = (start + 2 * end) / 3
        finish = (start + 4 * end + after) / 6
        segments = " ".join(
            f"C{a:.2f},{b:.2f} {c:.2f},{d:.2f} {e:.2f},{f:.2f}"
            for (a, b), (c, d), (e, f) in zip(first, second, finish)
        )
        subpaths.append(f"M{begin[0]:.2f},{begin[1]:.2f} {segments}")
    return " ".join(subpaths)


def _nearest_neighbour_tour(points: np.ndarray) -> np.ndarray:
    unvisited = np.ones(len(points), dtype=bool)
    order = np.empty(len(points), dtype=int)
    current = 0
    for step in range(len(points)):
        order[step] = current
        unvisited[current] = False
        candidates = np.flatnonzero(unvisited)
        if len(candidates) == 0:
            break
        gaps = np.sum((points[candidates] - points[current]) ** 2, axis=1)
        current = candidates[np.argmin(gaps)]
    return order


def _two_opt(points: np.ndarray, order: np.ndarray) -> None:
    """Shorten a closed tour in place by reversing stretches of it that cross."""
    count = len(order)
    near_gaps, near = KDTree(points).query(points, k=min(TOUR_NEIGHBOURS + 1, count))
    near_gaps, near = near_gaps[:, 1:].tolist(), near[:, 1:].tolist()
    xy = points.tolist()
    position = np.empty(count, dtype=int)
    position[order] = np.arange(count)
    for _ in range(MAX_TOUR_PASSES):
        improved = False
        for a in range(count):
            for step in (1, -1):
                b = int(order[(position[a] + step) % count])
                ab = math.dist(xy[a], xy[b])
                for c, ac in zip(near[a], near_gaps[a]):
                    if ac >= ab:
                        break
                    d = int(order[(position[c] + step) % count])
                    if a in (c, d) or c == b:
                        continue
                    gain = ab + math.dist(xy[c], xy[d]) - ac - math.dist(xy[b], xy[d])
                    if gain > MIN_GAIN_PX:
                        if step == 1:
                            _reverse(order, position, position[b], position[c])
                        else:
                            _reverse(order, position, position[a], position[d])
                        improved = True
                        break
        if not improved:
            return


def _reverse(order: np.ndarray, position: np.ndarray, first: int, last: int) -> None:
    """Reverse the tour from index first forwards to index last, wrapping round."""
    count = len(order)
    length = (last - first) % count + 1
    # Reversing the rest of the tour instead gives the same tour, mirrored.
    if 2 * length > count:
        first, last = (last + 1) % count, (first - 1) % count
        length = count - length
    stretch = (first + np.arange(length)) % count
    order[stretch] = order[stretch[::-1]]
    position[order[stretch]] = stretch
