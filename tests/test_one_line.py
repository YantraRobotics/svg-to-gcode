import math
import re

import numpy as np
import pytest

from svg_to_gcode import one_line

FIELD_PX = 60
CURVE_SAMPLES = 17


def bezier_segments(data: str) -> list[np.ndarray]:
    """Per subpath, its cubic Bezier segments, shape (segments, 4 points, 2)."""
    found = []
    for subpath in data.split("M")[1:]:
        numbers = np.array(re.findall(r"-?[\d.]+", subpath), dtype=float)
        rest = numbers[2:].reshape(-1, 3, 2)
        starts = np.vstack([numbers[:2], rest[:-1, 2]])
        found.append(np.concatenate([starts[:, np.newaxis], rest], axis=1))
    return found


def curve_length(data: str) -> float:
    t = np.linspace(0.0, 1.0, CURVE_SAMPLES)[:, np.newaxis, np.newaxis]
    total = 0.0
    for segments in bezier_segments(data):
        p0, p1, p2, p3 = (segments[:, corner] for corner in range(4))
        curve = (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t**2 * p2
            + t**3 * p3
        )
        total += np.linalg.norm(np.diff(curve, axis=0), axis=-1).sum()
    return total


def test_scatter_puts_points_in_proportion_to_density():
    density = np.zeros((FIELD_PX, FIELD_PX))
    half = FIELD_PX // 2
    density[:, :half], density[:, half:] = 0.1, 0.4
    count = round(density.sum())

    points = one_line.scatter(density, count, np.random.default_rng(0))

    assert points.shape == (count, 2)
    assert ((points >= 0) & (points <= FIELD_PX)).all()
    sparse, dense = np.sum(points[:, 0] < half), np.sum(points[:, 0] >= half)
    assert dense / sparse == pytest.approx(4, rel=0.1)


def test_tour_is_close_to_the_shortest_round_random_points():
    count = 500
    points = np.random.default_rng(0).uniform(0, FIELD_PX, (count, 2))

    order = one_line.tour(points)

    assert sorted(order) == list(range(count))
    loop = points[order]
    length = np.linalg.norm(np.roll(loop, -1, axis=0) - loop, axis=1).sum()
    # Shortest tours through n random points in area A run about 0.7124 sqrt(n A);
    # nearest neighbour alone runs about a quarter longer.
    assert length < 0.8 * math.sqrt(count * FIELD_PX**2)


def test_split_cuts_the_tour_only_at_long_jumps():
    row = np.column_stack([np.arange(10.0), np.zeros(10)])
    path = np.vstack([row, row[::-1] + [0.0, 50.0]])

    assert [len(piece) for piece in one_line.split(path, 3)] == [10, 10]
    assert [len(piece) for piece in one_line.split(path, 0)] == [20]


def test_curve_data_bends_smoothly_from_end_to_end():
    path = np.array([[0, 0], [10, 0], [10, 10], [20, 10], [20, 0]], dtype=float)

    data = one_line.curve_data([path])

    assert re.fullmatch(r"M[\d.,]+( C[\d.,]+ [\d.,]+ [\d.,]+)+", data)
    (segments,) = bezier_segments(data)
    assert segments[0, 0] == pytest.approx(path[0])
    assert segments[-1, 3] == pytest.approx(path[-1])
    arriving = segments[:-1, 3] - segments[:-1, 2]
    leaving = segments[1:, 1] - segments[1:, 0]
    assert leaving == pytest.approx(arriving, abs=0.02)


def test_draw_inks_the_share_the_field_asks_for():
    field = np.full((FIELD_PX, FIELD_PX), 0.4)

    paths = one_line.draw(field, 3, np.random.default_rng(0))

    assert 1 <= len(paths) <= 4
    length = curve_length(one_line.curve_data(paths))
    assert length == pytest.approx(0.4 * FIELD_PX**2, rel=0.05)
    assert (
        one_line.draw(np.zeros((FIELD_PX, FIELD_PX)), 3, np.random.default_rng(0)) == []
    )


def test_draw_lays_less_line_when_the_line_is_wider():
    field = np.full((2 * FIELD_PX, 2 * FIELD_PX), 0.4)
    line_px = 2.0

    paths = one_line.draw(field, 3, np.random.default_rng(0), line_px)

    length = curve_length(one_line.curve_data(paths))
    assert length == pytest.approx(0.4 * field.size / line_px, rel=0.05)
