import math
from itertools import pairwise

import pytest

from svg_to_gcode import gcode, hatch, pens, robot

CMYK = pens.parse_pens(pens.DEFAULT_PENS)
SQUARE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect x="0" y="0" width="100" height="100" fill="none" stroke="black"/>
</svg>"""
COLOURED_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect x="0" y="0" width="40" height="40" fill="#111111"/>
  <circle cx="70" cy="70" r="20" fill="#FFEE11"/>
  <path d="M 0 90 L 40 90" fill="none" stroke="#0099DD"/>
  <rect x="50" y="0" width="40" height="30" fill="white"/>
</svg>"""
OPEN_TRIANGLE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
  <path d="M 0 10 L 10 10 L 0 0" fill="black"/>
</svg>"""
OVERLAPPING_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
  <g style="mix-blend-mode:multiply"><rect width="10" height="10" fill="#FFF200"/></g>
  <g style="mix-blend-mode:multiply"><rect width="10" height="10" fill="#EC008C"/></g>
</svg>"""
LIFT_MM = 5.0
FEED_MM_MIN = 1500.0
RAPID_MM_MIN = 3000.0
MARGIN_MM = 10.0
OUTER = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
INNER = [(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)]


def layers_for(tmp_path, text: str, **settings) -> list[gcode.Layer]:
    svg_file = tmp_path / "drawing.svg"
    svg_file.write_text(text)
    options = {
        "paper": gcode.parse_paper("A4"),
        "margin": MARGIN_MM,
        "tolerance": 0.1,
        "pens": CMYK,
        "hatch_spacing": 0.8,
        "hatch_angle": 45.0,
        "outline": True,
    }
    return gcode.svg_to_layers(svg_file, **(options | settings))


def test_square_fits_a4_and_round_trips(tmp_path):
    paper = gcode.parse_paper("A4")
    (layer,) = layers_for(tmp_path, SQUARE_SVG)
    assert (layer.tool, layer.pen.name) == (4, "black")

    text = gcode.to_gcode([layer], FEED_MM_MIN, LIFT_MM)
    steps = robot.parse_gcode(text, (0.0, 0.0, 0.0), RAPID_MM_MIN)
    moves = [step for step in steps if isinstance(step, robot.Move)]

    drawn = [p for m in moves if m.is_drawing for p in (m.start, m.end)]
    assert min(x for x, _, _ in drawn) == pytest.approx(MARGIN_MM)
    assert max(x for x, _, _ in drawn) == pytest.approx(paper[0] - MARGIN_MM)
    assert {z for _, _, z in drawn} == {0.0}
    side = paper[0] - 2 * MARGIN_MM
    center_y = paper[1] / 2
    assert min(y for _, y, _ in drawn) == pytest.approx(center_y - side / 2)
    assert max(y for _, y, _ in drawn) == pytest.approx(center_y + side / 2)

    kinds = [
        "change"
        if isinstance(step, robot.PenChange)
        else "draw"
        if step.is_drawing
        else "rapid"
        if step.rapid
        else "plunge"
        for step in steps
    ]
    assert kinds == ["rapid", "change", "rapid", "plunge"] + ["draw"] * 4 + ["rapid"]
    assert moves[0].end[2] == LIFT_MM
    assert moves[-1].end[2] == LIFT_MM
    assert all(m.feed == FEED_MM_MIN for m in moves if m.is_drawing)
    assert all(m.feed == RAPID_MM_MIN for m in moves if m.rapid)


def test_placement_round_trips_through_base_frame():
    placement = robot.Placement(origin=(400.0, -120.0, 35.0), yaw=0.7, z_offset=-0.5)
    point = (12.5, 80.0, 5.0)
    assert placement.to_paper(placement.to_base(point)) == pytest.approx(point)


@pytest.mark.parametrize(
    ("inner", "evenodd", "hole_filled"),
    [
        # Nested rings winding the same way count twice inside the hole.
        (INNER, False, True),
        (INNER[::-1], False, False),
        (INNER, True, False),
        (INNER[::-1], True, False),
    ],
)
def test_hatching_a_square_with_a_hole(inner, evenodd, hole_filled):
    rows: dict[float, list[tuple[float, float]]] = {}
    for (x0, y0), (x1, y1) in hatch.hatch([OUTER, inner], 1.0, 0.0, evenodd):
        assert y0 == pytest.approx(y1)
        assert x0 < x1
        rows.setdefault(round(y0, 9), []).append((round(x0, 9), round(x1, 9)))

    assert sorted(rows) == list(range(10))
    assert rows[1] == [(0, 10)]
    assert rows[5] == ([(0, 10)] if hole_filled else [(0, 3), (7, 10)])


def test_hatch_lines_follow_the_angle_and_spacing():
    angle = math.radians(30.0)
    strokes = hatch.hatch([OUTER], 0.5, 30.0, evenodd=False)

    offsets, total = [], 0.0
    for start, end in strokes:
        length = math.dist(start, end)
        direction = ((end[0] - start[0]) / length, (end[1] - start[1]) / length)
        assert direction == pytest.approx((math.cos(angle), math.sin(angle)))
        offsets.append(-start[0] * math.sin(angle) + start[1] * math.cos(angle))
        total += length
    assert {round(b - a, 9) for a, b in pairwise(offsets)} == {0.5}
    assert total == pytest.approx(hatch.ring_area(OUTER) / 0.5, rel=0.1)


def test_each_pen_hatches_at_its_own_angle(tmp_path):
    yellow, magenta = layers_for(tmp_path, OVERLAPPING_SVG, outline=False)

    def directions(layer: gcode.Layer) -> set[float]:
        # Strokes may run either way along their line to cut travel.
        return {
            round(math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0, 6)
            for (x0, y0), (x1, y1) in layer.strokes
        }

    assert (yellow.pen.name, magenta.pen.name) == ("yellow", "magenta")
    # Four pens a quarter of a half turn apart: magenta, the third, turns 90 degrees.
    assert directions(yellow) == {45.0}
    assert directions(magenta) == {135.0}


def test_a_fills_opacity_hatches_its_pen_that_much_sparser(tmp_path):
    def hatch_lines(opacity: str) -> list[gcode.Stroke]:
        (layer,) = layers_for(
            tmp_path,
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
            f'<rect width="10" height="10" fill="#EC008C" fill-opacity="{opacity}"/>'
            "</svg>",
        )
        assert layer.pen.name == "magenta"
        return layer.strokes

    solid, half = hatch_lines("1"), hatch_lines("0.5")

    assert any(len(stroke) > 2 for stroke in solid), "solid fills are outlined"
    assert all(len(stroke) == 2 for stroke in half), "partial fills are not"
    assert len(half) == pytest.approx((len(solid) - 1) / 2, abs=1)


def test_open_subpaths_fill_as_if_closed(tmp_path):
    (layer,) = layers_for(tmp_path, OPEN_TRIANGLE_SVG, paper=(30.0, 30.0))
    (outline,) = [stroke for stroke in layer.strokes if len(stroke) > 2]
    hatches = [stroke for stroke in layer.strokes if stroke is not outline]

    assert outline[0] == outline[-1]
    assert sorted(outline[:-1]) == [(10.0, 10.0), (10.0, 20.0), (20.0, 10.0)]
    assert len(hatches) > 5
    for stroke in hatches:
        for x, y in stroke:
            assert x >= 10.0 - 1e-6 and y >= 10.0 - 1e-6 and x + y <= 30.0 + 1e-6


def test_strokes_run_nearest_first():
    near_backwards = [(10.0, 0.0), (5.0, 0.0)]
    ring = [(20.0, 0.0), (20.0, 5.0), (15.0, 5.0), (15.0, 0.0), (20.0, 0.0)]
    far = [(30.0, 0.0), (40.0, 0.0)]

    ordered = gcode.order_strokes([far, ring, near_backwards], (0.0, 0.0))

    assert ordered == [
        [(5.0, 0.0), (10.0, 0.0)],
        [(15.0, 0.0), (20.0, 0.0), (20.0, 5.0), (15.0, 5.0), (15.0, 0.0)],
        far,
    ]


def test_pen_stays_down_into_a_stroke_that_starts_where_the_last_ended():
    strokes = [
        [(0.0, 0.0), (10.0, 0.0)],
        [(10.0, 0.0), (10.0, 10.0)],
        [(20.0, 0.0), (30.0, 0.0)],
    ]
    text = gcode.to_gcode([gcode.Layer(1, CMYK[3], strokes)], FEED_MM_MIN, LIFT_MM)

    steps = robot.parse_gcode(text, (0.0, 0.0, LIFT_MM), RAPID_MM_MIN)
    inked = [
        len(chain)
        for chain in robot.chains(steps)
        if isinstance(chain, list) and chain[0].is_drawing
    ]
    assert text.count("G1 Z0.000") == 2
    assert inked == [2, 1]


def test_high_lift_rapids_down_to_the_approach_height_before_plunging():
    high_lift_mm = 50.0
    strokes = [[(0.0, 0.0), (10.0, 0.0)], [(20.0, 0.0), (30.0, 0.0)]]
    text = gcode.to_gcode([gcode.Layer(1, CMYK[3], strokes)], FEED_MM_MIN, high_lift_mm)

    steps = robot.parse_gcode(text, (0.0, 0.0, high_lift_mm), RAPID_MM_MIN)
    moves = [step for step in steps if isinstance(step, robot.Move)]
    second_stroke = moves.index(next(m for m in moves if m.end == (20.0, 0.0, 0.0)))
    lift, travel, approach, plunge = moves[second_stroke - 3 : second_stroke + 1]
    assert lift.end == (10.0, 0.0, high_lift_mm) and lift.rapid
    assert travel.end == (20.0, 0.0, high_lift_mm) and travel.rapid
    assert approach.end == (20.0, 0.0, gcode.APPROACH_MM) and approach.rapid
    assert plunge.feed == gcode.PLUNGE_FEED_MM_MIN and not plunge.rapid


def test_fills_without_hatch_or_outline_draw_nothing(tmp_path):
    with pytest.raises(ValueError, match="nothing to draw"):
        layers_for(tmp_path, OPEN_TRIANGLE_SVG, hatch_spacing=0.0, outline=False)


def test_layers_follow_the_pens_with_a_tool_table(tmp_path):
    layers = layers_for(tmp_path, COLOURED_SVG)
    assert [(layer.tool, layer.pen.name) for layer in layers] == [
        (1, "yellow"),
        (2, "cyan"),
        (4, "black"),
    ]
    assert len(layers[1].strokes) == 1

    lines = gcode.to_gcode(layers, FEED_MM_MIN, LIFT_MM).splitlines()
    assert lines[:6] == [
        "G21",
        "G90",
        "; T1 yellow #FFF200",
        "; T2 cyan #00AEEF",
        "; T4 black #231F20",
        "G0 Z5.000",
    ]
    assert [line for line in lines if "M6" in line] == ["T1 M6", "T2 M6", "T4 M6"]


def test_pen_changes_round_trip_through_the_robot_parser(tmp_path):
    layers = layers_for(tmp_path, COLOURED_SVG)
    text = gcode.to_gcode(layers, FEED_MM_MIN, LIFT_MM)
    steps = robot.parse_gcode(text, (0.0, 0.0, 0.0), RAPID_MM_MIN)

    changes = [step for step in steps if isinstance(step, robot.PenChange)]
    assert [(change.tool, change.pen) for change in changes] == [
        (layer.tool, layer.pen) for layer in layers
    ]
    assert all(change.position[2] == LIFT_MM for change in changes)
    drawn = [s for s in steps if isinstance(s, robot.Move) and s.is_drawing]
    expected = sum(len(stroke) - 1 for layer in layers for stroke in layer.strokes)
    assert len(drawn) == expected
