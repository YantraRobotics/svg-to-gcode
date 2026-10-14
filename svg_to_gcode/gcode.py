import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import svgelements as svg

from svg_to_gcode.hatch import hatch, ring_area
from svg_to_gcode.pens import PAPER_RGB, RGB, Pen, nearest

log = logging.getLogger(__name__)

type Point = tuple[float, float]
type Stroke = list[Point]
type Bounds = tuple[float, float, float, float]

PAPER_SIZES_MM = {
    "A3": (297.0, 420.0),
    "A4": (210.0, 297.0),
    "A5": (148.0, 210.0),
    "LETTER": (215.9, 279.4),
}
PLUNGE_FEED_MM_MIN = 600.0
MIN_HATCH_MM = 0.1
HALF_TURN_DEG = 180.0
OPAQUE = 255
# Below this depth an S-bend, whose midpoint sits on its chord, would pass as straight.
MIN_SUBDIVISION_DEPTH = 2
MAX_SUBDIVISION_DEPTH = 12


@dataclass(frozen=True)
class Shape:
    """An SVG shape and the pens nearest its fill and stroke (None draws nothing)."""

    path: svg.Path
    fill: Pen | None
    stroke: Pen | None
    evenodd: bool


@dataclass(frozen=True)
class Layer:
    """One pen's tool number and its strokes in paper mm."""

    tool: int
    pen: Pen
    strokes: list[Stroke]


def parse_paper(text: str) -> tuple[float, float]:
    """Paper width and height in mm from a name like A4 or WIDTHxHEIGHT."""
    if text.upper() in PAPER_SIZES_MM:
        return PAPER_SIZES_MM[text.upper()]
    try:
        width, height = (float(part) for part in text.lower().split("x"))
    except ValueError:
        names = ", ".join(PAPER_SIZES_MM)
        raise ValueError(
            f"paper must be one of {names} or WIDTHxHEIGHT in mm, got {text!r}"
        ) from None
    if not (0 < width < math.inf and 0 < height < math.inf):
        raise ValueError(f"paper size must be positive and finite, got {text!r}")
    return width, height


def svg_to_layers(
    svg_file: Path,
    paper: tuple[float, float],
    margin: float,
    tolerance: float,
    pens: list[Pen],
    hatch_spacing: float,
    hatch_angle: float,
    outline: bool,
) -> list[Layer]:
    """Each pen's strokes, in pen order, fitted inside the margin of the paper.

    Paper coordinates are mm with the origin bottom-left and Y up. Fills are hatched
    hatch_spacing mm apart (spacing 0 leaves them empty), the first pen's at
    hatch_angle degrees and each later pen's 180/len(pens) degrees further round, so
    fills that overlap to mix colours cross-hatch instead of retracing each other.
    Fills are outlined when outline is set; strokes are drawn along their paths.
    Each pen's strokes run nearest first (order_strokes) from where the pen before
    it stopped, the first pen from the paper's origin.
    """
    shapes = _load_shapes(svg_file, pens)
    bounds = _bounds([shape.path for shape in shapes])
    scale = _fit_scale(bounds, paper, margin)
    angle_step = HALF_TURN_DEG / len(pens)
    strokes: dict[Pen, list[Stroke]] = {pen: [] for pen in pens}
    for shape in shapes:
        polylines = [
            _to_paper(points, bounds, scale, paper)
            for points in _flatten(shape.path, tolerance / scale)
        ]
        if shape.fill is not None:
            spacing = (
                shape.fill.nib_mm
                if shape.fill.nib_mm is not None and hatch_spacing > 0
                else hatch_spacing
            )
            if spacing > 0:
                angle = hatch_angle + angle_step * pens.index(shape.fill)
                strokes[shape.fill] += hatch(polylines, spacing, angle, shape.evenodd)
            if outline:
                strokes[shape.fill] += [
                    _closed(ring) for ring in polylines if ring_area(ring) > 0
                ]
        if shape.stroke is not None:
            strokes[shape.stroke] += [line for line in polylines if len(line) > 1]
    layers = []
    position = (0.0, 0.0)
    for tool, pen in enumerate(pens, start=1):
        if strokes[pen]:
            ordered = order_strokes(strokes[pen], position)
            layers.append(Layer(tool, pen, ordered))
            position = ordered[-1][-1]
    if not layers:
        raise ValueError(f"{svg_file} has nothing to draw with these pens")
    return layers


def order_strokes(strokes: list[Stroke], start: Point) -> list[Stroke]:
    """The strokes in the order a greedy nearest-neighbour tour from start visits them.

    The pen enters an open stroke at its nearer end, reversing it if need be, and a
    closed ring at its nearest point, so it travels less between strokes.
    """
    closed = [len(stroke) > 2 and stroke[0] == stroke[-1] for stroke in strokes]
    entries = [
        stroke[:-1] if ring else [stroke[0], stroke[-1]]
        for stroke, ring in zip(strokes, closed)
    ]
    points = np.array([point for entry in entries for point in entry])
    bounds = np.cumsum([0, *(len(entry) for entry in entries)])
    unvisited = np.ones(len(points), dtype=bool)
    position = start
    ordered = []
    for _ in strokes:
        gaps = np.where(unvisited, np.hypot(*(points - position).T), np.inf)
        nearest = int(np.argmin(gaps))
        index = int(np.searchsorted(bounds, nearest, side="right")) - 1
        entry = nearest - bounds[index]
        unvisited[bounds[index] : bounds[index + 1]] = False
        stroke = strokes[index]
        if closed[index]:
            ring = stroke[:-1]
            stroke = [*ring[entry:], *ring[:entry], ring[entry]]
        elif entry:
            stroke = stroke[::-1]
        ordered.append(stroke)
        position = stroke[-1]
    return ordered


def to_gcode(layers: list[Layer], feed: float, lift: float) -> str:
    """G-code drawing each layer after a T<n> M6 pen change, listing the pens in comments.

    The pen stays down into a stroke that starts where the one before it ended.
    """
    lines = ["G21", "G90"]
    lines += [f"; T{layer.tool} {layer.pen.name} {layer.pen.hex}" for layer in layers]
    lines.append(f"G0 Z{lift:.3f}")
    for layer in layers:
        lines.append(f"T{layer.tool} M6")
        pen_down_at = None
        for (start_x, start_y), *rest in layer.strokes:
            if (start_x, start_y) != pen_down_at:
                if pen_down_at is not None:
                    lines.append(f"G0 Z{lift:.3f}")
                lines.append(f"G0 X{start_x:.3f} Y{start_y:.3f}")
                lines.append(f"G1 Z0.000 F{PLUNGE_FEED_MM_MIN:g}")
            for index, (x, y) in enumerate(rest):
                feed_word = f" F{feed:g}" if index == 0 else ""
                lines.append(f"G1 X{x:.3f} Y{y:.3f}{feed_word}")
            pen_down_at = (x, y) if rest else (start_x, start_y)
        lines.append(f"G0 Z{lift:.3f}")
    lines.append("M2")
    return "\n".join(lines) + "\n"


def _load_shapes(svg_file: Path, pens: list[Pen]) -> list[Shape]:
    """Every visible shape, including those nearest paper white, which still set the layout."""
    document = svg.SVG.parse(str(svg_file))
    shapes = []
    skipped = 0
    for element in document.elements():
        if element.values.get("visibility") == "hidden":
            continue
        if isinstance(element, svg.Shape):
            fill, stroke = _paint(element.fill), _paint(element.stroke)
            if fill is None and stroke is None:
                continue
            path = svg.Path(element)
            path.reify()
            if len(path):
                shapes.append(
                    Shape(
                        path,
                        fill=nearest(fill, pens) if fill else None,
                        stroke=nearest(stroke, pens) if stroke else None,
                        evenodd=element.values.get("fill-rule") == "evenodd",
                    )
                )
        elif isinstance(element, (svg.Text, svg.Image)):
            skipped += 1
    if skipped:
        log.warning(
            "skipped %d text/image elements; convert them to paths to draw them",
            skipped,
        )
    return shapes


def _paint(color: svg.Color | None) -> RGB | None:
    """The colour a fill or stroke leaves on paper white, or None when it leaves none."""
    if color is None or color.value is None or color.alpha == 0:
        return None
    opacity = color.alpha / OPAQUE
    channels = (color.red, color.green, color.blue)
    return tuple(
        round(opacity * channel + (1 - opacity) * paper)
        for channel, paper in zip(channels, PAPER_RGB)
    )


def _bounds(paths: list[svg.Path]) -> Bounds:
    boxes = [
        segment.bbox()
        for path in paths
        for segment in path
        if not isinstance(segment, svg.Move)
    ]
    drawn = [box for box in boxes if box[:2] != box[2:]]
    if not drawn:
        raise ValueError("the SVG has no drawable shapes")
    xs = [x for box in drawn for x in (box[0], box[2])]
    ys = [y for box in drawn for y in (box[1], box[3])]
    return min(xs), min(ys), max(xs), max(ys)


def _fit_scale(bounds: Bounds, paper: tuple[float, float], margin: float) -> float:
    min_x, min_y, max_x, max_y = bounds
    width, height = paper
    usable_width = width - 2 * margin
    usable_height = height - 2 * margin
    if usable_width <= 0 or usable_height <= 0:
        raise ValueError(
            f"a {margin:g} mm margin leaves no room on {width:g}x{height:g} mm paper"
        )
    scales = [
        usable / span
        for usable, span in (
            (usable_width, max_x - min_x),
            (usable_height, max_y - min_y),
        )
        if span > 0
    ]
    if not scales:
        raise ValueError("the drawing has zero size")
    return min(scales)


def _flatten(path: svg.Path, tolerance: float) -> list[Stroke]:
    strokes: list[Stroke] = []
    for segment in path:
        if isinstance(segment, svg.Move):
            strokes.append([_xy(segment.end)])
        elif not strokes:
            raise ValueError("SVG path data must start with a moveto")
        elif isinstance(segment, svg.Linear):
            strokes[-1].append(_xy(segment.end))
        else:
            strokes[-1].extend(_flatten_curve(segment, tolerance))
    return strokes


def _flatten_curve(segment: svg.Curve, tolerance: float) -> list[Point]:
    """Points after the start whose chords stay within tolerance of the curve."""
    points: list[Point] = []

    def subdivide(t0: float, p0: Point, t1: float, p1: Point, depth: int) -> None:
        t_mid = (t0 + t1) / 2
        p_mid = _xy(segment.point(t_mid))
        flat = _distance_to_chord(p_mid, p0, p1) <= tolerance
        if depth >= MAX_SUBDIVISION_DEPTH or (depth >= MIN_SUBDIVISION_DEPTH and flat):
            points.append(p1)
            return
        subdivide(t0, p0, t_mid, p_mid, depth + 1)
        subdivide(t_mid, p_mid, t1, p1, depth + 1)

    subdivide(0.0, _xy(segment.point(0.0)), 1.0, _xy(segment.point(1.0)), 0)
    return points


def _distance_to_chord(point: Point, start: Point, end: Point) -> float:
    chord_x, chord_y = end[0] - start[0], end[1] - start[1]
    length_sq = chord_x * chord_x + chord_y * chord_y
    if length_sq == 0:
        return math.dist(point, start)
    along = (
        (point[0] - start[0]) * chord_x + (point[1] - start[1]) * chord_y
    ) / length_sq
    along = min(max(along, 0.0), 1.0)
    nearest_point = (start[0] + along * chord_x, start[1] + along * chord_y)
    return math.dist(point, nearest_point)


def _to_paper(
    points: Stroke, bounds: Bounds, scale: float, paper: tuple[float, float]
) -> Stroke:
    """SVG points centred on the paper in mm, Y flipped up, repeats dropped."""
    min_x, min_y, max_x, max_y = bounds
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    width, height = paper
    fitted = [
        (
            round(width / 2 + (x - center_x) * scale, 3),
            round(height / 2 - (y - center_y) * scale, 3),
        )
        for x, y in points
    ]
    return [p for i, p in enumerate(fitted) if i == 0 or p != fitted[i - 1]]


def _closed(ring: Stroke) -> Stroke:
    return ring if ring[0] == ring[-1] else [*ring, ring[0]]


def _xy(point: svg.Point) -> Point:
    return float(point.x), float(point.y)
