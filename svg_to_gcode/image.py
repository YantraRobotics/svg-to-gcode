import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np
from contourpy import FillType, LineType, contour_generator
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage
from scipy.spatial import cKDTree

from svg_to_gcode import one_line
from svg_to_gcode.hatch import ring_area
from svg_to_gcode.pens import (
    PAPER_RGB,
    Pen,
    blends,
    coverage_mixes,
    ink_coverage,
    nearest_index,
    overlay,
    to_lab,
)

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
OPAQUE = 255
# A region's outline is where its blurred 0/1 mask crosses one half; the slight
# blur turns the staircase along pixel edges into a smooth curve.
EDGE_BLUR_PX = 1.0
MASK_LEVEL = 0.5
# A border of paper closes the outlines of regions that touch the image edge.
BORDER_PX = 1
PIXEL_CENTER = 0.5
# Labels pass through PIL's mode filter as 8-bit pixels, with 0 kept for paper.
MAX_LABEL = 255
SCREEN_ANGLE_DEG = 45.0
HALF_TURN_DEG = 180.0
SCREEN_SAMPLES_PER_PX = 2
# Floyd-Steinberg shares of the error at a sample: the next sample on the line,
# then the samples behind, beside and ahead of it on the next line.
DIFFUSE_AHEAD = 7 / 16
DIFFUSE_NEXT_LINE = ((-1, 3 / 16), (0, 5 / 16), (1, 1 / 16))
INK_LEVEL = 0.5
# The pen goes down once the ink owed rises PEN_DOWN_ABOVE past INK_LEVEL and
# lifts once it falls LIFT_BELOW under it. Lifting late makes a line overdraw and
# the lines beside it leave the matching gap, so tones come out as long strokes
# spaced apart rather than short dashes, while going down early still inks
# features too narrow for ink to pile up in. PEN_DOWN_ABOVE stays under
# 1 - INK_LEVEL so that full coverage puts the pen down at once.
PEN_DOWN_ABOVE = 0.4
LIFT_BELOW = 3.0
# Below this coverage a pen is not wanted: its strokes stop there instead of
# running on into other colours, and the ink owed there is dropped rather than
# piling up against the next edge.
UNWANTED_COVERAGE = 0.05
# Shorter dashes add specks rather than tone, so the pen stays down at least this long.
MIN_DASH_PX = 2.5
# Softens fine texture so lines break into fewer, longer dashes.
TONE_BLUR_PX = 0.7
# A fixed seed makes the same image always give the same one-line drawing.
ONE_LINE_SEED = 0
# One-line drawings sample tone this many times finer than the pen's width, so the
# line can follow edges finer than itself.
ONE_LINE_DETAIL = 2
# A skeleton branch no longer than this many times the stroke's half-width where it
# joins is a bump on the stroke's edge, so it is pruned. Corners of a stroke, the
# bars of small letters and arrowheads branch further out, and stay.
SPUR_RATIO = 1.0
PRUNE_ROUNDS = 2
# A junction keeps this many of its branches even when all are short, so a small
# glyph thins to a line rather than a dot.
KEPT_BRANCHES = 2
# Centre lines are averaged over this many pixels either side of each point, which
# irons out the staircase a one-pixel line takes across the grid.
CENTERLINE_SMOOTHING_PX = 2
# A lone skeleton pixel, the middle of a dot, is drawn as a stroke this long.
DOT_PX = 0.5
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)
# Night scenes. Glossy paint at night breaks up into reflections and highlights
# pixel by pixel, so colour is judged over NIGHT_COLOUR_BLUR of the image's long
# side: an area that saturated (CIE Lab chroma) takes the colour pen nearest its
# hue, if one is within NIGHT_MAX_HUE_DEG, unless the pixel is lighter than
# NIGHT_MAX_COLOUR_LIGHTNESS. Pens less saturated than NIGHT_MIN_PEN_CHROMA are
# not colour pens. The other pens draw lines where the lightness, blurred by
# NIGHT_LINE_BLUR_PX, crosses NIGHT_LINE_LEVELS (L*), darker pens darker levels.
NIGHT_COLOUR_BLUR = 0.006
NIGHT_MIN_CHROMA = 18.0
NIGHT_MIN_PEN_CHROMA = 20.0
NIGHT_MAX_HUE_DEG = 50.0
NIGHT_MAX_COLOUR_LIGHTNESS = 60.0
NIGHT_LINE_LEVELS = (8.0, 25.0, 55.0)
NIGHT_LINE_BLUR_PX = 1.0
# Photo fills stretch the lightness between these percentiles of the image to
# the full range and scale the colour by PHOTO_CHROMA_BOOST: pens are less
# varied than a photo's darks, so dim paint would otherwise go to the darkest pen.
PHOTO_LIGHTNESS_PERCENTILES = (1.0, 99.5)
PHOTO_CHROMA_BOOST = 1.5
# Hatches through the origin at the nib's width over these shares of the paper
# nest, so a pen's half shade continues its solid one's lines. Sparser shades
# space lines so far apart at broad nibs that they read as a grid, not a tone.
PHOTO_COVERAGES = (0.0, 0.5, 1.0)
FULL_LIGHTNESS = 100.0
# A pixel's neighbours as (row, column) steps, clockwise from north.
NEIGHBOURS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))

ET.register_namespace("", SVG_NS)
ET.register_namespace("inkscape", INKSCAPE_NS)


def image_to_svg(
    image_file: Path,
    pens: list[Pen],
    resolution: int,
    smoothing: int,
    speckle: float,
    blend: int,
) -> str:
    """An SVG with one Inkscape layer per pen, filling the regions that pen inks.

    The image is shrunk to at most resolution px on its long side, and each pixel
    goes to paper white or to the nearest colour that up to blend pens leave laid
    over each other; a region mixed from several pens is in each of their layers,
    which multiply on screen as the inks do on paper. A smoothing x smoothing mode
    filter votes away stray pixels, and regions or holes under speckle x speckle px
    are dropped.
    """
    groups = blends(pens, blend)
    labels = posterize(
        load_image(image_file, resolution),
        [overlay(group) for group in groups],
        smoothing,
    )
    height, width = labels.shape
    svg = _svg(width, height)
    for pen in pens:
        inked = [label for label, group in enumerate(groups, start=1) if pen in group]
        rings = trace(np.isin(labels, inked), speckle * speckle)
        if rings:
            _add_layer(
                svg,
                pen,
                {
                    "d": path_data(rings),
                    "fill": pen.hex,
                    "fill-rule": "evenodd",
                    "stroke": "none",
                },
            )
    if not len(svg):
        raise ValueError(
            f"{image_file} has nothing to draw: every region is nearest paper white "
            "or smaller than the speckle size"
        )
    return ET.tostring(svg, encoding="unicode")


def image_to_photo_fill(
    image_file: Path, pens: list[Pen], resolution: int, smoothing: int, speckle: float
) -> str:
    """An SVG with one Inkscape layer per pen, shading photos whose tones the pens
    only reach once lifted (lifted_lab) to span the darkest pen to paper white,
    so the darkest areas take that pen alone rather than every pen stacked.

    Each pixel goes to the mix of pens, each covering none or PHOTO_COVERAGES of
    the paper, nearest its lifted colour (coverage_mixes). A region a pen covers
    in part is filled with that share as its fill-opacity, which hatches it that
    much sparser. The image is shrunk to at most resolution px on its long side,
    a smoothing x smoothing mode filter votes away stray pixels in each pen's
    coverage, and regions or holes under speckle x speckle px are dropped.
    """
    darkest_pen = min(to_lab(np.array(pen.rgb))[0] for pen in pens)
    lab = lifted_lab(
        to_lab(np.asarray(load_image(image_file, resolution))), darkest_pen
    )
    mixes, mix_lab = coverage_mixes(pens, PHOTO_COVERAGES)
    _, nearest_mix = cKDTree(mix_lab).query(lab.reshape(-1, 3))
    height, width = lab.shape[:2]
    covered = mixes[nearest_mix].reshape(height, width, len(pens))
    svg = _svg(width, height)
    for index, pen in enumerate(pens):
        levels = np.searchsorted(PHOTO_COVERAGES, covered[..., index]).astype(np.uint8)
        levels = np.asarray(
            Image.fromarray(levels).filter(ImageFilter.ModeFilter(smoothing))
        )
        paths = [
            {
                "d": path_data(rings),
                "fill": pen.hex,
                "fill-opacity": f"{coverage:g}",
                "fill-rule": "evenodd",
                "stroke": "none",
            }
            for level, coverage in enumerate(PHOTO_COVERAGES)
            if coverage > 0 and (rings := trace(levels == level, speckle * speckle))
        ]
        if paths:
            _add_layer(svg, pen, *paths)
    if not len(svg):
        raise ValueError(
            f"{image_file} has nothing to draw: every region is nearest paper white "
            "or smaller than the speckle size"
        )
    return ET.tostring(svg, encoding="unicode")


def lifted_lab(lab: np.ndarray, floor: float) -> np.ndarray:
    """Colours (CIE L*a*b*) with the lightness stretched from floor to paper white
    and the colour boosted, as PHOTO_* describe."""
    lightness = lab[..., 0]
    darkest, lightest = np.percentile(lightness, PHOTO_LIGHTNESS_PERCENTILES)
    if lightest <= darkest:
        return lab
    stretched = np.clip((lightness - darkest) / (lightest - darkest), 0.0, 1.0)
    return np.stack(
        [
            floor + (FULL_LIGHTNESS - floor) * stretched,
            lab[..., 1] * PHOTO_CHROMA_BOOST,
            lab[..., 2] * PHOTO_CHROMA_BOOST,
        ],
        axis=-1,
    )


def image_to_line_screen(image_file: Path, pens: list[Pen], resolution: int) -> str:
    """An SVG with one Inkscape layer per pen, shading the image with parallel lines.

    The image is shrunk to at most resolution px on its long side and every pen
    draws lines one pixel apart across it, at SCREEN_ANGLE_DEG for the first pen
    and turned 180/len(pens) degrees per pen after it, inking the share of each
    area that its coverage (pens.ink_coverage) asks for. Set resolution to the
    drawing's width over the pen's line width.
    """
    image = load_image(image_file, resolution).filter(
        ImageFilter.GaussianBlur(TONE_BLUR_PX)
    )
    coverage = ink_coverage(np.asarray(image), pens)
    height, width = coverage.shape[:2]
    svg = _svg(width, height)
    _add_paper(svg, width, height)
    angle_step = HALF_TURN_DEG / len(pens)
    for index, pen in enumerate(pens):
        dashes = screen(coverage[..., index], SCREEN_ANGLE_DEG + angle_step * index)
        if dashes:
            _add_layer(
                svg,
                pen,
                {
                    "d": dash_data(dashes),
                    "fill": "none",
                    "stroke": pen.hex,
                    "stroke-width": "1",
                    "stroke-linecap": "round",
                },
            )
    if svg.find(f"{{{SVG_NS}}}g") is None:
        raise ValueError(f"{image_file} has nothing to draw: it is all paper white")
    return ET.tostring(svg, encoding="unicode")


def image_to_one_line(
    image_file: Path, pens: list[Pen], resolution: int, breaks: int
) -> str:
    """An SVG with one Inkscape layer per pen, each pen drawing one smooth line.

    Each pen's line wanders through points scattered by the share of each area
    that its coverage (pens.ink_coverage) asks for, lifting at most breaks times.
    The line is resolution lines wide across the image's long side, so set
    resolution to the drawing's width over the pen's line width; the tone is
    sampled ONE_LINE_DETAIL times finer than that.
    """
    image = load_image(image_file, resolution * ONE_LINE_DETAIL).filter(
        ImageFilter.GaussianBlur(TONE_BLUR_PX)
    )
    coverage = ink_coverage(np.asarray(image), pens)
    coverage = np.where(coverage < UNWANTED_COVERAGE, 0.0, coverage)
    height, width = coverage.shape[:2]
    svg = _svg(width, height)
    _add_paper(svg, width, height)
    rng = np.random.default_rng(ONE_LINE_SEED)
    for index, pen in enumerate(pens):
        paths = one_line.draw(coverage[..., index], breaks, rng, ONE_LINE_DETAIL)
        if paths:
            _add_layer(
                svg,
                pen,
                {
                    "d": one_line.curve_data(paths),
                    "fill": "none",
                    "stroke": pen.hex,
                    "stroke-width": str(ONE_LINE_DETAIL),
                    "stroke-linecap": "round",
                    "stroke-linejoin": "round",
                },
            )
    if svg.find(f"{{{SVG_NS}}}g") is None:
        raise ValueError(f"{image_file} has nothing to draw: it is all paper white")
    return ET.tostring(svg, encoding="unicode")


def image_to_centerline(
    image_file: Path, pens: list[Pen], resolution: int, smoothing: int, speckle: float
) -> str:
    """An SVG with one Inkscape layer per pen, drawing its lines once along their middle.

    For line art and lettering. The image is shrunk to at most resolution px on its
    long side, each pixel goes to paper white or the nearest pen, a smoothing x
    smoothing mode filter votes away stray pixels, and regions or holes under
    speckle x speckle px are dropped. Each pen's regions are then thinned to their
    centre lines, so the pen draws a line where the image has one instead of
    outlining it.
    """
    labels = posterize(load_image(image_file, resolution), pens, smoothing)
    height, width = labels.shape
    svg = _svg(width, height)
    _add_paper(svg, width, height)
    for index, pen in enumerate(pens, start=1):
        lines = centerlines(despeckle(labels == index, speckle * speckle))
        if lines:
            _add_layer(
                svg,
                pen,
                {
                    "d": polyline_data(lines),
                    "fill": "none",
                    "stroke": pen.hex,
                    "stroke-width": "2",
                    "stroke-linecap": "round",
                    "stroke-linejoin": "round",
                },
            )
    if svg.find(f"{{{SVG_NS}}}g") is None:
        raise ValueError(f"{image_file} has nothing to draw: it is all paper white")
    return ET.tostring(svg, encoding="unicode")


def image_to_night(
    image_file: Path, pens: list[Pen], resolution: int, smoothing: int, speckle: float
) -> str:
    """An SVG for lit subjects on a dark background, such as a stage at night.

    The dark is left as paper. Areas whose colour cast is saturated are filled by
    the colour pen nearest their hue, and the unsaturated pens draw contour lines
    where the lightness crosses NIGHT_LINE_LEVELS, outlining what is lit and the
    dark details inside it: the darkest pen the darkest level, a lighter one the
    lighter levels. The image is shrunk to at most resolution px on its
    long side, a smoothing x smoothing mode filter votes away stray pixels, and
    colour regions or holes under speckle x speckle px and lines shorter than
    speckle px are dropped.
    """
    neutrals, colours = _night_pens(pens)
    lab = to_lab(np.asarray(load_image(image_file, resolution)))
    labels = night_colours(lab, colours, smoothing)
    min_area = speckle * speckle
    height, width = labels.shape
    svg = _svg(width, height)
    _add_paper(svg, width, height)
    for index, pen in enumerate(colours, start=1):
        rings = trace(despeckle(labels == index, min_area), min_area)
        if rings:
            _add_layer(
                svg,
                pen,
                {
                    "d": path_data(rings),
                    "fill": pen.hex,
                    "fill-rule": "evenodd",
                    "stroke": "none",
                },
            )
    for pen, levels in _line_levels(neutrals).items():
        lines = lightness_lines(lab[..., 0], levels, speckle)
        if not lines:
            continue
        _add_layer(
            svg,
            pen,
            {
                "d": polyline_data(lines),
                "fill": "none",
                "stroke": pen.hex,
                "stroke-width": "1",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
            },
        )
    if svg.find(f"{{{SVG_NS}}}g") is None:
        raise ValueError(f"{image_file} has nothing to draw: it is all one tone")
    return ET.tostring(svg, encoding="unicode")


def night_colours(lab: np.ndarray, colours: list[Pen], smoothing: int) -> np.ndarray:
    """Per pixel, 0 for no colour or 1 + the index of the colour pen that inks it."""
    labels = np.zeros(lab.shape[:2], dtype=np.uint8)
    if not colours:
        return labels
    sigma = NIGHT_COLOUR_BLUR * max(labels.shape)
    a = ndimage.gaussian_filter(lab[..., 1], sigma)
    b = ndimage.gaussian_filter(lab[..., 2], sigma)
    pen_hues = np.array([_hue(pen) for pen in colours])
    turns = np.abs(
        np.angle(np.exp(1j * (np.arctan2(b, a)[..., np.newaxis] - pen_hues)))
    )
    inked = (
        (np.hypot(a, b) >= NIGHT_MIN_CHROMA)
        & (turns.min(axis=-1) <= math.radians(NIGHT_MAX_HUE_DEG))
        & (lab[..., 0] <= NIGHT_MAX_COLOUR_LIGHTNESS)
    )
    labels[inked] = np.argmin(turns, axis=-1)[inked] + 1
    return np.asarray(Image.fromarray(labels).filter(ImageFilter.ModeFilter(smoothing)))


def lightness_lines(
    lightness: np.ndarray, levels: Sequence[float], min_length: float
) -> list[np.ndarray]:
    """Lines (x, y in px) where the blurred lightness crosses each level (L*)."""
    blurred = ndimage.gaussian_filter(lightness, NIGHT_LINE_BLUR_PX)
    generator = contour_generator(z=blurred, line_type=LineType.Separate)
    lines = []
    for level in levels:
        for line in generator.lines(level):
            length = np.sum(np.hypot(*np.diff(line, axis=0).T))
            if len(line) > 1 and length >= min_length:
                lines.append(line + PIXEL_CENTER)
    return lines


def _night_pens(pens: list[Pen]) -> tuple[list[Pen], list[Pen]]:
    """The pens that draw lines, darkest first, and the saturated pens that fill colour.

    The darkest pen always draws lines, however saturated.
    """
    if not pens:
        raise ValueError("a night scene needs at least a dark pen")
    darkest, *others = sorted(pens, key=lambda pen: to_lab(np.array(pen.rgb))[0])
    colours = [pen for pen in others if _chroma(pen) >= NIGHT_MIN_PEN_CHROMA]
    if len(colours) > MAX_LABEL:
        raise ValueError(f"a night scene takes at most {MAX_LABEL} colour pens")
    return [darkest, *(pen for pen in others if pen not in colours)], colours


def _line_levels(neutrals: list[Pen]) -> dict[Pen, list[float]]:
    """NIGHT_LINE_LEVELS, darkest first, shared out one each to the pens darkest
    first, the lightest pen taking any left over."""
    shares: dict[Pen, list[float]] = {}
    for index, level in enumerate(sorted(NIGHT_LINE_LEVELS)):
        shares.setdefault(neutrals[min(index, len(neutrals) - 1)], []).append(level)
    return shares


def _chroma(pen: Pen) -> float:
    _, a, b = to_lab(np.array(pen.rgb))
    return math.hypot(a, b)


def _hue(pen: Pen) -> float:
    _, a, b = to_lab(np.array(pen.rgb))
    return math.atan2(b, a)


def despeckle(mask: np.ndarray, min_area: float) -> np.ndarray:
    """The mask without regions, or holes in them, smaller than min_area px."""
    regions, _ = ndimage.label(mask, structure=EIGHT_CONNECTED)
    kept = np.bincount(regions.ravel()) >= min_area
    kept[0] = False
    mask = kept[regions]
    holes, _ = ndimage.label(~mask)
    small = np.bincount(holes.ravel()) < min_area
    small[0] = False
    small[np.unique([holes[0], holes[-1]])] = False
    small[np.unique([holes[:, 0], holes[:, -1]])] = False
    return mask | small[holes]


def thin(mask: np.ndarray) -> np.ndarray:
    """The mask thinned to lines one pixel wide along its middle (Guo and Hall, 1989)."""
    image = np.pad(mask, 1).astype(np.uint8)
    height, width = image.shape
    centre = image[1:-1, 1:-1]
    while True:
        thinned = False
        for first_pass in (True, False):
            p2, p3, p4, p5, p6, p7, p8, p9 = (
                image[1 + row : height - 1 + row, 1 + column : width - 1 + column]
                for row, column in NEIGHBOURS
            )
            crossings = (
                ((p2 ^ 1) & (p3 | p4))
                + ((p4 ^ 1) & (p5 | p6))
                + ((p6 ^ 1) & (p7 | p8))
                + ((p8 ^ 1) & (p9 | p2))
            )
            occupied = np.minimum(
                (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8),
                (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9),
            )
            if first_pass:
                facing = (p6 | p7 | (p9 ^ 1)) & p8
            else:
                facing = (p2 | p3 | (p5 ^ 1)) & p4
            removable = (
                (centre == 1)
                & (crossings == 1)
                & (occupied >= 2)
                & (occupied <= 3)
                & (facing == 0)
            )
            if removable.any():
                centre[removable] = 0
                thinned = True
        if not thinned:
            return centre.astype(bool)


def centerlines(mask: np.ndarray) -> list[np.ndarray]:
    """The middle lines of a mask's regions as smoothed (x, y) points in px.

    Branches that only reach into a corner of a stroke are pruned. Lines that meet
    share the point where they meet, so a pen can run on from one into the next.
    """
    if not mask.any():
        return []
    half_widths = ndimage.distance_transform_edt(mask)
    skeleton = thin(mask)
    for _ in range(PRUNE_ROUNDS):
        rows, columns, neighbours = _skeleton_graph(skeleton)
        spurs = _spurs(_trace(neighbours), neighbours, half_widths[rows, columns])
        for path in spurs:
            skeleton[rows[path[:-1]], columns[path[:-1]]] = False
        if not spurs:
            break
    rows, columns, neighbours = _skeleton_graph(skeleton)
    lines = []
    for path in _trace(neighbours):
        points = np.column_stack([columns[path], rows[path]]) + PIXEL_CENTER
        if len(path) == 1:
            points = np.vstack([points, points + (DOT_PX, 0.0)])
        loop = len(path) > 2 and path[0] == path[-1] and len(neighbours[path[0]]) == 2
        lines.append(_smooth(points, loop))
    return lines


def polyline_data(lines: list[np.ndarray]) -> str:
    """SVG path data with one subpath of straight segments per line."""
    return " ".join(
        "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in line) for line in lines
    )


def screen(field: np.ndarray, angle_deg: float) -> list[np.ndarray]:
    """Dashes, as 2 x 2 arrays of start and end in pixels, of lines across a field.

    The lines run at angle_deg one pixel apart through pixel centres and are down
    for the share of their length that the field asks for: error diffusion
    carries what each sample over- or under-inks on to the samples after it and
    on the next line. Successive lines run forwards and backwards, so the pen
    sweeps to and fro.
    """
    height, width = field.shape
    angle = math.radians(angle_deg)
    along = np.array([math.cos(angle), math.sin(angle)])
    across = np.array([-math.sin(angle), math.cos(angle)])
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=float)
    offsets, reaches = corners @ across - PIXEL_CENTER, corners @ along
    sample_count = math.floor(np.ptp(reaches) * SCREEN_SAMPLES_PER_PX) + 1
    steps = reaches.min() + np.arange(sample_count) / SCREEN_SAMPLES_PER_PX
    min_samples = math.ceil(MIN_DASH_PX * SCREEN_SAMPLES_PER_PX)
    owed = np.zeros(sample_count)
    dashes = []
    forwards = True
    for line in range(math.ceil(offsets.min()), math.floor(offsets.max()) + 1):
        points = (line + PIXEL_CENTER) * across + steps[:, np.newaxis] * along
        x, y = points[:, 0], points[:, 1]
        inside = (x >= 0) & (x <= width) & (y >= 0) & (y <= height)
        coverage = np.where(inside, sample(field, x, y), 0.0)
        down, owed = _diffuse_line(coverage, owed, inside, forwards, min_samples)
        edges = np.flatnonzero(np.diff(down.astype(np.int8), prepend=0, append=0))
        found = [
            points[[start, end - 1]]
            for start, end in zip(edges[::2], edges[1::2])
            if end - start >= min_samples
        ]
        if not forwards:
            found = [dash[::-1] for dash in reversed(found)]
        dashes += found
        forwards = not forwards
    return dashes


def sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear field values at points in pixels, with pixel centres on half pixels."""
    height, width = field.shape
    column = np.clip(x - PIXEL_CENTER, 0, width - 1)
    row = np.clip(y - PIXEL_CENTER, 0, height - 1)
    left, top = np.floor(column).astype(int), np.floor(row).astype(int)
    right, bottom = np.minimum(left + 1, width - 1), np.minimum(top + 1, height - 1)
    across, down = column - left, row - top
    upper = field[top, left] * (1 - across) + field[top, right] * across
    lower = field[bottom, left] * (1 - across) + field[bottom, right] * across
    return upper * (1 - down) + lower * down


def dash_data(dashes: list[np.ndarray]) -> str:
    """SVG path data with one straight subpath per dash."""
    return " ".join(f"M{a:.2f},{b:.2f} L{c:.2f},{d:.2f}" for (a, b), (c, d) in dashes)


def load_image(image_file: Path, resolution: int) -> Image.Image:
    """The image upright, flattened onto paper white, at most resolution px on its long side."""
    with Image.open(image_file) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGBA")
    paper = Image.new("RGBA", image.size, (*PAPER_RGB, OPAQUE))
    flattened = Image.alpha_composite(paper, image).convert("RGB")
    flattened.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
    return flattened


def posterize(image: Image.Image, pens: list[Pen], smoothing: int) -> np.ndarray:
    """Per pixel, 0 for paper or 1 + the index of the nearest pen."""
    if len(pens) > MAX_LABEL:
        raise ValueError(
            f"{len(pens)} pen blends are too many to tell apart (at most {MAX_LABEL}); "
            "use fewer pens or a smaller --blend"
        )
    labels = nearest_index(np.asarray(image), pens).astype(np.uint8)
    voted = Image.fromarray(labels).filter(ImageFilter.ModeFilter(smoothing))
    return np.asarray(voted)


def trace(mask: np.ndarray, min_area: float) -> list[np.ndarray]:
    """Closed outlines of a mask's regions and their holes, in pixels, first point repeated last.

    Outer outlines wind one way and holes the other, so either fill rule fills them.
    """
    if not mask.any():
        return []
    bordered = np.pad(mask, BORDER_PX).astype(np.uint8) * OPAQUE
    blurred = Image.fromarray(bordered).filter(ImageFilter.GaussianBlur(EDGE_BLUR_PX))
    levels = np.asarray(blurred, dtype=float) / OPAQUE
    generator = contour_generator(z=levels, fill_type=FillType.OuterOffset)
    points, offsets = generator.filled(MASK_LEVEL, np.inf)
    rings = []
    for polygon, bounds in zip(points, offsets):
        outer, *holes = (polygon[a:b] for a, b in pairwise(bounds))
        if ring_area(outer) >= min_area:
            rings.append(outer)
            rings.extend(hole for hole in holes if ring_area(hole) >= min_area)
    return [ring - BORDER_PX + PIXEL_CENTER for ring in rings]


def path_data(rings: list[np.ndarray]) -> str:
    """SVG path data with one closed subpath per ring."""
    return " ".join(
        "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in ring[:-1]) + " Z"
        for ring in rings
    )


def _skeleton_graph(
    skeleton: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Rows, columns and neighbour lists of a skeleton's pixels, joined by m-adjacency.

    Diagonal neighbours join only where no shared four-neighbour already links
    them, so the steps of a staircase read as one path instead of small triangles.
    """
    rows, columns = np.nonzero(skeleton)
    padded = np.pad(skeleton, 1)
    index = np.full(padded.shape, -1)
    index[rows + 1, columns + 1] = np.arange(len(rows))
    near = {
        step: padded[rows + 1 + step[0], columns + 1 + step[1]] for step in NEIGHBOURS
    }
    north, east, south, west = near[-1, 0], near[0, 1], near[1, 0], near[0, -1]
    joined = {
        (-1, 0): north,
        (0, 1): east,
        (1, 0): south,
        (0, -1): west,
        (-1, 1): near[-1, 1] & ~north & ~east,
        (1, 1): near[1, 1] & ~south & ~east,
        (1, -1): near[1, -1] & ~south & ~west,
        (-1, -1): near[-1, -1] & ~north & ~west,
    }
    neighbours: list[list[int]] = [[] for _ in range(len(rows))]
    for (row, column), linked in joined.items():
        sources = np.flatnonzero(linked)
        targets = index[rows[sources] + 1 + row, columns[sources] + 1 + column]
        for source, target in zip(sources.tolist(), targets.tolist()):
            neighbours[source].append(target)
    return rows, columns, neighbours


def _trace(neighbours: list[list[int]]) -> list[list[int]]:
    """Paths of pixel indices between a graph's ends and junctions, then its loops.

    A path runs on through pixels with exactly two neighbours. A lone pixel is a
    path of its own, and a loop without a junction starts and ends on one pixel.
    """
    walked: set[tuple[int, int]] = set()

    def walk(start: int, step: int) -> list[int]:
        path = [start, step]
        walked.add((min(start, step), max(start, step)))
        previous, current = start, step
        while len(neighbours[current]) == 2:
            first, second = neighbours[current]
            following = second if first == previous else first
            edge = (min(current, following), max(current, following))
            if edge in walked:
                break
            walked.add(edge)
            path.append(following)
            previous, current = current, following
        return path

    paths = []
    for pixel, linked in enumerate(neighbours):
        if len(linked) == 2:
            continue
        if not linked:
            paths.append([pixel])
        for step in linked:
            if (min(pixel, step), max(pixel, step)) not in walked:
                paths.append(walk(pixel, step))
    covered = {pixel for path in paths for pixel in path}
    for pixel, linked in enumerate(neighbours):
        if len(linked) == 2 and pixel not in covered:
            loop = walk(pixel, linked[0])
            covered.update(loop)
            paths.append(loop)
    return paths


def _spurs(
    paths: list[list[int]], neighbours: list[list[int]], widths: np.ndarray
) -> list[list[int]]:
    """Paths from a free end to a junction within SPUR_RATIO half-widths of it, free
    end first, leaving every junction at least KEPT_BRANCHES branches."""
    short: dict[int, list[list[int]]] = {}
    for path in paths:
        if len(neighbours[path[0]]) != 1:
            path = path[::-1]
        junction = path[-1]
        if (
            len(neighbours[path[0]]) == 1
            and len(neighbours[junction]) >= 3
            and len(path) - 1 <= SPUR_RATIO * widths[junction]
        ):
            short.setdefault(junction, []).append(path)
    spurs = []
    for junction, branches in short.items():
        kept = KEPT_BRANCHES if len(neighbours[junction]) == len(branches) else 0
        spurs += sorted(branches, key=len)[: max(0, len(branches) - kept)]
    return spurs


def _smooth(points: np.ndarray, loop: bool) -> np.ndarray:
    """Points averaged with CENTERLINE_SMOOTHING_PX neighbours either side along the
    line; an open line keeps its end points, where other lines may meet it."""
    reach = CENTERLINE_SMOOTHING_PX
    span = 2 * reach + 1
    if loop:
        ring = points[:-1]
        shifts = range(-reach, reach + 1)
        averaged = sum(np.roll(ring, shift, axis=0) for shift in shifts) / span
        return np.vstack([averaged, averaged[:1]])
    padded = np.concatenate(
        [
            np.repeat(points[:1], reach, axis=0),
            points,
            np.repeat(points[-1:], reach, axis=0),
        ]
    )
    averaged = sum(padded[shift : shift + len(points)] for shift in range(span)) / span
    averaged[[0, -1]] = points[[0, -1]]
    return averaged


def _diffuse_line(
    coverage: np.ndarray,
    owed: np.ndarray,
    inside: np.ndarray,
    forwards: bool,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Which samples of a line are inked, and the error owed to the next line.

    Once down, the pen stays down for at least min_samples where it is wanted.
    """
    count = len(coverage)
    wanted = coverage + owed
    down = np.zeros(count, dtype=bool)
    owed = np.zeros(count)
    step = 1 if forwards else -1
    run = 0
    for index in range(count)[::step]:
        if not inside[index] or coverage[index] < UNWANTED_COVERAGE:
            run = 0
            continue
        if 0 < run < min_samples:
            inked = True
        else:
            threshold = INK_LEVEL - LIFT_BELOW if run else INK_LEVEL + PEN_DOWN_ABOVE
            inked = bool(wanted[index] > threshold)
        run = run + 1 if inked else 0
        down[index] = inked
        error = wanted[index] - inked
        if 0 <= index + step < count:
            wanted[index + step] += error * DIFFUSE_AHEAD
        for shift, share in DIFFUSE_NEXT_LINE:
            neighbour = index + shift * step
            if 0 <= neighbour < count:
                owed[neighbour] += error * share
    return down, owed


def _add_paper(svg: ET.Element, width: int, height: int) -> None:
    """A blank sheet behind the lines, which fixes the layout to the whole image so
    lines keep their pen width however far the ink reaches."""
    ET.SubElement(
        svg,
        f"{{{SVG_NS}}}rect",
        width=str(width),
        height=str(height),
        fill="white",
        stroke="none",
    )


def _svg(width: int, height: int) -> ET.Element:
    return ET.Element(
        f"{{{SVG_NS}}}svg",
        width=str(width),
        height=str(height),
        viewBox=f"0 0 {width} {height}",
    )


def _add_layer(svg: ET.Element, pen: Pen, *paths: dict[str, str]) -> None:
    """A pen's Inkscape layer holding the paths, multiplying over the layers below."""
    layer = ET.SubElement(
        svg,
        f"{{{SVG_NS}}}g",
        {
            f"{{{INKSCAPE_NS}}}groupmode": "layer",
            f"{{{INKSCAPE_NS}}}label": pen.name,
            "style": "mix-blend-mode:multiply",
        },
    )
    for path in paths:
        ET.SubElement(layer, f"{{{SVG_NS}}}path", path)
