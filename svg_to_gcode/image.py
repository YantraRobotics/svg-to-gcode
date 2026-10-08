import math
import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path

import numpy as np
from contourpy import FillType, contour_generator
from PIL import Image, ImageFilter, ImageOps

from svg_to_gcode import one_line
from svg_to_gcode.hatch import ring_area
from svg_to_gcode.pens import (
    PAPER_RGB,
    Pen,
    blends,
    ink_coverage,
    nearest_index,
    overlay,
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


def _add_layer(svg: ET.Element, pen: Pen, path: dict[str, str]) -> None:
    """A pen's Inkscape layer holding one path, multiplying over the layers below."""
    layer = ET.SubElement(
        svg,
        f"{{{SVG_NS}}}g",
        {
            f"{{{INKSCAPE_NS}}}groupmode": "layer",
            f"{{{INKSCAPE_NS}}}label": pen.name,
            "style": "mix-blend-mode:multiply",
        },
    )
    ET.SubElement(layer, f"{{{SVG_NS}}}path", path)
