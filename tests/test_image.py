import math
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image, ImageDraw

from svg_to_gcode import hatch, image, pens

CMYK = pens.parse_pens(pens.DEFAULT_PENS)
CARD_PX = 200
SQUARE_PX = 60
SQUARE_CORNERS = [(20, 20), (120, 20), (20, 120), (120, 120)]
RESOLUTION = 400
SMOOTHING = 5
SPECKLE = 8.0
BLEND = 2
YELLOW, CYAN, MAGENTA, BLACK = CMYK
LABEL = f"{{{image.INKSCAPE_NS}}}label"


def four_colour_card(background: tuple[int, ...]) -> Image.Image:
    card = Image.new("RGBA", (CARD_PX, CARD_PX), background)
    draw = ImageDraw.Draw(card)
    for (x, y), pen in zip(SQUARE_CORNERS, CMYK):
        draw.rectangle([x, y, x + SQUARE_PX - 1, y + SQUARE_PX - 1], fill=pen.rgb)
    return card


def layers(svg: str) -> dict[str, list[np.ndarray]]:
    """Pen name to the rings of its layer's path."""
    root = ET.fromstring(svg)
    found = {}
    for layer in root.iter(f"{{{image.SVG_NS}}}g"):
        path = layer.find(f"{{{image.SVG_NS}}}path")
        found[layer.get(LABEL)] = [
            np.array(re.findall(r"(-?[\d.]+),(-?[\d.]+)", subpath), dtype=float)
            for subpath in path.get("d").split("Z")
            if subpath.strip()
        ]
    return found


def convert(tmp_path, picture: Image.Image, **settings) -> str:
    image_file = tmp_path / "picture.png"
    picture.save(image_file)
    options = {
        "resolution": RESOLUTION,
        "smoothing": SMOOTHING,
        "speckle": SPECKLE,
        "blend": BLEND,
    }
    return image.image_to_svg(image_file, CMYK, **(options | settings))


@pytest.mark.parametrize("background", [(255, 255, 255, 255), (0, 0, 0, 0)])
def test_four_colour_card_gives_a_layer_per_pen(tmp_path, background):
    found = layers(convert(tmp_path, four_colour_card(background)))

    assert list(found) == ["yellow", "cyan", "magenta", "black"]
    for (x, y), rings in zip(SQUARE_CORNERS, found.values()):
        (ring,) = rings
        assert hatch.ring_area(ring) == pytest.approx(SQUARE_PX**2, rel=0.01)
        assert ring.min(axis=0) == pytest.approx((x, y), abs=0.01)
        assert ring.max(axis=0) == pytest.approx(
            (x + SQUARE_PX, y + SQUARE_PX), abs=0.01
        )


def mixed_card() -> Image.Image:
    """Orange that yellow and magenta mix on the left, green from yellow and cyan on the right."""
    card = Image.new("RGB", (CARD_PX, CARD_PX), "white")
    draw = ImageDraw.Draw(card)
    for (x, y), group in zip(SQUARE_CORNERS, [(YELLOW, MAGENTA), (YELLOW, CYAN)]):
        draw.rectangle(
            [x, y, x + SQUARE_PX - 1, y + SQUARE_PX - 1], fill=pens.overlay(group).rgb
        )
    return card


def test_mixed_colours_go_to_every_pen_in_the_mix(tmp_path):
    svg = convert(tmp_path, mixed_card())
    found = layers(svg)

    assert list(found) == ["yellow", "cyan", "magenta"]
    orange, green = (pytest.approx(corner, abs=0.01) for corner in SQUARE_CORNERS[:2])
    corners = {
        pen: sorted(ring.min(axis=0).tolist() for ring in rings)
        for pen, rings in found.items()
    }
    assert corners == {"yellow": [orange, green], "cyan": [green], "magenta": [orange]}
    assert svg.count("mix-blend-mode:multiply") == len(found)


def test_without_blending_each_colour_takes_one_pen(tmp_path):
    found = layers(convert(tmp_path, mixed_card(), blend=1))

    assert sum(len(rings) for rings in found.values()) == 2


def test_ring_keeps_its_hole(tmp_path):
    picture = Image.new("RGB", (CARD_PX, CARD_PX), "white")
    draw = ImageDraw.Draw(picture)
    draw.ellipse([40, 40, 160, 160], fill="black")
    draw.ellipse([70, 70, 130, 130], fill="white")
    svg = convert(tmp_path, picture)

    outer, hole = layers(svg)["black"]
    assert hatch.ring_area(outer) == pytest.approx(math.pi * 60.5**2, rel=0.01)
    assert hatch.ring_area(hole) == pytest.approx(math.pi * 30.5**2, rel=0.02)
    assert 'fill-rule="evenodd"' in svg
    through_centre = hatch.hatch([outer, hole], 100.0, 0.0, evenodd=True)
    assert [[x for x, _ in stroke] for stroke in through_centre] == [
        pytest.approx([40.0, 70.0], abs=0.6),
        pytest.approx([131.0, 161.0], abs=0.6),
    ]


@pytest.mark.parametrize("smoothing", [1, SMOOTHING])
def test_specks_are_dropped(tmp_path, smoothing):
    picture = Image.new("RGB", (CARD_PX, CARD_PX), "white")
    draw = ImageDraw.Draw(picture)
    draw.rectangle([100, 100, 139, 139], fill="black")
    draw.rectangle([110, 110, 113, 113], fill="white")
    for corner in (10, 40, 70):
        draw.rectangle([corner, corner, corner + 3, corner + 3], fill="black")

    (square,) = layers(convert(tmp_path, picture, smoothing=smoothing))["black"]
    assert hatch.ring_area(square) == pytest.approx(40**2, rel=0.02)


def test_large_images_shrink_to_the_resolution(tmp_path):
    picture = four_colour_card((255, 255, 255, 255)).resize((1000, 1000))
    root = ET.fromstring(convert(tmp_path, picture, resolution=100))
    assert (root.get("width"), root.get("height")) == ("100", "100")


def test_blank_image_has_nothing_to_draw(tmp_path):
    with pytest.raises(ValueError, match="nothing to draw"):
        convert(tmp_path, Image.new("RGB", (CARD_PX, CARD_PX), "white"))


def screen_dashes(svg: str) -> dict[str, np.ndarray]:
    """Pen name to its layer's dashes, shape (dashes, 2 ends, 2)."""
    root = ET.fromstring(svg)
    return {
        layer.get(LABEL): np.array(
            re.findall(
                r"(-?[\d.]+),(-?[\d.]+)", layer.find(f"{{{image.SVG_NS}}}path").get("d")
            ),
            dtype=float,
        ).reshape(-1, 2, 2)
        for layer in root.iter(f"{{{image.SVG_NS}}}g")
    }


def line_screen(tmp_path, picture: Image.Image) -> str:
    image_file = tmp_path / "picture.png"
    picture.save(image_file)
    return image.image_to_line_screen(image_file, CMYK, CARD_PX)


FIELD_PX = 120


@pytest.mark.parametrize("share", [0.1, 0.5, 0.8])
def test_screen_inks_the_share_the_field_asks_for(share):
    dashes = image.screen(np.full((FIELD_PX, FIELD_PX), share), 30.0)

    lengths = np.array([math.dist(*dash) for dash in dashes])
    sample_px = 1 / image.SCREEN_SAMPLES_PER_PX
    inked_px = np.sum(lengths + sample_px)
    assert inked_px == pytest.approx(share * FIELD_PX**2, rel=0.05)
    assert round(lengths.min() + sample_px, 6) >= image.MIN_DASH_PX


def test_screen_draws_every_line_of_a_full_field_to_and_fro():
    dashes = image.screen(np.ones((FIELD_PX, FIELD_PX)), 0.0)

    assert [dash[0, 1] for dash in dashes] == [row + 0.5 for row in range(FIELD_PX)]
    assert [dash[:, 0].tolist() for dash in dashes[:2]] == [
        [0, FIELD_PX],
        [FIELD_PX, 0],
    ]
    assert image.screen(np.zeros((FIELD_PX, FIELD_PX)), 45.0) == []


LONG_STROKE_PX = 15


def test_screen_draws_textured_tones_as_long_strokes():
    texture = np.random.default_rng(0).uniform(0.3, 0.7, (FIELD_PX, FIELD_PX))

    dashes = image.screen(texture, 45.0)

    lengths = np.array([math.dist(*dash) for dash in dashes])
    sample_px = 1 / image.SCREEN_SAMPLES_PER_PX
    assert np.sum(lengths + sample_px) == pytest.approx(texture.sum(), rel=0.05)
    assert np.median(lengths) >= LONG_STROKE_PX


def test_screen_stops_strokes_where_the_pen_is_not_wanted():
    band = np.zeros((FIELD_PX, FIELD_PX))
    left, right = FIELD_PX // 3, 2 * FIELD_PX // 3
    band[:, left:right] = 0.5

    dashes = np.array(image.screen(band, 45.0))

    assert dashes[..., 0].min() >= left - image.PIXEL_CENTER
    assert dashes[..., 0].max() <= right + image.PIXEL_CENTER


def test_line_screen_shades_only_where_a_pen_is_needed(tmp_path):
    picture = Image.new("RGB", (CARD_PX, CARD_PX), "white")
    half = CARD_PX // 2
    ImageDraw.Draw(picture).rectangle([0, 0, half - 1, CARD_PX - 1], fill=YELLOW.rgb)

    svg = line_screen(tmp_path, picture)

    found = screen_dashes(svg)
    assert list(found) == ["yellow"]
    dashes = found["yellow"]
    assert dashes[..., 0].max() < half + 2
    length = np.linalg.norm(dashes[:, 1] - dashes[:, 0], axis=1).sum()
    assert length == pytest.approx(half * CARD_PX, rel=0.03)
    assert f'<rect width="{CARD_PX}" height="{CARD_PX}" fill="white"' in svg
    assert svg.count("mix-blend-mode:multiply") == 1


def test_blank_image_has_no_tone_to_draw(tmp_path):
    with pytest.raises(ValueError, match="nothing to draw"):
        line_screen(tmp_path, Image.new("RGB", (CARD_PX, CARD_PX), "white"))


ONE_LINE_RESOLUTION = 50


def one_line_drawing(
    tmp_path, picture: Image.Image, breaks: int
) -> dict[str, list[np.ndarray]]:
    """Pen name to the points of each subpath of its layer, in px of the drawing."""
    image_file = tmp_path / "picture.png"
    picture.save(image_file)
    svg = image.image_to_one_line(image_file, CMYK, ONE_LINE_RESOLUTION, breaks)
    return {
        layer.get(LABEL): [
            np.array(re.findall(r"(-?[\d.]+),(-?[\d.]+)", subpath), dtype=float)
            for subpath in layer.find(f"{{{image.SVG_NS}}}path").get("d").split("M")
            if subpath.strip()
        ]
        for layer in ET.fromstring(svg).iter(f"{{{image.SVG_NS}}}g")
    }


@pytest.mark.parametrize(("breaks", "strokes"), [(0, 1), (3, 2)])
def test_one_line_lifts_only_to_jump_between_patches(tmp_path, breaks, strokes):
    picture = Image.new("RGB", (CARD_PX, CARD_PX), "white")
    for x, y in SQUARE_CORNERS[:2]:
        ImageDraw.Draw(picture).rectangle(
            [x, y, x + SQUARE_PX - 1, y + SQUARE_PX - 1], fill=YELLOW.rgb
        )

    found = one_line_drawing(tmp_path, picture, breaks)

    assert list(found) == ["yellow"]
    assert len(found["yellow"]) == strokes
    if strokes == 2:
        gap_middle = ONE_LINE_RESOLUTION * image.ONE_LINE_DETAIL / 2
        sides = [
            "left"
            if (stroke[:, 0] < gap_middle).all()
            else "right"
            if (stroke[:, 0] > gap_middle).all()
            else "both"
            for stroke in found["yellow"]
        ]
        assert sorted(sides) == ["left", "right"]


def test_blank_image_has_no_line_to_draw(tmp_path):
    with pytest.raises(ValueError, match="nothing to draw"):
        one_line_drawing(tmp_path, Image.new("RGB", (CARD_PX, CARD_PX), "white"), 3)


def test_thinning_leaves_one_pixel_along_the_middle():
    mask = np.zeros((30, 80), dtype=bool)
    mask[10:19, 10:70] = True

    middle = image.thin(mask)[:, 20:60]

    assert (middle.sum(axis=0) == 1).all()
    assert set(np.nonzero(middle)[0]) <= {13, 14, 15}


def test_centerline_traces_a_ring_once_along_its_middle():
    rows, columns = np.mgrid[:80, :80]
    radius = np.hypot(columns - 39.5, rows - 39.5)

    (line,) = image.centerlines((radius >= 20) & (radius <= 28))

    assert (line[0] == line[-1]).all()
    assert np.hypot(line[:, 0] - 40, line[:, 1] - 40) == pytest.approx(24, abs=1.0)


def test_centerline_keeps_the_short_bar_of_a_bold_letter():
    letter_t = np.zeros((60, 50), dtype=bool)
    letter_t[10:18, 12:35] = True
    letter_t[10:52, 19:27] = True

    lines = image.centerlines(letter_t)

    ends = np.array([point for line in lines for point in (line[0], line[-1])])
    stem_left, stem_right = 19, 27
    assert ends[:, 0].min() < stem_left
    assert ends[:, 0].max() > stem_right
    assert ends[:, 1].max() > 47


def test_centerline_draws_each_pens_lines_once_along_their_middle(tmp_path):
    picture = Image.new("RGB", (CARD_PX, CARD_PX), "white")
    draw = ImageDraw.Draw(picture)
    draw.line([(30, 50), (170, 50)], fill=BLACK.rgb, width=9)
    draw.line([(30, 120), (170, 120)], fill=CYAN.rgb, width=5)
    image_file = tmp_path / "picture.png"
    picture.save(image_file)

    svg = image.image_to_centerline(image_file, CMYK, CARD_PX, 1, 3.0)

    found = {
        layer.get(LABEL): [
            np.array(re.findall(r"(-?[\d.]+),(-?[\d.]+)", subpath), dtype=float)
            for subpath in layer.find(f"{{{image.SVG_NS}}}path").get("d").split("M")
            if subpath.strip()
        ]
        for layer in ET.fromstring(svg).iter(f"{{{image.SVG_NS}}}g")
    }
    assert list(found) == ["cyan", "black"]
    for (line,), row in zip(found.values(), (120.5, 50.5)):
        assert line[:, 1] == pytest.approx(row, abs=1.0)
        assert line[:, 0].min() < 35
        assert line[:, 0].max() > 165


NIGHT_PENS = pens.parse_pens("grey=#8A8D8F,red=#D52B1E,blue=#1B4F9C,black=#231F20")
NIGHT_GREY, NIGHT_RED, NIGHT_BLUE, NIGHT_BLACK = (
    next(pen for pen in NIGHT_PENS if pen.name == name)
    for name in ("grey", "red", "blue", "black")
)


def night_layers(svg: str) -> dict[str, list[ET.Element]]:
    return {
        layer.get(LABEL): layer.findall(f"{{{image.SVG_NS}}}path")
        for layer in ET.fromstring(svg).iter(f"{{{image.SVG_NS}}}g")
    }


def test_night_pens_split_into_line_pens_and_colour_pens():
    neutrals, colours = image._night_pens(NIGHT_PENS)
    assert neutrals == [NIGHT_BLACK, NIGHT_GREY]
    assert set(colours) == {NIGHT_RED, NIGHT_BLUE}
    levels = image._line_levels(neutrals)
    assert levels[NIGHT_BLACK] == [min(image.NIGHT_LINE_LEVELS)]
    assert max(image.NIGHT_LINE_LEVELS) in levels[NIGHT_GREY]
    assert image._line_levels([NIGHT_BLACK]) == {
        NIGHT_BLACK: sorted(image.NIGHT_LINE_LEVELS)
    }


def test_night_scene_fills_paint_and_outlines_what_is_lit(tmp_path):
    picture = Image.new("RGB", (CARD_PX, CARD_PX), "black")
    draw = ImageDraw.Draw(picture)
    draw.rectangle([20, 20, 90, 90], fill=(240, 240, 240))
    draw.rectangle([110, 110, 180, 180], fill=(40, 60, 150))
    for x in range(115, 180, 12):
        draw.line([(x, 110), (x, 180)], fill=(200, 200, 210), width=2)
    image_file = tmp_path / "night.jpg"
    picture.save(image_file)

    svg = image.image_to_night(image_file, NIGHT_PENS, CARD_PX, 3, 4.0)

    found = night_layers(svg)
    assert set(found) == {"blue", "black", "grey"}
    (blue,) = found["blue"]
    assert blue.get("fill") == NIGHT_BLUE.hex
    rings = [
        np.array(re.findall(r"(-?[\d.]+),(-?[\d.]+)", ring), dtype=float)
        for ring in blue.get("d").split("Z")
        if ring.strip()
    ]
    painted = np.vstack(rings)
    assert painted.min(axis=0) == pytest.approx((110, 110), abs=6)
    assert painted.max(axis=0) == pytest.approx((180, 180), abs=6)
    assert len(rings) > 1, "the highlights stay paper"
    (outline,) = found["black"]
    assert outline.get("fill") == "none"
    points = np.array(
        re.findall(r"(-?[\d.]+),(-?[\d.]+)", outline.get("d")), dtype=float
    )
    assert points.min(axis=0) == pytest.approx((20, 20), abs=3)
    assert points.max(axis=0) == pytest.approx((180, 180), abs=3)


def test_dark_image_has_no_night_scene_to_draw(tmp_path):
    image_file = tmp_path / "dark.png"
    Image.new("RGB", (CARD_PX, CARD_PX), "black").save(image_file)
    with pytest.raises(ValueError, match="nothing to draw"):
        image.image_to_night(image_file, NIGHT_PENS, CARD_PX, 3, 4.0)
