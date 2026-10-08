import numpy as np
import pytest

from svg_to_gcode import pens

CMYK = pens.parse_pens(pens.DEFAULT_PENS)


def test_default_pens_sort_light_to_dark():
    assert [pen.name for pen in CMYK] == ["yellow", "cyan", "magenta", "black"]
    assert CMYK[0].hex == "#FFF200"


def test_lab_matches_reference_values():
    assert pens.to_lab(np.array(pens.PAPER_RGB)) == pytest.approx((100, 0, 0), abs=1e-6)
    assert pens.to_lab(np.array((0, 0, 0))) == pytest.approx((0, 0, 0), abs=1e-6)
    # sRGB red in CIE Lab under D65.
    assert pens.to_lab(np.array((255, 0, 0))) == pytest.approx(
        (53.24, 80.09, 67.20), abs=0.01
    )


@pytest.mark.parametrize(
    ("rgb", "expected"),
    [
        ((255, 255, 255), None),
        ((240, 238, 230), None),
        ((10, 10, 10), "black"),
        ((128, 128, 128), "black"),
        ((0, 160, 230), "cyan"),
        ((230, 20, 130), "magenta"),
        ((255, 0, 0), "magenta"),
        ((250, 230, 40), "yellow"),
    ],
)
def test_nearest_pen_or_paper(rgb, expected):
    pen = pens.nearest(rgb, CMYK)
    assert (pen.name if pen else None) == expected


def test_nearest_index_maps_an_image():
    image = np.array([[[255, 255, 255], [35, 31, 32]], [[0, 174, 239], [255, 242, 0]]])
    assert pens.nearest_index(image, CMYK).tolist() == [[0, 4], [2, 1]]


def test_blends_list_single_pens_then_pairs():
    yellow, cyan, magenta, black = CMYK
    groups = pens.blends(CMYK, 2)

    assert groups[:4] == [(yellow,), (cyan,), (magenta,), (black,)]
    assert groups[4:] == [
        (yellow, cyan),
        (yellow, magenta),
        (yellow, black),
        (cyan, magenta),
        (cyan, black),
        (magenta, black),
    ]
    assert pens.blends(CMYK, 1) == groups[:4]
    with pytest.raises(ValueError, match="at least one pen"):
        pens.blends(CMYK, 0)


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (("yellow",), (255, 242, 0)),
        (("yellow", "magenta"), (236, 0, 0)),
        (("yellow", "cyan"), (0, 165, 0)),
        (("cyan", "magenta"), (0, 0, 131)),
    ],
)
def test_overlaid_inks_multiply(names, expected):
    by_name = {pen.name: pen for pen in CMYK}
    mixed = pens.overlay([by_name[name] for name in names])

    assert mixed == pens.Pen("+".join(names), expected)


def test_ink_coverage_leaves_paper_bare_and_finds_pens_and_their_mixes():
    yellow, cyan, magenta, _ = CMYK
    colours = [
        pens.PAPER_RGB,
        yellow.rgb,
        cyan.rgb,
        pens.overlay([yellow, magenta]).rgb,
    ]

    coverage = pens.ink_coverage(np.array(colours), CMYK)

    expected = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0]]
    assert coverage == pytest.approx(np.array(expected, dtype=float), abs=0.05)


@pytest.mark.parametrize(
    "rgb", [(128, 128, 128), (200, 120, 60), (60, 90, 140), (30, 20, 25)]
)
def test_free_ink_coverage_reproduces_colours_the_pens_can_reach(rgb):
    coverage = pens.ink_coverage(np.array(rgb), CMYK, ink_cost=0.0)

    absorbed = 1 - pens.to_linear(np.array([pen.rgb for pen in CMYK]))
    light = np.prod(1 - coverage[:, np.newaxis] * absorbed, axis=0)
    assert light == pytest.approx(pens.to_linear(np.array(rgb)), abs=0.002)


def test_black_stands_in_for_the_other_pens_in_greys():
    coverage = pens.ink_coverage(np.array((128, 128, 128)), CMYK)

    assert coverage[:3] == pytest.approx(np.zeros(3), abs=0.01)
    assert coverage[3] > 0.7


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "NAME=#RRGGBB"),
        ("red", "NAME=#RRGGBB"),
        ("red=#FF00", "#RRGGBB"),
        ("red=FF0000", "#RRGGBB"),
        ("=#FF0000", "pen name"),
        ("light red=#FF8080", "pen name"),
        ("red=#FF0000,red=#EE0000", "unique"),
    ],
)
def test_parse_pens_rejects_bad_input(text, message):
    with pytest.raises(ValueError, match=message):
        pens.parse_pens(text)
