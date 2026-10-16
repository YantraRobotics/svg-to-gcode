import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

type RGB = tuple[int, int, int]

DEFAULT_PENS = "yellow=#FFF200,cyan=#00AEEF,magenta=#EC008C,black=#231F20"
MIN_NIB_MM = 0.1
PAPER_RGB: RGB = (255, 255, 255)
CHANNEL_MAX = 255.0
PEN_NAME = re.compile(r"[A-Za-z0-9_-]+")
HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{6}")
# sRGB transfer function and sRGB to CIE XYZ under sRGB's own D65 white,
# as given in CSS Color Module Level 4.
SRGB_LINEAR_LIMIT = 0.04045
SRGB_LINEAR_SLOPE = 12.92
SRGB_OFFSET = 0.055
SRGB_GAMMA = 2.4
SRGB_TO_XYZ = np.array(
    [
        [506752 / 1228815, 87881 / 245763, 12673 / 70218],
        [87098 / 409605, 175762 / 245763, 12673 / 175545],
        [7918 / 409605, 87881 / 737289, 1001167 / 1053270],
    ]
)
D65_WHITE = np.array([0.3127 / 0.3290, 1.0, (1.0 - 0.3127 - 0.3290) / 0.3290])
LAB_EPSILON = 216 / 24389
LAB_KAPPA = 24389 / 27
# ink_coverage fits colours in light raised to 1/DISPLAY_GAMMA, close to how
# the eye weighs differences, by damped Gauss-Newton steps.
DISPLAY_GAMMA = 2.2
MIN_LIGHT = 1e-3
COVERAGE_ITERATIONS = 40
# Four pens can mix most colours more than one way; the damping keeps steps
# along those equally good mixes finite.
COVERAGE_DAMPING = 0.01
# Stacked inks multiply, so the fit has local minima; it starts from each of
# these coverages and keeps the best.
COVERAGE_STARTS = (0.0, 0.5, 1.0)
# Coverage cost for a pen absorbing all light. Every line costs drawing time, so
# black stands in for the cyan, magenta and yellow it would take to match it,
# while lighter pens cost less, since dark lines show the most.
INK_COST = 0.03


@dataclass(frozen=True)
class Pen:
    name: str
    rgb: RGB
    nib_mm: float | None = None

    @property
    def hex(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(*self.rgb)


def parse_hex(text: str) -> RGB:
    if not HEX_COLOR.fullmatch(text):
        raise ValueError(f"colour must look like #RRGGBB, got {text!r}")
    return int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)


def parse_nib(text: str) -> float:
    cleaned = text.strip().removesuffix("mm").strip()
    try:
        value = float(cleaned)
    except ValueError:
        raise ValueError(f"nib size must be a number in mm, got {text!r}") from None
    if not (MIN_NIB_MM <= value < math.inf):
        raise ValueError(
            f"nib size must be at least {MIN_NIB_MM:g} mm and finite, got {text!r}"
        )
    return value


def parse_pens(text: str) -> list[Pen]:
    """Pens from NAME=#RRGGBB[:NIB_MM],... sorted light to dark, the order they draw in."""
    pens = []
    for item in text.split(","):
        name, separator, spec = (part.strip() for part in item.partition("="))
        if not separator:
            raise ValueError(
                f"pens must be NAME=#RRGGBB separated by commas, got {item!r}"
            )
        if not PEN_NAME.fullmatch(name):
            raise ValueError(
                f"pen name {name!r} must be letters, digits, '-' or '_' with no spaces"
            )
        color, nib_sep, nib_text = (part.strip() for part in spec.partition(":"))
        nib_mm = parse_nib(nib_text) if nib_sep else None
        pens.append(Pen(name, parse_hex(color), nib_mm))
    names = [pen.name for pen in pens]
    if duplicates := sorted({name for name in names if names.count(name) > 1}):
        raise ValueError(f"pen names must be unique, got {', '.join(duplicates)} twice")
    return sorted(pens, key=lambda pen: -to_lab(np.array(pen.rgb))[0])


def to_linear(rgb: np.ndarray) -> np.ndarray:
    """Linear light 0..1 of sRGB colours with 0-255 channels, shape (..., 3)."""
    encoded = np.asarray(rgb, dtype=float) / CHANNEL_MAX
    return np.where(
        encoded <= SRGB_LINEAR_LIMIT,
        encoded / SRGB_LINEAR_SLOPE,
        ((encoded + SRGB_OFFSET) / (1.0 + SRGB_OFFSET)) ** SRGB_GAMMA,
    )


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """CIE L*a*b* (D65) of sRGB colours with 0-255 channels, shape (..., 3)."""
    return linear_to_lab(to_linear(rgb))


def linear_to_lab(linear: np.ndarray) -> np.ndarray:
    """CIE L*a*b* (D65) of linear sRGB light 0..1, shape (..., 3)."""
    xyz = np.asarray(linear, dtype=float) @ SRGB_TO_XYZ.T / D65_WHITE
    f = np.where(xyz > LAB_EPSILON, np.cbrt(xyz), (LAB_KAPPA * xyz + 16) / 116)
    lightness = 116 * f[..., 1] - 16
    return np.stack(
        [lightness, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def nearest_index(rgb: np.ndarray, pens: list[Pen]) -> np.ndarray:
    """Per colour, 0 when paper white is nearest in Lab, else 1 + the nearest pen's index."""
    return nearest_lab_index(to_lab(rgb), pens)


def nearest_lab_index(lab: np.ndarray, pens: list[Pen]) -> np.ndarray:
    """nearest_index for colours already in CIE L*a*b*, shape (..., 3)."""
    inks = to_lab(np.array([PAPER_RGB, *(pen.rgb for pen in pens)]))
    distances = np.linalg.norm(lab[..., np.newaxis, :] - inks, axis=-1)
    return np.argmin(distances, axis=-1)


def coverage_mixes(
    pens: list[Pen], levels: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Every way of covering the paper with each pen at one of levels, shape
    (mixes, pens), and the colour (CIE L*a*b*) each leaves, shape (mixes, 3).

    Hatches of different pens cross, so in linear light the paper reflects, channel
    by channel, the product over the pens of 1 - coverage * (1 - pen reflectance).
    """
    mixes = np.array(list(product(levels, repeat=len(pens))), dtype=float)
    absorbed = 1.0 - to_linear(np.array([pen.rgb for pen in pens]))
    reflected = np.prod(1.0 - mixes[..., np.newaxis] * absorbed, axis=1)
    return mixes, linear_to_lab(reflected)


def blends(pens: list[Pen], most: int) -> list[tuple[Pen, ...]]:
    """Every group of one to most pens, single pens first and in pen order."""
    if most < 1:
        raise ValueError(f"a blend needs at least one pen, got {most}")
    return [group for size in range(1, most + 1) for group in combinations(pens, size)]


def overlay(group: Sequence[Pen]) -> Pen:
    """The pens laid over each other, as one pen of the colour they leave on paper white.

    Inks filter the light like stacked gels, so their channels multiply.
    """
    channels = np.prod([np.array(pen.rgb) / CHANNEL_MAX for pen in group], axis=0)
    return Pen(
        "+".join(pen.name for pen in group),
        tuple(round(channel * CHANNEL_MAX) for channel in channels),
    )


def ink_coverage(
    rgb: np.ndarray, pens: list[Pen], ink_cost: float = INK_COST
) -> np.ndarray:
    """Per colour, the share of the paper each pen should cover, 0..1, shape (..., pens).

    Lines of different pens cross at random, so on average the paper reflects, in
    linear light and channel by channel, the product over the pens of
    1 - coverage * (1 - pen reflectance). Each pen's coverage costs ink_cost times
    the share of light it absorbs, traded against the colour error.
    """
    target = np.maximum(to_linear(rgb), MIN_LIGHT) ** (1 / DISPLAY_GAMMA)
    absorbed = 1.0 - to_linear(np.array([pen.rgb for pen in pens]))
    cost = ink_cost * absorbed.mean(axis=-1)
    fits = np.stack(
        [_fit_coverage(target, absorbed, cost, start) for start in COVERAGE_STARTS]
    )
    seen = _seen(_passed(fits, absorbed))
    misfits = np.sum((seen - target) ** 2, axis=-1) + fits @ cost
    best = np.argmin(misfits, axis=0)[np.newaxis, ..., np.newaxis]
    return np.take_along_axis(fits, best, axis=0)[0]


def nearest(rgb: RGB, pens: list[Pen]) -> Pen | None:
    """The pen closest to a colour, or None when it is closer to paper white."""
    index = int(nearest_index(np.array(rgb), pens))
    return pens[index - 1] if index else None


def _fit_coverage(
    target: np.ndarray, absorbed: np.ndarray, cost: np.ndarray, start: float
) -> np.ndarray:
    """Coverage fitted from start by damped Gauss-Newton steps.

    A pen at 0 or 1 that the gradient pushes further out is pinned for the step.
    """
    damping = COVERAGE_DAMPING * np.eye(len(absorbed))
    coverage = np.full((*target.shape[:-1], len(absorbed)), start)
    for _ in range(COVERAGE_ITERATIONS):
        passed = _passed(coverage, absorbed)
        seen = _seen(passed)
        slope = -seen[..., np.newaxis, :] * absorbed / (DISPLAY_GAMMA * passed)
        gradient = 2 * np.einsum("...pc,...c->...p", slope, seen - target) + cost
        pinned = ((coverage <= 0) & (gradient > 0)) | ((coverage >= 1) & (gradient < 0))
        slope = np.where(pinned[..., np.newaxis], 0.0, slope)
        curvature = 2 * np.einsum("...pc,...qc->...pq", slope, slope) + damping
        free_gradient = np.where(pinned, 0.0, gradient)[..., np.newaxis]
        step = np.linalg.solve(curvature, free_gradient)[..., 0]
        coverage = np.clip(coverage - step, 0.0, 1.0)
    return coverage


def _passed(coverage: np.ndarray, absorbed: np.ndarray) -> np.ndarray:
    """Share of light each pen's coverage lets through, shape (..., pens, 3)."""
    return np.maximum(1.0 - coverage[..., np.newaxis] * absorbed, MIN_LIGHT)


def _seen(passed: np.ndarray) -> np.ndarray:
    """Gamma-encoded light the paper reflects through every pen's ink."""
    return np.maximum(np.prod(passed, axis=-2), MIN_LIGHT) ** (1 / DISPLAY_GAMMA)
