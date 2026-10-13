"""Small, bounded vector drawings in normalized coordinates, then paper millimeters."""

import math
import re
from dataclasses import dataclass, field
from html import escape
from itertools import pairwise

from svg_to_gcode import gcode
from svg_to_gcode.pens import Pen


def circle(x: float, y: float, radius: float) -> gcode.Stroke:
    return [
        (
            x + radius * math.cos(i * math.tau / 24),
            y + radius * math.sin(i * math.tau / 24),
        )
        for i in range(25)
    ]


MOTIFS: dict[str, list[gcode.Stroke]] = {
    "tree": [
        [(0.5, 0), (0.5, 0.3)],
        [
            (0.1, 0.3),
            (0.3, 0.55),
            (0.2, 0.55),
            (0.5, 1),
            (0.8, 0.55),
            (0.7, 0.55),
            (0.9, 0.3),
            (0.1, 0.3),
        ],
    ],
    "sun": [circle(0.5, 0.5, 0.25)]
    + [
        [
            (0.5 + r * math.cos(i * math.tau / 8), 0.5 + r * math.sin(i * math.tau / 8))
            for r in (0.35, 0.48)
        ]
        for i in range(8)
    ],
    "city": [
        [
            (0, 0),
            (0, 0.55),
            (0.3, 0.55),
            (0.3, 0),
            (0.3, 0.9),
            (0.65, 0.9),
            (0.65, 0),
            (0.65, 0.4),
            (1, 0.4),
            (1, 0),
            (0, 0),
        ],
        [(0.4, 0.7), (0.55, 0.7)],
        [(0.4, 0.5), (0.55, 0.5)],
    ],
    "person": [
        circle(0.5, 0.8, 0.15),
        [(0.5, 0.65), (0.5, 0.3), (0.2, 0)],
        [(0.5, 0.3), (0.8, 0)],
        [(0.15, 0.5), (0.5, 0.6), (0.85, 0.5)],
    ],
    "heart": [
        [
            (0.5, 0.05),
            (0.05, 0.5),
            (0.05, 0.75),
            (0.2, 0.9),
            (0.35, 0.9),
            (0.5, 0.75),
            (0.65, 0.9),
            (0.8, 0.9),
            (0.95, 0.75),
            (0.95, 0.5),
            (0.5, 0.05),
        ]
    ],
    "bulb": [
        [
            (0.35, 0.2),
            (0.35, 0.35),
            (0.15, 0.6),
            (0.15, 0.8),
            (0.3, 0.95),
            (0.7, 0.95),
            (0.85, 0.8),
            (0.85, 0.6),
            (0.65, 0.35),
            (0.65, 0.2),
            (0.35, 0.2),
        ],
        [(0.35, 0.1), (0.65, 0.1)],
    ],
    "robot": [
        [(0.1, 0.25), (0.1, 0.8), (0.9, 0.8), (0.9, 0.25), (0.1, 0.25)],
        [(0.5, 0.8), (0.5, 1)],
        [(0.25, 0.6), (0.35, 0.6)],
        [(0.65, 0.6), (0.75, 0.6)],
        [(0.3, 0.4), (0.7, 0.4)],
        [(0.3, 0.25), (0.3, 0)],
        [(0.7, 0.25), (0.7, 0)],
    ],
    "rocket": [
        [
            (0.35, 0.2),
            (0.35, 0.7),
            (0.5, 1),
            (0.65, 0.7),
            (0.65, 0.2),
            (0.85, 0.1),
            (0.65, 0.5),
        ],
        [(0.35, 0.5), (0.15, 0.1), (0.35, 0.2)],
        [(0.4, 0.15), (0.5, 0), (0.6, 0.15)],
    ],
    "book": [
        [
            (0.5, 0.1),
            (0.1, 0.2),
            (0.1, 0.9),
            (0.5, 0.8),
            (0.9, 0.9),
            (0.9, 0.2),
            (0.5, 0.1),
            (0.5, 0.8),
        ]
    ],
    "cloud": [
        [
            (0.2, 0.2),
            (0.05, 0.35),
            (0.05, 0.5),
            (0.2, 0.65),
            (0.35, 0.65),
            (0.4, 0.85),
            (0.6, 0.95),
            (0.8, 0.8),
            (0.8, 0.65),
            (0.95, 0.55),
            (0.95, 0.35),
            (0.8, 0.2),
            (0.2, 0.2),
        ]
    ],
    "growth": [
        [(0.05, 0.95), (0.05, 0.05), (0.95, 0.05)],
        [(0.15, 0.2), (0.4, 0.45), (0.6, 0.35), (0.9, 0.85), (0.65, 0.8)],
        [(0.9, 0.85), (0.9, 0.6)],
    ],
    "globe": [
        circle(0.5, 0.5, 0.45),
        [(0.05, 0.5), (0.95, 0.5)],
        [
            (0.5, 0.05),
            (0.3, 0.25),
            (0.25, 0.5),
            (0.3, 0.75),
            (0.5, 0.95),
            (0.7, 0.75),
            (0.75, 0.5),
            (0.7, 0.25),
            (0.5, 0.05),
        ],
    ],
}

KEYWORDS = {
    "tree": "tree trees forest forests nature green environment plants garden",
    "sun": "sun solar sunshine renewable energy electricity power",
    "city": "city cities building buildings housing urban infrastructure home",
    "person": "people person team teams community communities collaboration human humans",
    "heart": "love care health healthcare kindness compassion heart",
    "bulb": "idea ideas innovation creative creativity invention thinking",
    "robot": "robot robots robotics ai artificial intelligence automation technology computers",
    "rocket": "rocket space launch moon mars exploration ambition",
    "book": "book books education learning learn school schools knowledge teaching",
    "cloud": "cloud clouds rain weather storm climate",
    "growth": "growth growing progress business economy economic revenue improve",
    "globe": "world earth global planet international countries",
}


def keyword_motif(text: str, recent: list[str]) -> str | None:
    """A deterministic, English-only fallback; unrelated speech produces no drawing."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    scores = {name: len(words & set(keys.split())) for name, keys in KEYWORDS.items()}
    candidates = [name for name in MOTIFS if scores[name] and name not in recent[-3:]]
    return max(candidates, key=scores.get) if candidates else None


@dataclass
class Canvas:
    paper: tuple[float, float]
    margin: float = 10
    cell: float = 40
    strokes: list[gcode.Stroke] = field(default_factory=list)
    names: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in (*self.paper, self.cell)):
            raise ValueError("paper and cell sizes must be positive and finite")
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be non-negative and finite")
        if self.capacity < 1:
            raise ValueError("paper must fit at least one cell inside its margins")

    @property
    def columns(self) -> int:
        return max(0, int((self.paper[0] - 2 * self.margin) // self.cell))

    @property
    def capacity(self) -> int:
        rows = max(0, int((self.paper[1] - 2 * self.margin) // self.cell))
        return self.columns * rows

    def place(self, name: str, feed: float, seconds: float) -> list[gcode.Stroke]:
        """Reserve a fresh cell. Limit ink length; travel and lifts add time."""
        if len(self.names) >= self.capacity:
            raise ValueError("canvas full; replace the paper and start a new session")
        shape = MOTIFS[name]
        length = sum(math.dist(a, b) for line in shape for a, b in pairwise(line))
        size = min(self.cell * 0.75, feed / 60 * seconds / length)
        if size < 8:
            raise ValueError(
                "drawing budget too small: increase --ink-seconds or --feed"
            )
        row, column = divmod(len(self.names), self.columns)
        if row % 2:
            column = self.columns - 1 - column
        x = self.margin + column * self.cell + (self.cell - size) / 2
        y = self.paper[1] - self.margin - (row + 1) * self.cell + (self.cell - size) / 2
        strokes = [
            [(x + px * size, y + py * size) for px, py in line] for line in shape
        ]
        self.names.append(name)
        self.strokes.extend(strokes)
        return strokes

    def svg(self) -> str:
        width, height = self.paper
        paths = []
        for line in self.strokes:
            points = " ".join(f"{x:.3f},{height - y:.3f}" for x, y in line)
            paths.append(f'<polyline points="{points}"/>')
        title = escape(", ".join(self.names) or "Listening for ideas")
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
            f'<title>{title}</title><rect width="100%" height="100%" fill="white"/>'
            '<g fill="none" stroke="#231f20" stroke-width="0.8" '
            'stroke-linecap="round" stroke-linejoin="round">'
            + "".join(paths)
            + "</g></svg>\n"
        )


def drawing_code(strokes: list[gcode.Stroke], feed: float, lift: float) -> str:
    """Use the loaded black pen, without invoking an interactive pen swap."""
    code = gcode.to_gcode(
        [gcode.Layer(1, Pen("black", (35, 31, 32)), strokes)], feed, lift
    )
    return code.replace("T1 M6\n", "")
