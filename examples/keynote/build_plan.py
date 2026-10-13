"""Codex-authored, text-free artwork fixture for the supplied test.txt keynote.

This is an explicit example plan, not a general transcript-to-art model. Every
batch includes a quotation from its own source window for validation and review.
"""

import json
from pathlib import Path

import svgelements as svg

from svg_to_gcode.storyboard import windows


class Sketch:
    def __init__(self):
        self.strokes = []

    def p(self, d, pen="black"):
        self.strokes.append({"pen": pen, "d": d})
        return self

    def add(self, other, x=0, y=0, width=100, height=None):
        height = width if height is None else height
        for stroke in other.strokes:
            path = svg.Path(stroke["d"])
            path *= svg.Matrix(
                f"translate({x} {y}) scale({width / 100} {height / 100})"
            )
            path.reify()
            self.p(path.d(), stroke["pen"])
        return self


def ring(x, y, r, pen="black"):
    k = r * 0.55228475
    return Sketch().p(
        f"M{x + r} {y} C{x + r} {y + k} {x + k} {y + r} {x} {y + r} "
        f"C{x - k} {y + r} {x - r} {y + k} {x - r} {y} "
        f"C{x - r} {y - k} {x - k} {y - r} {x} {y - r} "
        f"C{x + k} {y - r} {x + r} {y - k} {x + r} {y} Z",
        pen,
    )


def phone():
    return (
        Sketch()
        .p(
            "M28 6 Q22 6 22 14 L22 85 Q22 93 30 93 L69 93 Q77 93 77 85 L77 14 Q77 6 69 6 Z"
        )
        .p("M29 19 L70 19 L70 77 L29 77 Z")
        .p("M44 12 L56 12")
        .add(ring(49, 85, 3))
    )


def note(pen="blue"):
    return (
        Sketch()
        .p("M30 70 L30 20 L75 10 L75 61", pen)
        .add(ring(23, 72, 8, pen))
        .add(ring(68, 63, 8, pen))
    )


def globe():
    return (
        Sketch()
        .add(ring(50, 50, 37))
        .p("M13 50 Q50 40 87 50 M50 13 C25 28 25 72 50 87 C75 72 75 28 50 13", "blue")
        .p("M20 30 Q50 40 80 30 M20 70 Q50 61 80 70", "blue")
    )


def receiver():
    return Sketch().p(
        "M22 15 Q13 10 13 24 C17 51 46 82 73 86 Q88 86 84 72 L69 59 Q63 57 57 67 Q40 59 31 39 Q43 30 36 23 Z"
    )


def laptop():
    return (
        Sketch()
        .p("M17 13 Q15 11 13 15 L16 70 L84 70 L87 15 Q86 10 80 11 Z")
        .p("M21 19 L79 18 L77 62 L24 63 Z")
        .p("M16 70 L4 85 Q49 94 96 85 L84 70 M35 77 L66 77 M39 83 L62 83")
    )


def picture():
    return (
        Sketch()
        .p("M13 15 L88 12 L90 84 L15 87 Z")
        .p("M22 71 L40 49 L56 64 L68 39 L82 70", "green")
        .add(ring(36, 33, 6, "red"))
    )


def envelope():
    return (
        Sketch()
        .p("M10 25 L90 25 L90 80 L10 80 Z")
        .p("M10 25 L50 56 L90 25 M10 80 L34 53 M90 80 L66 53", "blue")
    )


def person():
    return (
        Sketch()
        .add(ring(50, 18, 10))
        .p("M47 16 L47 17 M54 16 L54 17 M46 22 Q51 26 56 21")
        .p("M44 29 Q34 43 39 64 Q50 69 62 63 Q66 46 56 29")
        .p("M44 66 Q42 79 32 91 L44 91 M58 65 Q61 80 72 88 L61 90")
        .p("M41 34 Q29 46 19 39 M60 34 Q73 42 82 27")
    )


def finger():
    return Sketch().p(
        "M63 92 C56 80 45 72 29 56 Q20 46 27 42 Q33 39 45 53 L43 14 Q43 4 50 5 Q56 5 56 15 L59 47 Q66 38 71 48 Q76 42 81 52 Q88 49 91 61 L94 79"
    )


def three_become_one():
    s = Sketch()
    s.add(phone(), 2, 0, 25).add(ring(14, 15, 4, "red"))
    s.add(receiver(), 36, 2, 22).add(globe(), 70, 0, 25)
    s.p("M15 26 C15 44 42 36 46 52", "red")
    s.p("M47 26 C48 40 53 41 51 52", "green")
    s.p("M83 26 C83 40 59 39 57 52", "blue")
    s.add(phone(), 19, 40, 62)
    s.p("M36 57 L36 82 L49 87 L49 62 Z M36 57 L61 57 L61 82 L49 87", "blue")
    s.p("M29 82 L26 91 Q48 101 75 90 L73 82 M30 91 Q50 96 70 90")
    return s


def fingers_lead():
    s = Sketch().add(finger(), 34, 36, 60)
    s.add(note(), 3, 19, 22).add(picture(), 38, 9, 22)
    s.add(person(), 69, 19, 24)
    return s.p("M10 52 Q36 36 79 49", "blue").p(
        "M44 31 L44 35 M33 33 L36 36 M56 32 L53 36", "red"
    )


def sync_life():
    s = Sketch().add(laptop(), 0, 15, 40).add(phone(), 70, 23, 31)
    s.p("M29 36 Q51 17 80 39", "blue")
    s.add(note(), 38, 33, 15).add(picture(), 56, 36, 15)
    return s.p("M38 74 Q59 91 83 65 M79 67 L83 65 L83 71", "green")


def senses():
    s = Sketch().add(person(), 0, 15, 54).add(phone(), 48, 16, 45)
    s.p("M31 24 Q49 5 68 14 M62 8 L68 14 L60 16", "blue")
    s.p("M59 81 Q40 95 25 80 M29 86 L25 80 L32 80", "blue")
    s.add(ring(85, 9, 4, "red"))
    return s.p("M85 1 L85 3 M94 9 L92 9 M91 3 L90 4", "red")


def touch_music():
    s = (
        Sketch()
        .add(ring(31, 48, 23))
        .add(ring(31, 48, 16, "blue"))
        .add(ring(31, 48, 4))
    )
    s.add(finger(), 6, 0, 34)
    s.p("M45 70 C60 95 64 33 93 44", "blue")
    return s.add(note(), 58, 48, 22).add(note(), 78, 18, 20)


def calls_join():
    s = Sketch().add(person(), 0, 11, 37).add(person(), 66, 9, 37)
    s.add(receiver(), 32, 49, 39)
    s.p("M24 35 C21 61 34 61 43 61", "blue")
    s.p("M77 34 C78 55 66 60 63 59", "green")
    return s.p("M42 79 Q50 93 60 80", "red")


def pick_voicemail():
    s = Sketch().p("M3 22 Q50 11 97 23", "blue")
    for x in [2, 37, 72]:
        s.add(envelope(), x, 19, 26)
    s.add(person(), 22, 44, 48)
    return s.p("M59 56 Q76 57 83 43 M76 46 L83 43 L82 50", "green")


def open_web():
    s = Sketch().p("M9 13 L91 13 L91 84 L9 84 Z M9 24 L91 24")
    s.add(globe(), 23, 25, 60)
    s.p("M12 27 L1 36 L1 78 L12 84 M88 27 L99 36 L99 78 L88 84", "blue")
    return s.add(person(), 24, 45, 42)


def map_coffee():
    s = Sketch().p(
        "M8 52 L34 43 L62 53 L90 44 L96 86 L66 95 L38 85 L12 95 Z M34 43 L38 85 M62 53 L66 95"
    )
    s.p("M21 81 C16 53 45 79 49 56 C52 42 67 48 75 30", "blue")
    s.p("M62 16 L88 16 Q89 39 76 39 Q63 39 62 16 M88 19 Q100 16 94 29 L87 30")
    s.p("M70 11 C63 6 76 5 71 0 M80 11 C73 7 86 4 80 0", "red")
    return s.p("M19 67 Q5 51 19 48 Q32 52 19 67", "green").add(ring(19, 54, 2, "green"))


def flow_together():
    s = Sketch().p("M3 62 C23 87 27 29 50 56 C69 80 82 32 98 51", "blue")
    s.add(note(), 0, 12, 24).add(receiver(), 27, 24, 24)
    s.add(envelope(), 52, 17, 27).add(picture(), 56, 12, 18)
    return s.add(note("green"), 76, 48, 26)


def launch():
    s = Sketch().p(
        "M3 43 L39 39 L41 83 L5 86 Z M4 51 L39 47 M12 35 L13 46 M28 33 L29 44"
    )
    s.p("M45 27 L82 9 L71 41 L62 29 L45 27 Z M62 29 L82 9", "blue")
    s.p("M39 41 Q54 45 57 32", "blue")
    return (
        s.add(ring(79, 72, 19))
        .add(ring(79, 72, 11, "green"))
        .p("M96 52 L79 72 M79 65 L79 72 L86 71", "red")
    )


def future():
    s = Sketch().add(person(), 0, 5, 62)
    s.p("M12 61 L47 61 M22 67 L46 67", "blue")
    s.p("M49 40 L71 75 L85 75")
    s.add(ring(82, 87, 4))
    s.p("M49 88 Q65 93 74 88", "blue")
    return s.p("M74 48 Q83 34 95 43 L92 47 M95 43 L94 37", "green")


SPECS = [
    (
        1,
        "three-become-one",
        [350, 345, 300, 300],
        "Three devices merge into one pocket-sized doorway",
        "these are not three separate devices",
        three_become_one,
    ),
    (
        1,
        "fingers-lead",
        [35, 35, 280, 230],
        "A fingertip conducts changing controls",
        "use our fingers",
        fingers_lead,
    ),
    (
        1,
        "sync-life",
        [350, 35, 280, 230],
        "Personal media travel from computer to phone",
        "syncs to your PC",
        sync_life,
    ),
    (
        2,
        "phone-senses",
        [675, 35, 280, 230],
        "Phone and person mirror each other",
        "accelerometer",
        senses,
    ),
    (
        2,
        "touch-music",
        [35, 285, 260, 210],
        "A fingertip sets a record and melody in motion",
        "touch your music",
        touch_music,
    ),
    (
        3,
        "calls-join",
        [35, 520, 260, 210],
        "Separate people join one conversation",
        "merge calls",
        calls_join,
    ),
    (
        3,
        "pick-voicemail",
        [705, 285, 260, 210],
        "Choose one hanging message instead of waiting in line",
        "random access voice mail",
        pick_voicemail,
    ),
    (
        4,
        "open-web",
        [705, 520, 260, 210],
        "A browser opens like shutters onto the world",
        "real Web browser",
        open_web,
    ),
    (
        5,
        "map-coffee",
        [25, 770, 210, 200],
        "A map route becomes steam from the coffee you found",
        "Starbucks",
        map_coffee,
    ),
    (
        6,
        "flow-together",
        [270, 770, 210, 200],
        "One melody passes through a call and a shared photo",
        "Music fades out",
        flow_together,
    ),
    (
        7,
        "launch-ambition",
        [515, 770, 210, 200],
        "A calendar launches a paper plane toward a target",
        "shipping these in June",
        launch,
    ),
    (
        8,
        "future-puck",
        [760, 770, 210, 200],
        "A skater aims toward where the puck is going",
        "where the puck is going",
        future,
    ),
]


def build():
    root = Path(__file__).resolve().parents[2]
    chunks = windows((root / "test.txt").read_text())
    target = Path(__file__).with_name("plans")
    target.mkdir(exist_ok=True)
    for window in chunks:
        items = []
        for batch, name, box, meaning, search, artist in SPECS:
            if batch != window.id:
                continue
            start = window.text.lower().find(search.lower())
            if start < 0:
                raise ValueError(f"{name}: source evidence missing from window {batch}")
            evidence = window.text[start : start + max(len(search), 40)]
            items.append(
                {
                    "id": name,
                    "meaning": meaning,
                    "evidence": evidence,
                    "box": box,
                    "strokes": artist().strokes,
                }
            )
        plan = {
            "window_id": window.id,
            "window_sha256": window.sha256,
            "base_revision": window.id - 1,
            "summary": "; ".join(item["meaning"] for item in items),
            "additions": items,
        }
        (target / f"{window.id:02d}.json").write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    build()
