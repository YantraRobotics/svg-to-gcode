"""Replay timestamped transcript windows as immutable, colored drawing additions.

Plans are data authored by Codex for the prototype. This module never calls a
model or a controller. Future planners must satisfy the same validation boundary.
"""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path

import svgelements as svg
from PIL import Image, ImageDraw

from svg_to_gcode import gcode, robot
from svg_to_gcode.pens import Pen

PALETTE = {
    "black": Pen("black", (35, 31, 32)),
    "blue": Pen("blue", (27, 79, 156)),
    "green": Pen("green", (0, 132, 61)),
    "red": Pen("red", (213, 43, 30)),
}
PATH_CHARS = re.compile(r"[MLCQAZmlcqaz0-9eE.,+\s-]+")
IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{0,63}")
GRID = 1000.0
# Canvas-grid units: blank border, smallest doodle box, and clear space between boxes.
MARGIN = 20.0
MIN_BOX = 40.0
GAP = 8.0
# Drawing may use this share of each interval; the rest covers planning and timing slack.
DRAW_SHARE = 0.8


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def clean_transcript(text: str) -> str:
    """Discard transcript wrapper/page credits, but retain the spoken source order."""
    if "TRANSCRIPT:" in text:
        text = text.split("TRANSCRIPT:", 1)[1]
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("©"):
            continue
        if re.match(
            r"^(Steve Jobs|Stan Sigman|Dr\. Eric Schmidt|Jerry Yang)\s*[–-]", line
        ):
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        lines.append(line)
    return " ".join(" ".join(lines).split())


@dataclass(frozen=True)
class Window:
    id: int
    start_seconds: float
    end_seconds: float
    text: str
    sha256: str
    partial: bool


def windows(text: str, interval: float = 600, wpm: float = 150) -> list[Window]:
    """Synthetic timestamps for an untimed transcript; never infer actual speech timing."""
    if not all(math.isfinite(v) and v > 0 for v in (interval, wpm)):
        raise ValueError("interval and words per minute must be positive and finite")
    words = clean_transcript(text).split()
    size = math.floor(interval * wpm / 60)
    if size < 1:
        raise ValueError("a window must contain at least one word")
    result = []
    for start in range(0, len(words), size):
        chunk = " ".join(words[start : start + size])
        end = min(start + size, len(words))
        result.append(
            Window(
                len(result) + 1,
                start / wpm * 60,
                end / wpm * 60,
                chunk,
                digest(chunk),
                end - start < size,
            )
        )
    return result


def overlaps(a, b, gap=GAP) -> bool:
    x, y, w, h = a
    bx, by, bw, bh = b
    return not (
        x + w + gap <= bx
        or bx + bw + gap <= x
        or y + h + gap <= by
        or by + bh + gap <= y
    )


def validate_plan(plan: dict, window: Window, committed: list[dict]) -> list[dict]:
    if (
        plan.get("window_id"),
        plan.get("window_sha256"),
        plan.get("base_revision"),
    ) != (window.id, window.sha256, window.id - 1):
        raise ValueError("plan has wrong window, source hash, or base revision")
    additions = plan.get("additions")
    if not isinstance(additions, list) or len(additions) > 3:
        raise ValueError("a window may add zero to three doodles")
    occupied = [item["box"] for item in committed]
    used = {item["id"] for item in committed}
    for item in additions:
        name = item.get("id", "")
        if not isinstance(name, str) or not IDENTIFIER.fullmatch(name) or name in used:
            raise ValueError("doodle IDs must be unique simple names")
        used.add(name)
        evidence = item.get("evidence", "")
        if (
            not isinstance(evidence, str)
            or len(evidence) < 12
            or evidence not in window.text
        ):
            raise ValueError(f"{name}: evidence must occur in this transcript window")
        box = item.get("box")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
        ):
            raise ValueError(f"{name}: box must contain four finite numbers")
        x, y, width, height = box
        if (
            min(x, y) < MARGIN
            or min(width, height) < MIN_BOX
            or max(x + width, y + height) > GRID - MARGIN
        ):
            raise ValueError(f"{name}: box must fit inside canvas margins")
        if any(overlaps(box, previous) for previous in occupied):
            raise ValueError(
                f"{name}: reserved space overlaps previous ink or another addition"
            )
        occupied.append(box)
        strokes = item.get("strokes")
        if not isinstance(strokes, list) or not 1 <= len(strokes) <= 100:
            raise ValueError(f"{name}: expected 1–100 strokes")
        for stroke in strokes:
            if stroke.get("pen") not in PALETTE:
                raise ValueError("only red, green, blue and black pens are allowed")
            d = stroke.get("d", "")
            if not isinstance(d, str) or len(d) > 4000 or not PATH_CHARS.fullmatch(d):
                raise ValueError(
                    "drawing data must be a bounded SVG path, never text or code"
                )
            path = svg.Path(d)
            if not 1 <= len(path) <= 200:
                raise ValueError("path has too many segments or is empty")
            bounds = path.bbox()
            if (
                bounds is None
                or not all(math.isfinite(v) for v in bounds)
                or min(bounds) < -0.001
                or max(bounds) > 100.001
            ):
                raise ValueError("path must fit its local 0–100 drawing space")
    return additions


def paths(item: dict, paper: tuple[float, float]):
    """Flatten local curves into fixed paper coordinates, without fitting old ink again."""
    x, y, width, height = item["box"]
    scale_x = width / 100 * paper[0] / GRID
    scale_y = height / 100 * paper[1] / GRID
    for stroke in item["strokes"]:
        for line in gcode._flatten(svg.Path(stroke["d"]), 0.1 / max(scale_x, scale_y)):
            if len(line) < 2:
                continue
            yield (
                stroke["pen"],
                [
                    (
                        x / GRID * paper[0] + px * scale_x,
                        paper[1] - y / GRID * paper[1] - py * scale_y,
                    )
                    for px, py in line
                ],
            )


def layers(additions: list[dict], paper: tuple[float, float]) -> list[gcode.Layer]:
    by_pen = {name: [] for name in PALETTE}
    for item in additions:
        for pen, points in paths(item, paper):
            by_pen[pen].append(points)
    return [
        gcode.Layer(i, PALETTE[name], by_pen[name])
        for i, name in enumerate(PALETTE, 1)
        if by_pen[name]
    ]


def svg_group(item: dict) -> str:
    x, y, width, height = item["box"]
    strokes = "".join(
        f'<path d="{escape(stroke["d"], quote=True)}" stroke="{PALETTE[stroke["pen"]].hex}"/>'
        for stroke in item["strokes"]
    )
    return (
        f'<g id="{item["id"]}" transform="translate({x} {y}) scale({width / 100} {height / 100})" '
        f'fill="none" stroke-width="1.15" stroke-linecap="round" stroke-linejoin="round">'
        + strokes
        + "</g>"
    )


def svg_document(items: list[dict]) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">'
        + "".join(svg_group(item) for item in items)
        + "</svg>\n"
    )


def save_png(
    items: list[dict], path: Path, paper: tuple[float, float] = (GRID, GRID)
) -> None:
    """Render actual vector paths, including pen colors, for an inspectable preview."""
    scale = 2000 / max(paper)
    size = (round(paper[0] * scale), round(paper[1] * scale))
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for item in items:
        _, _, box_width, box_height = item["box"]
        box_mm = min(box_width / GRID * paper[0], box_height / GRID * paper[1])
        width = max(2, round(1.15 * box_mm / 100 * scale))
        for pen, line in paths(item, paper):
            points = [(x * scale, (paper[1] - y) * scale) for x, y in line]
            draw.line(points, fill=PALETTE[pen].rgb, width=width, joint="curve")
    image.resize((size[0] // 2, size[1] // 2), Image.Resampling.LANCZOS).save(path)


def compile_batch(
    window: Window,
    summary: str,
    additions: list[dict],
    committed: list[dict],
    available_at: float,
    paper: tuple[float, float],
    feed: float,
    rapid: float,
    lift: float,
    swap_seconds: float,
) -> dict:
    """One window's new ink as G-code, with its estimated serial drawing slot."""
    compiled = layers(additions, paper)
    code = gcode.to_gcode(compiled, feed, lift) if compiled else ""
    motion, move_count = 0.0, 0
    if code:
        robot.check_on_paper(code, paper)
        segments, _ = robot.plan_motion(robot.parse_gcode(code, (0, 0, lift), rapid), 0)
        _, _, motion = robot.summarize(segments, robot.Placement((0, 0, 0), 0, 0))
        move_count = sum(len(part) for part in segments if isinstance(part, list))
    # Budget includes conservative per-motion settling and round trips to swap height.
    estimate = (
        motion
        + move_count * 0.1
        + len(compiled) * (swap_seconds + 2 * robot.PEN_CHANGE_HEIGHT_MM / rapid * 60)
    )
    starts_at = max(window.end_seconds, available_at)
    return {
        "window": asdict(window),
        "summary": summary,
        "additions": additions,
        "code": code,
        "motion_seconds": round(motion, 1),
        "estimated_seconds": round(estimate, 1),
        "pen_loads": len(compiled),
        "draw_starts_at": round(starts_at, 1),
        "draw_finishes_at": round(starts_at + estimate, 1),
        "wait_seconds": round(starts_at - window.end_seconds, 1),
        "canvas_sha256": digest(svg_document(committed + additions)),
        "svg": svg_document(additions),
    }


def write_batch(
    output: Path, batch: dict, committed: list[dict], paper: tuple[float, float]
) -> None:
    """Write a batch's speech, new ink, and the canvas it completes (committed)."""
    index = batch["window"]["id"]
    (output / f"{index:02d}-transcript.txt").write_text(batch["window"]["text"] + "\n")
    (output / f"{index:02d}-additions.svg").write_text(batch["svg"])
    if batch["code"]:
        (output / f"{index:02d}-additions.gcode").write_text(batch["code"])
    (output / f"{index:02d}-canvas.svg").write_text(svg_document(committed))
    save_png(committed, output / f"{index:02d}-canvas.png", paper)


def write_session(
    output: Path, report: dict, committed: list[dict], paper: tuple[float, float]
) -> None:
    """Write the session report, the cumulative canvas and the batch viewer."""
    (output / "session.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "canvas.svg").write_text(svg_document(committed))
    save_png(committed, output / "canvas.png", paper)
    template = Path(__file__).with_name("storyboard.html").read_text()
    payload = json.dumps(report).replace("<", "\\u003c")
    (output / "index.html").write_text(template.replace("__SESSION_DATA__", payload))


def render(
    transcript: Path,
    plans: Path,
    output: Path,
    interval=600.0,
    wpm=150.0,
    paper=(250.0, 250.0),
    feed=1000.0,
    rapid=3000.0,
    lift=2.0,
    swap_seconds=20.0,
) -> dict:
    """Compile a complete keyless replay. Physical execution is deliberately separate."""
    chunks = windows(transcript.read_text(), interval, wpm)
    if not chunks:
        raise ValueError("transcript contains no speech")
    if not all(math.isfinite(v) and v > 0 for v in (*paper, feed, rapid, lift)):
        raise ValueError(
            "paper dimensions, feed rates and lift must be positive and finite"
        )
    if max(feed, rapid) > robot.MAX_FEED_MM_MIN:
        raise ValueError("feed rate exceeds driver limit")
    if not math.isfinite(swap_seconds) or swap_seconds < 0:
        raise ValueError("swap duration must be finite and non-negative")
    # Validate the whole fixture before creating output; never accept changed source silently.
    committed, batches = [], []
    for chunk in chunks:
        plan = json.loads((plans / f"{chunk.id:02d}.json").read_text())
        additions = validate_plan(plan, chunk, committed)
        available_at = batches[-1]["draw_finishes_at"] if batches else 0.0
        batch = compile_batch(
            chunk,
            plan["summary"],
            additions,
            committed,
            available_at,
            paper,
            feed,
            rapid,
            lift,
            swap_seconds,
        )
        if batch["estimated_seconds"] > interval * DRAW_SHARE:
            raise ValueError(
                f"window {chunk.id}: estimated drawing time exceeds "
                f"{DRAW_SHARE:.0%} of the update interval"
            )
        committed.extend(additions)
        batches.append(batch)
    output.mkdir(parents=True, exist_ok=False)
    committed = []
    for batch in batches:
        index = batch["window"]["id"]
        request = {
            "instruction": "Add zero to three explanatory doodles using only this window and previous context. "
            "No text or numbers in artwork. Only black, red, green and blue single-line paths. "
            "Return additions; never redraw or move committed ink.",
            "window": batch["window"],
            "base_revision": index - 1,
            "prior_canvas": [
                {
                    "id": item["id"],
                    "box": item["box"],
                    "meaning": item["meaning"],
                    "sha256": digest(svg_group(item)),
                }
                for item in committed
            ],
            "palette": {name: pen.hex for name, pen in PALETTE.items()},
        }
        (output / f"{index:02d}-planner-input.json").write_text(
            json.dumps(request, indent=2) + "\n"
        )
        committed.extend(batch["additions"])
        write_batch(output, batch, committed, paper)
    report = {
        "mode": "codex-authored replay; no model requests or robot motion",
        "timing": "synthetic at constant words per minute, not original recording timestamps",
        "interval_seconds": interval,
        "wpm": wpm,
        "paper_mm": paper,
        "doodles": len(committed),
        "batches": batches,
    }
    write_session(output, report, committed, paper)
    return report
