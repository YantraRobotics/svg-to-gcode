"""Build the visual replay for the physical-intelligence transcript in test.txt.

The generated raster artwork is a visual concept, not executable robot paths.
"""

import hashlib
import json
from pathlib import Path

from svg_to_gcode.storyboard import windows

SCENES = [
    (
        1,
        "reliable-coffee",
        "Useful autonomy means handling a full cup reliably, without babysitting.",
        "smoothly handle cups",
    ),
    (
        1,
        "improvement-loop",
        "Practice, feedback, and another attempt turn failure into improvement.",
        "learn from its failures",
    ),
    (
        2,
        "human-recovery",
        "A human helps separate stuck boxes, rescuing a dead-end attempt.",
        "separate out the two boxes",
    ),
    (
        2,
        "judge-progress",
        "A shared value model recognizes progress and setbacks across tasks.",
        "general purpose value function",
    ),
    (
        3,
        "memory",
        "Remembering completed steps lets a robot finish a longer kitchen-cleaning task.",
        "memory at multiple time scales",
    ),
    (
        4,
        "one-model",
        "One generalist model brings coffee making, folding, and assembly together.",
        "single general purpose model",
    ),
    (
        4,
        "diverse-data",
        "Diverse experience becomes useful when the model also receives its context.",
        "data from the web",
    ),
    (
        5,
        "skill-transfer",
        "A folding skill transfers to a robot with different joints and proportions.",
        "lengths of the linkages",
    ),
    (
        6,
        "research-engineering",
        "Research and engineering offer different paths into building capable robots.",
        "how to handle uncertainty",
    ),
    (
        7,
        "real-practice",
        "Watching a skilled person helps; learning still needs physical practice.",
        "Roger Federer play tennis",
    ),
    (
        8,
        "swapped-hands",
        "A robot unexpectedly transfers an assembly skill between its grippers.",
        "pin with its left gripper",
    ),
    (
        8,
        "start-building",
        "Hands-on experiments with a small robot can become a route into robotics.",
        "in her bedroom",
    ),
]


def build():
    root = Path(__file__).resolve().parents[2]
    source = (root / "test.txt").read_text()
    chunks = windows(source)
    additions = []
    for batch, name, meaning, search in SCENES:
        window = chunks[batch - 1]
        start = window.text.lower().find(search.lower())
        if start < 0:
            raise ValueError(f"{name}: evidence missing from window {batch}")
        additions.append(
            {
                "id": name,
                "meaning": meaning,
                "evidence": window.text[max(0, start - 45) : start + len(search) + 100],
                "batch": batch,
                "ready": window.end_seconds,
                "window_sha256": window.sha256,
            }
        )
    output = root / "output/physical-intelligence-evolution"
    output.mkdir(parents=True, exist_ok=True)
    data = {
        "words": " ".join(c.text for c in chunks).split(),
        "duration": chunks[-1].end_seconds,
        "additions": additions,
        "source": source,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "mode": "Pre-generated illustrated replay; no live generation or robot motion",
    }
    template = (root / "examples/keynote/evolution.html").read_text()
    template = template.replace(
        "Keynote — watch the canvas grow",
        "Physical intelligence — watch the canvas grow",
    )
    template = template.replace(
        "Watch the keynote become a drawing.",
        "Physical intelligence, drawn as it unfolds.",
    )
    template = template.replace(
        "Cumulative text-free keynote doodles",
        "Cumulative text-free physical intelligence doodles",
    )
    (output / "index.html").write_text(
        template.replace("__DATA__", json.dumps(data).replace("<", "\\u003c"))
    )
    (output / "replay.json").write_text(json.dumps(data, indent=2) + "\n")
    print(
        f"{len(data['words'])} words, {len(chunks)} windows, {len(additions)} doodles: {output / 'index.html'}"
    )


if __name__ == "__main__":
    build()
