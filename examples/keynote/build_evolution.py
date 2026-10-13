"""Build an illustrated transcript replay; raster artwork is not robot motion data."""

import json
from pathlib import Path

from svg_to_gcode.storyboard import windows


def build():
    root = Path(__file__).resolve().parents[2]
    output = root / "output/keynote-evolution"
    if not (output / "artwork.png").is_file():
        raise FileNotFoundError(
            "Save the generated canvas as output/keynote-evolution/artwork.png"
        )
    chunks = windows((root / "test.txt").read_text())
    additions = []
    for chunk in chunks:
        plan = json.loads(
            (Path(__file__).with_name("plans") / f"{chunk.id:02}.json").read_text()
        )
        if plan["window_sha256"] != chunk.sha256:
            raise ValueError("Transcript differs from the artwork's source plans")
        for item in plan["additions"]:
            additions.append(
                {
                    "id": item["id"],
                    "meaning": item["meaning"],
                    "evidence": item["evidence"],
                    "batch": chunk.id,
                    "ready": chunk.end_seconds,
                }
            )
    data = {
        "words": " ".join(c.text for c in chunks).split(),
        "duration": chunks[-1].end_seconds,
        "additions": additions,
        "source": (root / "test.txt").read_text(),
    }
    template = Path(__file__).with_name("evolution.html").read_text()
    (output / "index.html").write_text(
        template.replace("__DATA__", json.dumps(data).replace("<", "\\u003c"))
    )
    (output / "replay.json").write_text(json.dumps(data, indent=2) + "\n")
    print(output / "index.html")


if __name__ == "__main__":
    build()
