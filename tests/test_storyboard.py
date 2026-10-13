import copy
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from svg_to_gcode import gcode, robot, storyboard

ROOT = Path(__file__).resolve().parents[1]


def sample():
    window = storyboard.windows("three devices become one phone", 600, 150)[0]
    plan = {
        "window_id": 1,
        "base_revision": 0,
        "window_sha256": window.sha256,
        "summary": "one phone",
        "additions": [
            {
                "id": "one-phone",
                "box": [50, 50, 200, 200],
                "meaning": "three devices become one",
                "evidence": "three devices become one",
                "strokes": [
                    {"pen": "black", "d": "M10 10 Q50 0 90 10 L90 90 L10 90 Z"},
                    {"pen": "red", "d": "M20 20 L80 80"},
                ],
            }
        ],
    }
    return window, plan


def test_ten_minute_windows_and_final_flush_are_lossless():
    text = " ".join(f"word{i}" for i in range(3101))
    chunks = storyboard.windows(text)
    assert [w.end_seconds for w in chunks] == pytest.approx([600, 1200, 1240.4])
    assert [w.partial for w in chunks] == [False, False, True]
    assert " ".join(w.text for w in chunks) == text
    assert "word1500" not in chunks[0].text.split()


def test_transcript_wrapper_does_not_become_speech():
    text = "intro\nTRANSCRIPT:\nSteve Jobs- Apple CEO\nHello world.\n© 2020 Publisher\n[Video Clip]\nMore speech."
    assert storyboard.clean_transcript(text) == "Hello world. More speech."


@pytest.mark.parametrize("interval,wpm", [(0, 150), (600, math.nan), (0.01, 1)])
def test_bad_timing_rejected(interval, wpm):
    with pytest.raises(ValueError):
        storyboard.windows("hello", interval, wpm)


@pytest.mark.parametrize(
    "field,value", [("window_id", 2), ("base_revision", 1), ("window_sha256", "stale")]
)
def test_stale_plans_rejected(field, value):
    window, plan = sample()
    plan[field] = value
    with pytest.raises(ValueError, match="wrong window"):
        storyboard.validate_plan(plan, window, [])


def test_future_source_cannot_support_a_current_doodle():
    window, plan = sample()
    plan["additions"][0]["evidence"] = "this information has not been spoken yet"
    with pytest.raises(ValueError, match="this transcript window"):
        storyboard.validate_plan(plan, window, [])


def test_existing_ink_cannot_be_overwritten_or_reused():
    window, plan = sample()
    committed = copy.deepcopy(plan["additions"])
    with pytest.raises(ValueError, match="unique"):
        storyboard.validate_plan(plan, window, committed)
    plan["additions"][0]["id"] = "different-name"
    with pytest.raises(ValueError, match="overlaps"):
        storyboard.validate_plan(plan, window, committed)


@pytest.mark.parametrize(
    "path", ["<text>hello</text>", "M0 0 L101 99", "M0 0 Lnan 0", "M0 0 L1e999 1", ""]
)
def test_invalid_or_off_region_geometry_is_rejected(path):
    window, plan = sample()
    plan["additions"][0]["strokes"][0]["d"] = path
    with pytest.raises(ValueError):
        storyboard.validate_plan(plan, window, [])


def test_only_available_pens_are_accepted():
    window, plan = sample()
    plan["additions"][0]["strokes"][0]["pen"] = "purple"
    with pytest.raises(ValueError, match="only red"):
        storyboard.validate_plan(plan, window, [])


def test_delta_gcode_uses_fixed_placement_and_explicit_pen_changes():
    window, plan = sample()
    items = storyboard.validate_plan(plan, window, [])
    before = storyboard.svg_group(items[0])
    layers = storyboard.layers(items, (250, 250))
    code = gcode.to_gcode(layers, 1000, 2)
    robot.check_on_paper(code, (250, 250))
    steps = robot.parse_gcode(code, (0, 0, 2), 3000)
    assert {step.pen.name for step in steps if isinstance(step, robot.PenChange)} == {
        "black",
        "red",
    }
    assert storyboard.svg_group(items[0]) == before
    root = ET.fromstring(storyboard.svg_document(items))
    assert not root.findall(".//{http://www.w3.org/2000/svg}text")


def test_replay_preserves_prior_vectors_and_serializes_tail(tmp_path):
    # Use small windows with enough words to create a final partial batch.
    transcript = tmp_path / "input.txt"
    transcript.write_text(" ".join(["three devices become one phone"] * 301))
    chunks = storyboard.windows(transcript.read_text())
    plans = tmp_path / "plans"
    plans.mkdir()
    for chunk in chunks:
        _, plan = sample()
        plan.update(
            window_id=chunk.id, base_revision=chunk.id - 1, window_sha256=chunk.sha256
        )
        plan["additions"][0]["id"] = f"phone-{chunk.id}"
        plan["additions"][0]["box"] = [50 + (chunk.id - 1) * 300, 50, 200, 200]
        (plans / f"{chunk.id:02d}.json").write_text(json.dumps(plan))
    output = tmp_path / "replay"
    report = storyboard.render(transcript, plans, output)
    assert len(report["batches"]) == 2
    first, second = report["batches"]
    assert second["wait_seconds"] > 0
    assert second["draw_starts_at"] == first["draw_finishes_at"]
    first_svg = ET.parse(output / "01-canvas.svg").getroot()
    second_svg = ET.parse(output / "02-canvas.svg").getroot()
    assert ET.tostring(first_svg[0]) == ET.tostring(second_svg[0])
    delta = ET.parse(output / "02-additions.svg").getroot()
    assert len(delta) == 1 and delta[0].attrib["id"] == "phone-2"
    request = json.loads((output / "01-planner-input.json").read_text())
    assert request["prior_canvas"] == []
    assert request["window"]["text"] == chunks[0].text
    assert "__SESSION_DATA__" not in (output / "index.html").read_text()
    with pytest.raises(FileExistsError):
        storyboard.render(transcript, plans, output)


def test_rejects_budget_overrun_before_creating_output(tmp_path):
    window, plan = sample()
    transcript = tmp_path / "input.txt"
    transcript.write_text(window.text)
    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / "01.json").write_text(json.dumps(plan))
    output = tmp_path / "replay"
    with pytest.raises(ValueError, match="drawing time"):
        storyboard.render(transcript, plans, output, feed=1)
    assert not output.exists()


def test_keynote_fixture_has_twelve_text_free_doodles_in_eight_windows():
    plans = sorted((ROOT / "examples/keynote/plans").glob("*.json"))
    assert len(plans) == 8
    additions = [
        item for path in plans for item in json.loads(path.read_text())["additions"]
    ]
    assert len(additions) == 12
    assert {stroke["pen"] for item in additions for stroke in item["strokes"]} == set(
        storyboard.PALETTE
    )
    for i, a in enumerate(additions):
        for b in additions[i + 1 :]:
            assert not storyboard.overlaps(a["box"], b["box"])
