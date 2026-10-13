import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest

import main
from svg_to_gcode import robot, speech, storyboard, talk

PHRASE = "robots learn from their failures"


def answer(*quotes: str) -> dict:
    return {
        "summary": "Robots learn by trying again.",
        "additions": [
            {
                "meaning": f"a doodle of {quote}",
                "evidence": quote,
                "strokes": [
                    {"pen": "black", "d": "M10 10 C 30 0 70 0 90 10 L 90 90 Z"},
                    {"pen": "red", "d": "M20 80 Q 50 40 80 80"},
                ],
            }
            for quote in quotes
        ],
    }


@pytest.fixture
def planner(monkeypatch):
    """Stand in for Codex: record each prompt's context and answer from replies."""
    contexts, replies = [], []

    def ask(prompt, schema, model):
        contexts.append(json.loads(prompt[prompt.index('{\n "story_so_far"') :]))
        reply = replies.pop(0) if replies else answer(PHRASE)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(talk, "ask_codex", ask)
    monkeypatch.setattr(talk.shutil, "which", lambda name: f"/usr/bin/{name}")
    return SimpleNamespace(contexts=contexts, replies=replies)


def rehearsal(tmp_path, words: str, *extra: str):
    path = tmp_path / "talk.txt"
    path.write_text(words)
    return main.build_parser().parse_args(
        ["talk", str(tmp_path / "session"), "--transcript", str(path)]
        + ["--paper", "250x250", *extra]
    )


def session(args) -> list[dict]:
    return json.loads((args.output / "session.json").read_text())["batches"]


def statuses(args) -> list[str]:
    lines = (args.output / "events.jsonl").read_text().splitlines()
    return [json.loads(line)["status"] for line in lines]


def test_rehearsal_doodles_each_window_without_moving_old_ink(tmp_path, planner):
    # 1500 words fill a ten-minute window at 150 wpm; the rest is a final partial one.
    words = " ".join(
        [PHRASE + " today"] * 250 + ["the arm folds laundry after practice"]
    )
    planner.replies.extend([answer(PHRASE), answer("the arm folds laundry")])
    args = rehearsal(tmp_path, words)
    assert main.run_talk(args) == 0
    first, second = session(args)
    assert second["window"]["partial"]
    assert "laundry" not in json.dumps(planner.contexts[0])
    assert planner.contexts[1]["story_so_far"] == ["Robots learn by trying again."]
    assert planner.contexts[1]["already_drawn"] == [first["additions"][0]["meaning"]]
    boxes = [batch["additions"][0]["box"] for batch in (first, second)]
    assert boxes == talk.cells((250, 250), 60)[:2]
    old = ET.parse(args.output / "01-canvas.svg").getroot()[0]
    new = ET.parse(args.output / "02-canvas.svg").getroot()[0]
    assert ET.tostring(old) == ET.tostring(new)
    for index in (1, 2):
        code = (args.output / f"{index:02d}-additions.gcode").read_text()
        robot.check_on_paper(code, (250, 250))
    assert statuses(args) == ["planned", "previewed"] * 2
    assert "__SESSION_DATA__" not in (args.output / "index.html").read_text()


def test_doodles_that_fail_checks_are_dropped_and_the_rest_kept(tmp_path, planner):
    planner.replies.append(answer("words nobody said in this talk", PHRASE))
    args = rehearsal(tmp_path, PHRASE, "--doodles", "2")
    main.run_talk(args)
    (batch,) = session(args)
    assert [item["evidence"] for item in batch["additions"]] == [PHRASE]
    assert batch["additions"][0]["box"] == talk.cells((250, 250), 60)[0]


def test_planner_failure_skips_only_that_window(tmp_path, planner):
    planner.replies.extend([talk.PlannerError("timed out"), answer(PHRASE)])
    args = rehearsal(tmp_path, " ".join([PHRASE] * 600))
    main.run_talk(args)
    first, second = session(args)
    assert first["additions"] == [] and len(second["additions"]) == 1
    assert not (args.output / "01-additions.gcode").exists()
    assert statuses(args)[0] == "planner-failed"


def test_doodles_too_slow_for_the_interval_are_dropped(tmp_path, planner):
    args = rehearsal(tmp_path, PHRASE, "--feed", "1")
    main.run_talk(args)
    (batch,) = session(args)
    assert batch["additions"] == [] and batch["code"] == ""


def test_robot_draws_each_batch_once_in_order(tmp_path, planner):
    args = rehearsal(tmp_path, " ".join([PHRASE] * 600))
    received = []
    talk.run(args, received.append)
    assert received == [
        (args.output / f"{index:02d}-additions.gcode").read_text() for index in (1, 2)
    ]
    assert statuses(args) == ["planned", "drawn"] * 2


def test_full_canvas_ends_the_session(tmp_path, planner):
    args = rehearsal(tmp_path, " ".join([PHRASE] * 1500), "--paper", "130x130")
    main.run_talk(args)
    assert len(talk.cells((130, 130), 60)) == 4
    assert len(session(args)) == 4


def test_execute_requires_calibration_before_anything_is_written(tmp_path, planner):
    args = rehearsal(tmp_path, PHRASE, "--execute")
    with pytest.raises(ValueError, match="--origin"):
        main.run_talk(args)
    assert not args.output.exists()


@pytest.mark.parametrize("paper", [(250, 250), (210, 297), (420, 297)])
def test_doodle_cells_are_square_on_paper_and_fit_the_layout_checks(paper):
    boxes = talk.cells(paper, 60)
    for x, y, width, height in boxes:
        assert width * paper[0] == pytest.approx(height * paper[1])
        assert min(x, y) >= storyboard.MARGIN
        assert max(x + width, y + height) <= storyboard.GRID - storyboard.MARGIN
    assert boxes == sorted(boxes, key=lambda box: (box[1], box[0]))
    for cell in (5, 1000):
        with pytest.raises(ValueError):
            talk.cells(paper, cell)


def test_live_speech_is_sealed_by_capture_time(tmp_path, monkeypatch):
    monkeypatch.setattr(talk, "GRACE_SECONDS", 0.05)

    def source(args, stop):
        yield "robots  learn", time.monotonic()
        yield "from failures", time.monotonic()
        stop.wait(0.6)
        yield "the last idea", time.monotonic()

    monkeypatch.setattr(speech, "transcripts", source)
    windows = list(
        talk.heard_windows(SimpleNamespace(output=tmp_path), 0.3, threading.Event())
    )
    assert [w.text for w in windows] == ["robots learn from failures", "the last idea"]
    assert windows[0].end_seconds == pytest.approx(0.3)
    assert [w.partial for w in windows] == [False, True]
    journal = (tmp_path / "transcript.txt").read_text().splitlines()
    assert journal == ["robots  learn", "from failures", "the last idea"]


def test_ctrl_c_while_listening_doodles_the_last_part(tmp_path, monkeypatch):
    def source(args, stop):
        yield "closing thoughts", time.monotonic()
        stop.wait(5)

    real_collect, calls = talk.collect, []

    def interrupted_once(segments, pending, deadline):
        calls.append(deadline)
        if len(calls) == 1:
            real_collect(segments, pending, time.monotonic() + 0.2)
            raise KeyboardInterrupt
        return real_collect(segments, pending, deadline)

    monkeypatch.setattr(speech, "transcripts", source)
    monkeypatch.setattr(talk, "collect", interrupted_once)
    windows = list(
        talk.heard_windows(SimpleNamespace(output=tmp_path), 600, threading.Event())
    )
    assert [(w.text, w.partial) for w in windows] == [("closing thoughts", True)]


def test_codex_plans_read_only_from_an_empty_folder(monkeypatch):
    seen = {}

    def run(command, **kwargs):
        def option(name):
            return command[command.index(name) + 1]

        seen["command"], seen["stdin"] = command, kwargs["input"]
        seen["files"] = sorted(path.name for path in Path(option("--cd")).iterdir())
        seen["schema"] = json.loads(Path(option("--output-schema")).read_text())
        Path(option("--output-last-message")).write_text(json.dumps(answer()))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(talk.subprocess, "run", run)
    assert talk.ask_codex("the prompt", talk.answer_schema(2), None) == answer()
    command = seen["command"]
    assert command[:2] == ["codex", "exec"] and command[-1] == "-"
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in command and "--model" not in command
    assert seen["stdin"] == "the prompt" and seen["files"] == ["schema.json"]

    def strict(node):
        if isinstance(node, list):
            for value in node:
                strict(value)
        elif isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert sorted(node["required"]) == sorted(node["properties"])
            for value in node.values():
                strict(value)

    strict(seen["schema"])


@pytest.mark.parametrize("outcome", ["exit", "timeout", "not json"])
def test_codex_failures_become_planner_errors(monkeypatch, outcome):
    def run(command, **kwargs):
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, talk.PLANNER_TIMEOUT_S)
        answer_file = Path(command[command.index("--output-last-message") + 1])
        answer_file.write_text(outcome)
        returncode = 1 if outcome == "exit" else 0
        return subprocess.CompletedProcess(command, returncode, "", "expired")

    monkeypatch.setattr(talk.subprocess, "run", run)
    with pytest.raises(talk.PlannerError):
        talk.ask_codex("prompt", talk.answer_schema(1), None)
