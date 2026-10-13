import io
import json
import math
import queue
import threading
import time
import wave
from itertools import pairwise
from urllib.error import URLError

import pytest

import main
from svg_to_gcode import doodles, live, robot, speech


@pytest.mark.parametrize("name", doodles.MOTIFS)
def test_doodles_are_bounded_and_budgeted(name):
    canvas = doodles.Canvas((250, 250))
    strokes = canvas.place(name, 1000, 8)
    length = sum(math.dist(a, b) for line in strokes for a, b in pairwise(line))
    assert length / 1000 * 60 <= 8 + 1e-9
    assert all(10 <= x <= 240 and 10 <= y <= 240 for line in strokes for x, y in line)
    code = doodles.drawing_code(strokes, 1000, 2)
    robot.check_on_paper(code, canvas.paper)
    steps = robot.parse_gcode(code, (0, 0, 2), 3000)
    assert not any(isinstance(step, robot.PenChange) for step in steps)
    assert steps[-1].end[2] == 2


def test_full_canvas_has_disjoint_cells_and_preserves_previous_strokes():
    canvas = doodles.Canvas((100, 100), 10, 40)
    bounds = []
    for _ in range(canvas.capacity):
        old = list(canvas.strokes)
        strokes = canvas.place("tree", 1000, 8)
        assert canvas.strokes[: len(old)] == old
        points = [p for line in strokes for p in line]
        bounds.append(
            (
                min(x for x, y in points),
                max(x for x, y in points),
                min(y for x, y in points),
                max(y for x, y in points),
            )
        )
    for i, a in enumerate(bounds):
        for b in bounds[i + 1 :]:
            assert a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]
    with pytest.raises(ValueError, match="canvas full"):
        canvas.place("tree", 1000, 8)


def test_small_canvas_and_unreadable_budget_rejected():
    with pytest.raises(ValueError, match="at least one cell"):
        doodles.Canvas((20, 20))
    with pytest.raises(ValueError, match="budget too small"):
        doodles.Canvas((250, 250)).place("tree", 1000, 0.01)


def test_keywords_ignore_filler_and_recent_repetition():
    assert doodles.keyword_motif("Um, thanks everyone", []) is None
    assert doodles.keyword_motif("Grow forests and trees", []) == "tree"
    assert doodles.keyword_motif("Grow forests and trees", ["tree"]) is None


@pytest.mark.parametrize(
    "content, expected",
    [
        ('{"motif":"tree"}', "tree"),
        ('{"motif":"skip"}', None),
        ('{"motif":"G0 X9999"}', "sun"),
        ('{"motif":[]}', "sun"),
        ("not json", "sun"),
    ],
)
def test_model_output_never_controls_geometry(monkeypatch, content, expected):
    def response(request, timeout):
        assert request.full_url == "http://127.0.0.1:11434/api/chat"
        assert json.loads(request.data)["stream"] is False
        return io.BytesIO(json.dumps({"message": {"content": content}}).encode())

    monkeypatch.setattr(live, "urlopen", response)
    assert live.select_motif("solar energy", [], [], "local-model") == expected


def test_model_failure_uses_offline_fallback(monkeypatch):
    def unavailable(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(live, "urlopen", unavailable)
    assert live.select_motif("plant trees", [], [], "local-model") == "tree"


def test_queue_replaces_old_work_and_retains_final_item():
    slot, errors = queue.Queue(maxsize=1), queue.Queue()
    stop, done = threading.Event(), threading.Event()
    source = ((text, time.monotonic()) for text in ["trees", "robots", "books"])
    live.plan(source, "keywords", slot, stop, done, errors)
    assert done.is_set()
    assert errors.empty()
    assert slot.qsize() == 1
    assert slot.get().motif == "book"


def arguments(tmp_path):
    return main.build_parser().parse_args(
        [
            "live",
            str(tmp_path / "session"),
            "--transcript",
            "examples/talk.txt",
            "--preview-speed",
            "1000",
        ]
    )


def test_pipeline_writes_preview_and_valid_gcode_without_controller(
    tmp_path, monkeypatch
):
    args = arguments(tmp_path)

    def source(args, stop):
        yield "plant trees", time.monotonic()

    monkeypatch.setattr(speech, "transcripts", source)
    assert main.run_live(args) == 0
    assert "polyline" in (args.output / "canvas.svg").read_text()
    code = (args.output / "canvas.gcode").read_text()
    robot.check_on_paper(code, args.paper)
    events = [
        json.loads(line)
        for line in (args.output / "events.jsonl").read_text().splitlines()
    ]
    assert [e["status"] for e in events] == ["planned", "previewed"]
    with pytest.raises(FileExistsError):
        main.run_live(args)


def test_stale_ideas_never_reach_robot(tmp_path, monkeypatch):
    def source(args, stop):
        yield "trees", time.monotonic() - 100

    args = arguments(tmp_path)
    monkeypatch.setattr(speech, "transcripts", source)
    received = []
    assert live.run(args, received.append) == 0
    assert received == []
    assert not (args.output / "canvas.gcode").exists()


def test_worker_failure_is_reported(tmp_path, monkeypatch):
    def source(args, stop):
        raise RuntimeError("microphone disconnected")
        yield

    monkeypatch.setattr(speech, "transcripts", source)
    with pytest.raises(RuntimeError, match="microphone disconnected"):
        live.run(arguments(tmp_path))


def test_robot_failure_stops_session_and_is_recorded(tmp_path, monkeypatch):
    def source(args, stop):
        yield "trees", time.monotonic()

    def draw(code):
        raise robot.DrawError("tracking failed")

    args = arguments(tmp_path)
    monkeypatch.setattr(speech, "transcripts", source)
    with pytest.raises(robot.DrawError, match="tracking failed"):
        live.run(args, draw)
    events = [
        json.loads(line)
        for line in (args.output / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["status"] == "interrupted"


def test_execute_requires_calibration(tmp_path):
    args = arguments(tmp_path)
    args.execute = True
    with pytest.raises(ValueError, match="--origin"):
        main.run_live(args)
    assert not args.output.exists()


def test_full_canvas_stops_before_second_drawing(tmp_path, monkeypatch):
    def source(args, stop):
        yield "trees", time.monotonic()
        stop.wait(1)
        yield "robots", time.monotonic()

    args = arguments(tmp_path)
    args.paper = (60, 60)
    monkeypatch.setattr(speech, "transcripts", source)
    received = []
    live.run(args, received.append)
    assert len(received) == 1


def test_new_speech_replaces_pending_work_while_robot_draws(tmp_path, monkeypatch):
    drawing = threading.Event()
    planned = threading.Event()

    def source(args, stop):
        yield "trees", time.monotonic()
        assert drawing.wait(2)
        yield "robots", time.monotonic()
        yield "books", time.monotonic()
        planned.set()

    received = []

    def draw(code):
        received.append(code)
        drawing.set()
        assert planned.wait(2)

    args = arguments(tmp_path)
    monkeypatch.setattr(speech, "transcripts", source)
    live.run(args, draw)
    assert len(received) == 2
    events = [
        json.loads(line)
        for line in (args.output / "events.jsonl").read_text().splitlines()
    ]
    assert [event["motif"] for event in events if event["status"] == "drawn"] == [
        "tree",
        "book",
    ]


def test_wav_input_checks_format_and_keeps_tail(tmp_path):
    path = tmp_path / "test.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes(100))
    chunks = list(speech.wav_chunks(path, 4, threading.Event()))
    assert len(chunks) == 1 and len(chunks[0][0]) == 100
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(bytes(100))
    with pytest.raises(ValueError, match="16 kHz"):
        list(speech.wav_chunks(path, 4, threading.Event()))


def test_whisper_silence_does_not_launch_process(tmp_path, monkeypatch):
    model = tmp_path / "model.bin"
    model.touch()
    monkeypatch.setattr(speech.shutil, "which", lambda _: "/fake/whisper-cli")
    whisper = speech.Whisper("whisper-cli", model, "en")
    assert whisper.transcribe(bytes(32000), threading.Event()) == ""


def test_microphone_keeps_recording_and_only_retains_latest_chunk(monkeypatch):
    import sys
    from types import SimpleNamespace

    stop = threading.Event()
    streams = []

    class Stream:
        active = True

        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]
            streams.append(self)

        def __enter__(self):
            self.callback(bytes(32000), 16000, None, None)
            return self

        def __exit__(self, *args):
            self.active = False

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(RawInputStream=Stream, PortAudioError=RuntimeError),
    )
    source = speech.microphone_chunks(None, 1, stop)
    assert len(next(source)[0]) == 32000
    streams[0].callback(b"\x01\x00" * 32000, 32000, None, None)
    assert next(source)[0] == b"\x01\x00" * 16000
    source.close()
    assert not streams[0].active
