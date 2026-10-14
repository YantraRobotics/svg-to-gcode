import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from PIL import Image, ImageDraw

import main
from svg_to_gcode import server

SQUARE_SVG = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect x="10" y="10" width="80" height="80" fill="none" stroke="#231F20"/>
</svg>"""


def sample_png_bytes() -> bytes:
    card = Image.new("RGB", (80, 80), "white")
    draw = ImageDraw.Draw(card)
    draw.rectangle([15, 15, 65, 65], fill=(35, 31, 32))
    buffer = io.BytesIO()
    card.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def base_url():
    with server.make_server("127.0.0.1", 0) as httpd:
        host, port = httpd.server_address[:2]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://{host}:{port}"
        finally:
            httpd.shutdown()
            thread.join(timeout=2)


def test_cli_parses_port_in_all_forms():
    parser = main.build_parser()
    top = parser.parse_args(["--port", "8123"])
    assert top.port == 8123

    sub_pos = parser.parse_args(["serve", "8123"])
    assert sub_pos.port == 8123

    sub_flag = parser.parse_args(["serve", "--port", "8123"])
    assert sub_flag.port_flag == 8123


def test_health_and_conversion_endpoints(base_url):
    with urlopen(f"{base_url}/health") as response:
        assert response.status == 200
        assert json.loads(response.read()) == {"status": "ok"}

    png = sample_png_bytes()
    image_req = Request(
        f"{base_url}/image?resolution=80",
        data=png,
        headers={"Content-Type": "image/png"},
        method="POST",
    )
    with urlopen(image_req) as response:
        svg_body = response.read().decode("utf-8")
        assert "<svg" in svg_body

    gcode_req = Request(
        f"{base_url}/gcode?paper=250x250&feed=1000&lift=2",
        data=SQUARE_SVG,
        headers={"Content-Type": "image/svg+xml"},
        method="POST",
    )
    with urlopen(gcode_req) as response:
        gcode_text = response.read().decode("utf-8")
        assert "G21" in gcode_text
        assert "M2" in gcode_text

    draw_req = Request(
        f"{base_url}/draw?origin=-125,-460,30.6&dry_run=true&tilt=0",
        data=gcode_text.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
        method="POST",
    )
    with urlopen(draw_req) as response:
        summary = json.loads(response.read())
        assert summary["dry_run"] is True
        assert summary["moves"] > 0
        assert summary["pen_changes"] == 1


def test_invalid_requests_return_400(base_url):
    bad_req = Request(f"{base_url}/gcode", data=b"", method="POST")
    with pytest.raises(HTTPError) as exc_info:
        urlopen(bad_req)
    assert exc_info.value.code == 400
    payload = json.loads(exc_info.value.read())
    assert "empty" in payload["error"]


def test_serves_built_web_app(base_url):
    with urlopen(f"{base_url}/") as response:
        assert response.status == 200
        html = response.read().decode("utf-8")
        assert "Owl 6.9 Plotter Workbench" in html


def test_centerline_preserves_thin_line_art(base_url):
    card = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(card)
    draw.line([(20, 100), (180, 100)], fill=(35, 31, 32), width=3)
    buffer = io.BytesIO()
    card.save(buffer, format="PNG")

    req = Request(
        f"{base_url}/image?style=centerline",
        data=buffer.getvalue(),
        headers={"Content-Type": "image/png"},
        method="POST",
    )
    with urlopen(req) as response:
        svg_body = response.read().decode("utf-8")
        assert "<path" in svg_body


def test_conveyor_endpoint_and_cli(base_url, monkeypatch):
    parser = main.build_parser()
    parsed = parser.parse_args(
        ["conveyor", "--serial-port", "/dev/cu.usbserial-0001", "fwd", "start"]
    )
    assert parsed.command == "conveyor"
    assert parsed.serial_port == "/dev/cu.usbserial-0001"
    assert parsed.commands == ["fwd", "start"]

    sent: list[str] = []

    class FakeConveyor:
        def __init__(self, port=None, baud=115200):
            self.ser = type("FakeSer", (), {"port": port or "/dev/cu.usbserial-0001"})()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def send(self, cmd: str):
            sent.append(cmd)
            if cmd == "START":
                return True, ["RUNNING", "FWD"]
            if cmd == "STOP":
                return True, ["STOPPED", "FWD"]
            if cmd == "REV":
                return True, ["RUNNING", "REV"]
            return True, ["STOPPED", "FWD"]

    monkeypatch.setattr(server.conveyor, "Conveyor", FakeConveyor)

    start_req = Request(f"{base_url}/conveyor?cmd=start", data=b"", method="POST")
    with urlopen(start_req) as response:
        payload = json.loads(response.read())
        assert payload == {
            "running": True,
            "state": "RUNNING",
            "direction": "FWD",
            "port": "/dev/cu.usbserial-0001",
        }

    stop_req = Request(f"{base_url}/conveyor?cmd=stop", data=b"", method="POST")
    with urlopen(stop_req) as response:
        payload = json.loads(response.read())
        assert payload["running"] is False
        assert payload["state"] == "STOPPED"

    assert sent == ["START", "STOP"]
