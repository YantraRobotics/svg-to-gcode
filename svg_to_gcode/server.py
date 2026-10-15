from __future__ import annotations

import itertools
import json
import math
import mimetypes
import tempfile
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from xml.etree.ElementTree import ParseError

from PIL import UnidentifiedImageError

import conveyor
from robot_interface import CMD_PORT, DEFAULT_HOST, PUB_PORT, Shell
from svg_to_gcode import gcode, image, pens, robot

MIN_PORT = 1
MAX_PORT = 65535
DEFAULT_BIND_HOST = "0.0.0.0"
MAX_BODY_BYTES = 50 * 1024 * 1024
DEFAULT_RESOLUTION = 400
DEFAULT_SMOOTHING = 5
DEFAULT_SPECKLE = 8.0
DEFAULT_CENTERLINE_RESOLUTION = 1536
DEFAULT_CENTERLINE_SMOOTHING = 1
DEFAULT_CENTERLINE_SPECKLE = 3.0
DEFAULT_BLEND = 2
DEFAULT_BREAKS = 3
DEFAULT_PAPER = "A4"
DEFAULT_MARGIN_MM = 10.0
DEFAULT_TOLERANCE_MM = 0.1
DEFAULT_FEED_MM_MIN = 1500.0
DEFAULT_LIFT_MM = 5.0
DEFAULT_HATCH_MM = 0.8
DEFAULT_HATCH_ANGLE_DEG = 45.0
DEFAULT_RAPID_MM_MIN = 3000.0
DEFAULT_TILT_DEG = 15.0
DEFAULT_SWAP_WAIT_S = 5.0
CORNER_COORDINATE_COUNT = 12
ORIGIN_COORDINATE_COUNT = 3
STYLES = ("fill", "tone", "one-line", "centerline")
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


@dataclass(frozen=True)
class Response:
    status: HTTPStatus
    content_type: str
    body: bytes


def query_str(params: dict[str, list[str]], name: str, default: str) -> str:
    values = params.get(name)
    if not values or values[-1] == "":
        return default
    return values[-1]


def query_float(params: dict[str, list[str]], name: str, default: float) -> float:
    raw = query_str(params, name, "")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {raw!r}")
    return value


def query_int(params: dict[str, list[str]], name: str, default: int) -> int:
    raw = query_str(params, name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {raw!r}") from None


def query_bool(params: dict[str, list[str]], name: str, default: bool) -> bool:
    raw = query_str(params, name, "").lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes"):
        return True
    if raw in ("0", "false", "no"):
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


def parse_float_list(raw: str, expected: int, name: str) -> list[float]:
    parts = raw.replace(",", " ").split()
    if len(parts) != expected:
        raise ValueError(f"{name} needs {expected} numbers, got {len(parts)}")
    try:
        values = [float(part) for part in parts]
    except ValueError:
        raise ValueError(f"{name} must contain only numbers, got {raw!r}") from None
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} numbers must be finite")
    return values


def convert_image_bytes(body: bytes, params: dict[str, list[str]]) -> str:
    if not body:
        raise ValueError("request body is empty; expected raster image bytes")
    style = query_str(params, "style", "fill")
    if style not in STYLES:
        raise ValueError(f"style must be one of {', '.join(STYLES)}, got {style!r}")
    pen_list = pens.parse_pens(query_str(params, "pens", pens.DEFAULT_PENS))
    default_resolution = (
        DEFAULT_CENTERLINE_RESOLUTION if style == "centerline" else DEFAULT_RESOLUTION
    )
    default_smoothing = (
        DEFAULT_CENTERLINE_SMOOTHING if style == "centerline" else DEFAULT_SMOOTHING
    )
    default_speckle = (
        DEFAULT_CENTERLINE_SPECKLE if style == "centerline" else DEFAULT_SPECKLE
    )
    resolution = query_int(params, "resolution", default_resolution)
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    smoothing = query_int(params, "smoothing", default_smoothing)
    if smoothing <= 0 or smoothing % 2 == 0:
        raise ValueError(f"smoothing must be a positive odd integer, got {smoothing}")
    speckle = query_float(params, "speckle", default_speckle)
    if speckle < 0:
        raise ValueError(f"speckle must not be negative, got {speckle}")
    blend = query_int(params, "blend", DEFAULT_BLEND)
    if blend <= 0:
        raise ValueError(f"blend must be positive, got {blend}")
    breaks = query_int(params, "breaks", DEFAULT_BREAKS)
    if breaks < 0:
        raise ValueError(f"breaks must not be negative, got {breaks}")

    with tempfile.TemporaryDirectory(prefix="svg-to-gcode-") as folder:
        image_path = Path(folder) / "input.img"
        image_path.write_bytes(body)
        if style == "tone":
            return image.image_to_line_screen(image_path, pen_list, resolution)
        if style == "one-line":
            return image.image_to_one_line(image_path, pen_list, resolution, breaks)
        if style == "centerline":
            return image.image_to_centerline(
                image_path, pen_list, resolution, smoothing, speckle
            )
        return image.image_to_svg(
            image_path, pen_list, resolution, smoothing, speckle, blend
        )


def looks_like_svg(body: bytes, content_type: str) -> bool:
    if "svg" in content_type.lower() or "xml" in content_type.lower():
        return True
    stripped = body.lstrip()
    return stripped.startswith(b"<")


def convert_to_gcode(
    body: bytes, params: dict[str, list[str]], content_type: str = "image/svg+xml"
) -> str:
    if not body:
        raise ValueError("request body is empty; expected SVG or raster image")
    paper = gcode.parse_paper(query_str(params, "paper", DEFAULT_PAPER))
    pen_list = pens.parse_pens(query_str(params, "pens", pens.DEFAULT_PENS))
    margin = query_float(params, "margin", DEFAULT_MARGIN_MM)
    if margin < 0:
        raise ValueError(f"margin must not be negative, got {margin}")
    tolerance = query_float(params, "tolerance", DEFAULT_TOLERANCE_MM)
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    feed = query_float(params, "feed", DEFAULT_FEED_MM_MIN)
    if not 0 < feed <= robot.MAX_FEED_MM_MIN:
        raise ValueError(
            f"feed must be in (0, {robot.MAX_FEED_MM_MIN:g}], got {feed:g}"
        )
    lift = query_float(params, "lift", DEFAULT_LIFT_MM)
    if lift <= 0:
        raise ValueError(f"lift must be positive, got {lift}")
    hatch_mm = query_float(params, "hatch", DEFAULT_HATCH_MM)
    if hatch_mm < 0 or 0 < hatch_mm < gcode.MIN_HATCH_MM:
        raise ValueError(
            f"hatch must be 0 or at least {gcode.MIN_HATCH_MM:g} mm, got {hatch_mm:g}"
        )
    hatch_angle = query_float(params, "hatch_angle", DEFAULT_HATCH_ANGLE_DEG)
    outline = query_bool(params, "outline", True)

    svg_text = (
        body.decode("utf-8")
        if looks_like_svg(body, content_type)
        else convert_image_bytes(body, params)
    )
    with tempfile.TemporaryDirectory(prefix="svg-to-gcode-") as folder:
        svg_path = Path(folder) / "input.svg"
        svg_path.write_text(svg_text, encoding="utf-8")
        layers = gcode.svg_to_layers(
            svg_path,
            paper,
            margin,
            tolerance,
            pen_list,
            hatch_mm,
            hatch_angle,
            outline,
        )
    return gcode.to_gcode(layers, feed, lift)


def resolve_placement(
    params: dict[str, list[str]], z_offset: float
) -> tuple[robot.Placement, tuple[float, float] | None]:
    origin_raw = query_str(params, "origin", "")
    corners_raw = query_str(params, "corners", "")
    if bool(origin_raw) == bool(corners_raw):
        raise ValueError("pass either origin=X,Y,Z or corners=BLX,BLY,...,TLZ")
    if origin_raw:
        origin = tuple(parse_float_list(origin_raw, ORIGIN_COORDINATE_COUNT, "origin"))
        yaw = math.radians(query_float(params, "yaw", 0.0))
        return robot.Placement(origin, yaw, z_offset), None
    if query_str(params, "yaw", ""):
        raise ValueError("yaw goes with origin; corners set the paper's yaw")
    coords = parse_float_list(corners_raw, CORNER_COORDINATE_COUNT, "corners")
    corners = [tuple(corner) for corner in itertools.batched(coords, 3)]
    placement = robot.corner_placement(corners, z_offset)
    return placement, placement.size


def run_draw_request(body: bytes, params: dict[str, list[str]]) -> dict[str, object]:
    if not body:
        raise ValueError("request body is empty; expected G-code")
    text = body.decode("utf-8")
    z_offset = query_float(params, "z_offset", 0.0)
    placement, size = resolve_placement(params, z_offset)
    if size is not None:
        robot.check_on_paper(text, size)
    rapid = query_float(params, "rapid", DEFAULT_RAPID_MM_MIN)
    if not 0 < rapid <= robot.MAX_FEED_MM_MIN:
        raise ValueError(
            f"rapid must be in (0, {robot.MAX_FEED_MM_MIN:g}], got {rapid:g}"
        )
    tilt_deg = query_float(params, "tilt", DEFAULT_TILT_DEG)
    if not 0 <= tilt_deg <= robot.MAX_TILT_DEG:
        raise ValueError(
            f"tilt must be in [0, {robot.MAX_TILT_DEG:g}], got {tilt_deg:g}"
        )
    tool_raw = query_str(params, "tool", "")
    tool = query_int(params, "tool", 0) if tool_raw else None
    if tool is not None and tool <= 0:
        raise ValueError(f"tool must be positive, got {tool}")
    dry_run = query_bool(params, "dry_run", False)
    pen_frames = robot.parse_pen_frames(query_str(params, "frames", ""))

    steps = robot.parse_gcode(text, (0.0, 0.0, 0.0), rapid)
    if pen_frames:
        robot.pen_runs(steps, pen_frames, tool)
    moves = [step for step in steps if isinstance(step, robot.Move)]
    start = next((m.end for m in moves if m.start[:2] != m.end[:2]), moves[0].start)
    steps = robot.steps_to_draw(text, start, rapid, tool)
    moves = [step for step in steps if isinstance(step, robot.Move)]
    segments, hops = robot.plan_motion(steps, tilt_deg)
    low, high, seconds = robot.summarize(segments, placement)
    changes = sum(isinstance(segment, robot.PenChange) for segment in segments)
    summary: dict[str, object] = {
        "moves": len(moves),
        "pen_changes": changes,
        "hops": hops,
        "estimated_seconds": round(seconds, 2),
        "bounds_mm": {"min": list(low), "max": list(high)},
        "dry_run": dry_run,
        "pen_frames": pen_frames,
    }
    if dry_run:
        return summary

    if not query_str(params, "pen_offset", ""):
        raise ValueError("pen_offset is required when dry_run is false")
    pen_offset = query_float(params, "pen_offset", 0.0)
    swap_wait = query_float(params, "swap_wait", DEFAULT_SWAP_WAIT_S)
    if swap_wait < 0:
        raise ValueError(f"swap_wait must not be negative, got {swap_wait}")
    host = query_str(params, "host", DEFAULT_HOST)
    pub = query_str(params, "pub", "") or f"tcp://{host}:{PUB_PORT}"
    cmd = query_str(params, "cmd", "") or f"tcp://{host}:{CMD_PORT}"
    worst = robot.draw(
        Shell(pub, cmd),
        text,
        placement,
        pen_offset,
        rapid,
        tilt_deg,
        swap_wait,
        tool,
        pen_frames,
    )
    summary["worst_error_mm"] = round(worst, 2)
    return summary


def run_point_down_request(params: dict[str, list[str]]) -> dict[str, object]:
    host = query_str(params, "host", DEFAULT_HOST)
    pub = query_str(params, "pub", "") or f"tcp://{host}:{PUB_PORT}"
    cmd = query_str(params, "cmd", "") or f"tcp://{host}:{CMD_PORT}"
    return {"turned_deg": round(robot.point_down(Shell(pub, cmd)), 1)}


def run_frames_request(params: dict[str, list[str]]) -> dict[str, object]:
    host = query_str(params, "host", DEFAULT_HOST)
    pub = query_str(params, "pub", "") or f"tcp://{host}:{PUB_PORT}"
    cmd = query_str(params, "cmd", "") or f"tcp://{host}:{CMD_PORT}"
    tips = Shell(pub, cmd).get_target_frame()
    if tips is None:
        raise RuntimeError(
            f"no target frame list from {cmd}; is the controller serving, and does "
            "it support GET_TARGET_FRAME?"
        )
    return tips


def run_teach_request(params: dict[str, list[str]]) -> dict[str, object]:
    pen_offset = query_float(params, "pen_offset", 0.0)
    host = query_str(params, "host", DEFAULT_HOST)
    pub = query_str(params, "pub", "") or f"tcp://{host}:{PUB_PORT}"
    cmd = query_str(params, "cmd", "") or f"tcp://{host}:{CMD_PORT}"
    shell = Shell(pub, cmd)
    if pen_offset:
        robot.check_tip_link(shell)
    x, y, z = robot.pen_tip_position(shell, pen_offset)
    return {"origin": [round(x, 1), round(y, 1), round(z, 1)]}


def run_conveyor_request(params: dict[str, list[str]]) -> dict[str, object]:
    cmd = query_str(params, "cmd", "STATUS").upper()
    if cmd not in conveyor.COMMANDS:
        raise ValueError(
            f"cmd must be one of {', '.join(conveyor.COMMANDS)}, got {cmd!r}"
        )
    serial_port = query_str(params, "port", "") or None
    baud = query_int(params, "baud", conveyor.DEFAULT_BAUD)
    if baud <= 0:
        raise ValueError(f"baud must be positive, got {baud}")
    with conveyor.Conveyor(serial_port, baud) as conv:
        ok, fields = conv.send(cmd)
        port_name = conv.ser.port
    if not ok:
        raise ValueError(" ".join(fields) or f"conveyor rejected {cmd}")
    state = fields[0] if len(fields) >= 1 else "UNKNOWN"
    direction = fields[1] if len(fields) >= 2 else "FWD"
    return {
        "running": state == "RUNNING",
        "state": state,
        "direction": direction,
        "port": port_name,
    }


def json_response(status: HTTPStatus, payload: dict[str, object]) -> Response:
    return Response(
        status,
        "application/json; charset=utf-8",
        (json.dumps(payload) + "\n").encode("utf-8"),
    )


def static_response(path: str) -> Response | None:
    if not WEB_DIST.is_dir():
        return None
    relative = "index.html" if path == "/" else path.lstrip("/")
    candidate = (WEB_DIST / relative).resolve()
    if candidate.is_relative_to(WEB_DIST) and candidate.is_file():
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        return Response(HTTPStatus.OK, mime, candidate.read_bytes())
    return None


def dispatch(method: str, target: str, body: bytes, content_type: str = "") -> Response:
    parsed = urlsplit(target)
    path = parsed.path.rstrip("/") or "/"
    params = parse_qs(parsed.query, keep_blank_values=False)

    if method == "GET":
        if path == "/health":
            return json_response(HTTPStatus.OK, {"status": "ok"})
        if path == "/conveyor":
            try:
                return json_response(HTTPStatus.OK, run_conveyor_request(params))
            except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
                return json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        if path == "/frames":
            try:
                return json_response(HTTPStatus.OK, run_frames_request(params))
            except RuntimeError as exc:
                return json_response(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
        served = static_response(path)
        if served is not None:
            return served
        return json_response(
            HTTPStatus.NOT_FOUND, {"error": f"unknown endpoint {path}"}
        )
    if method != "POST":
        return json_response(
            HTTPStatus.METHOD_NOT_ALLOWED, {"error": f"unsupported method {method}"}
        )
    try:
        if path == "/image":
            svg_text = convert_image_bytes(body, params)
            return Response(
                HTTPStatus.OK,
                "image/svg+xml; charset=utf-8",
                svg_text.encode("utf-8"),
            )
        if path in ("/gcode", "/convert"):
            code = convert_to_gcode(body, params, content_type)
            return Response(
                HTTPStatus.OK,
                "text/plain; charset=utf-8",
                code.encode("utf-8"),
            )
        if path == "/draw":
            result = run_draw_request(body, params)
            return json_response(HTTPStatus.OK, result)
        if path == "/teach":
            result = run_teach_request(params)
            return json_response(HTTPStatus.OK, result)
        if path == "/point_down":
            return json_response(HTTPStatus.OK, run_point_down_request(params))
        if path == "/conveyor":
            result = run_conveyor_request(params)
            return json_response(HTTPStatus.OK, result)
    except (
        ValueError,
        RuntimeError,
        TimeoutError,
        ParseError,
        UnicodeDecodeError,
        UnidentifiedImageError,
        OSError,
        robot.DrawError,
    ) as exc:
        return json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return json_response(HTTPStatus.NOT_FOUND, {"error": f"unknown endpoint {path}"})


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def _handle(self, method: str) -> None:
        length_header = self.headers.get("Content-Length", "0")
        try:
            length = int(length_header)
        except ValueError:
            self._send(
                json_response(
                    HTTPStatus.BAD_REQUEST, {"error": "invalid Content-Length"}
                )
            )
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self._send(
                json_response(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": f"body exceeds {MAX_BODY_BYTES} bytes"},
                )
            )
            return
        body = self.rfile.read(length) if length else b""
        content_type = self.headers.get("Content-Type", "")
        self._send(dispatch(method, self.path, body, content_type))

    def _send(self, response: Response) -> None:
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.body)))
        self.end_headers()
        self.wfile.write(response.body)

    def log_message(self, format: str, *args: object) -> None:
        return


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    if not 0 <= port <= MAX_PORT:
        raise ValueError(f"port must be between 0 and {MAX_PORT}, got {port}")
    return ThreadingHTTPServer((host, port), RequestHandler)


def serve(host: str, port: int) -> int:
    if not MIN_PORT <= port <= MAX_PORT:
        raise ValueError(f"port must be between {MIN_PORT} and {MAX_PORT}, got {port}")
    with make_server(host, port) as httpd:
        bound_host, bound_port = httpd.server_address[:2]
        print(f"listening on http://{bound_host}:{bound_port}", flush=True)
        httpd.serve_forever()
    return 0
