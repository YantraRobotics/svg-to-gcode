import argparse
import itertools
import logging
import math
import sys
from collections.abc import Callable
from pathlib import Path
from xml.etree.ElementTree import ParseError

import prepare_owl69_model
from robot_interface import Shell, add_endpoint_arguments, resolve_endpoints
from svg_to_gcode import gcode, image, pens, robot, viewer

CORNER_METAVARS = tuple(
    f"{corner}{axis}" for corner in ("BL", "BR", "TR", "TL") for axis in "XYZ"
)


def finite(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"{text} is not a finite number")
    return value


def positive(text: str) -> float:
    value = finite(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{text} must be positive")
    return value


def non_negative(text: str) -> float:
    value = finite(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text} must not be negative")
    return value


def hatch_spacing(text: str) -> float:
    value = non_negative(text)
    if 0 < value < gcode.MIN_HATCH_MM:
        raise argparse.ArgumentTypeError(
            f"{text} must be 0 or at least {gcode.MIN_HATCH_MM:g} mm"
        )
    return value


def feed_rate(text: str) -> float:
    value = positive(text)
    if value > robot.MAX_FEED_MM_MIN:
        raise argparse.ArgumentTypeError(
            f"{text} exceeds the {robot.MAX_FEED_MM_MIN:g} mm/min limit"
        )
    return value


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{text} must be a positive whole number")
    return value


def non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text} must not be negative")
    return value


def odd_size(text: str) -> int:
    value = positive_int(text)
    if value % 2 == 0:
        raise argparse.ArgumentTypeError(f"{text} must be odd")
    return value


def tilt_angle(text: str) -> float:
    value = non_negative(text)
    if value > robot.MAX_TILT_DEG:
        raise argparse.ArgumentTypeError(
            f"{text} exceeds the {robot.MAX_TILT_DEG:g} degree limit"
        )
    return value


def paper_size(text: str) -> tuple[float, float]:
    try:
        return gcode.parse_paper(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def pen_set(text: str) -> list[pens.Pen]:
    try:
        return pens.parse_pens(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def run_image(args: argparse.Namespace) -> int:
    if args.tone:
        svg = image.image_to_line_screen(args.image, args.pens, args.resolution)
    elif args.one_line:
        svg = image.image_to_one_line(
            args.image, args.pens, args.resolution, args.breaks
        )
    elif args.centerline:
        svg = image.image_to_centerline(
            args.image, args.pens, args.resolution, args.smoothing, args.speckle
        )
    else:
        svg = image.image_to_svg(
            args.image,
            args.pens,
            args.resolution,
            args.smoothing,
            args.speckle,
            args.blend,
        )
    args.output.write_text(svg)
    print(f"wrote {args.output}")
    return 0


def run_gcode(args: argparse.Namespace) -> int:
    layers = gcode.svg_to_layers(
        args.svg,
        args.paper,
        args.margin,
        args.tolerance,
        args.pens,
        args.hatch,
        args.hatch_angle,
        args.outline,
    )
    args.output.write_text(gcode.to_gcode(layers, args.feed, args.lift))
    counts = ", ".join(f"{len(layer.strokes)} {layer.pen.name}" for layer in layers)
    print(f"wrote {counts} strokes to {args.output}")
    return 0


def run_teach(args: argparse.Namespace) -> int:
    shell = Shell(*resolve_endpoints(args))
    if args.pen_offset:
        robot.check_tip_link(shell)
    x, y, z = robot.pen_tip_position(shell, args.pen_offset)
    print(f"--origin {x:.1f} {y:.1f} {z:.1f}")
    return 0


def paper_placement(
    args: argparse.Namespace, z_offset: float
) -> tuple[robot.Placement, tuple[float, float] | None]:
    """Where the paper lies, and its size when its corners were given."""
    if args.corners is None:
        yaw = math.radians(args.yaw or 0.0)
        return robot.Placement(tuple(args.origin), yaw, z_offset), None
    if args.yaw is not None:
        raise ValueError("--yaw goes with --origin; --corners set the paper's yaw")
    corners = list(itertools.batched(args.corners, 3))
    placement = robot.corner_placement(corners, z_offset)
    width, height = placement.size
    angles = ", ".join(f"{angle:.0f}" for angle in robot.corner_angles(corners))
    heights = [corner[2] for corner in corners]
    print(
        f"paper {width:.1f} x {height:.1f} mm between the corners; they meet at "
        f"{angles} deg and sit {min(heights):.1f} .. {max(heights):.1f} mm high"
    )
    return placement, placement.size


def run_draw(args: argparse.Namespace) -> int:
    text = args.gcode.read_text()
    placement, size = paper_placement(args, args.z_offset)
    if size is not None:
        robot.check_on_paper(text, size)
    if args.tool is not None:
        pen = robot.tool_table(text).get(args.tool)
        print(f"drawing only T{args.tool}" + (f" {pen.name} {pen.hex}" if pen else ""))
    if args.dry_run:
        # The arm's start pose is unknown offline, so begin at the first XY travel.
        steps = robot.parse_gcode(text, (0.0, 0.0, 0.0), args.rapid)
        moves = [step for step in steps if isinstance(step, robot.Move)]
        start = next((m.end for m in moves if m.start[:2] != m.end[:2]), moves[0].start)
        steps = robot.steps_to_draw(text, start, args.rapid, args.tool)
        moves = [step for step in steps if isinstance(step, robot.Move)]
        segments, hops = robot.plan_motion(steps, args.tilt)
        low, high, seconds = robot.summarize(segments, placement)
        for axis, lo, hi in zip("xyz", low, high):
            print(f"pen tip {axis}: {lo:8.1f} .. {hi:8.1f} mm (base frame)")
        changes = sum(isinstance(segment, robot.PenChange) for segment in segments)
        print(
            f"{len(moves)} moves, {changes} pen changes and {hops} hops, about "
            f"{seconds / 60:.1f} min of motion plus the approach from the arm's "
            "current pose and the pen swaps"
        )
        return 0
    shell = Shell(*resolve_endpoints(args))
    worst = robot.draw(
        shell,
        text,
        placement,
        args.pen_offset,
        args.rapid,
        args.tilt,
        args.swap_wait,
        args.tool,
    )
    print(f"done; largest pen-tip error while drawing {worst:.2f} mm")
    return 0


def run_view(args: argparse.Namespace) -> int:
    placement, size = paper_placement(args, 0.0)
    paper = size or args.paper
    pub, cmd = resolve_endpoints(args)
    served = prepare_owl69_model.fetch_served_model(cmd)
    bundle = prepare_owl69_model.prepare_from_controller(cmd, served=served)
    urdf, _ = served
    model = viewer.build_scene(bundle, urdf, paper, placement, args.pen_offset)
    viewer.show(Shell(pub, cmd), model, placement, paper)
    return 0


def robot_drawer(
    args: argparse.Namespace, swap_wait: float | None
) -> Callable[[str], None] | None:
    """Draw jobs on the controller with --execute; None leaves them as previews."""
    if not args.execute:
        return None
    if args.origin is None or args.pen_offset is None:
        raise ValueError("--execute requires --origin X Y Z and --pen-offset")
    placement = robot.Placement(tuple(args.origin), math.radians(args.yaw), 0.0)
    shell = Shell(*resolve_endpoints(args))

    def draw(code: str) -> None:
        robot.draw(shell, code, placement, args.pen_offset, args.rapid, 0.0, swap_wait)

    return draw


def run_live(args: argparse.Namespace) -> int:
    from svg_to_gcode import live

    if args.transcript is None and args.whisper_model is None:
        raise ValueError("microphone and --wav input require --whisper-model")
    return live.run(args, robot_drawer(args, None))


def run_talk(args: argparse.Namespace) -> int:
    from svg_to_gcode import talk

    if args.transcript is None and args.whisper_model is None:
        raise ValueError("microphone and --wav input require --whisper-model")
    return talk.run(args, robot_drawer(args, args.swap_wait))


def run_storyboard(args: argparse.Namespace) -> int:
    from svg_to_gcode import storyboard

    report = storyboard.render(
        args.transcript,
        args.plans,
        args.output,
        args.interval_seconds,
        args.wpm,
        args.paper,
        args.feed,
        args.rapid,
        args.lift,
        args.swap_seconds,
    )
    print(f"{report['doodles']} doodles in {len(report['batches'])} immutable batches")
    print(f"Open {(args.output / 'index.html').resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw SVG images on paper with the Owl 6.9 arm."
    )
    commands = parser.add_subparsers(required=True)

    paper = argparse.ArgumentParser(add_help=False)
    paper.add_argument(
        "--paper",
        type=paper_size,
        default="A4",
        help="A3, A4, A5, LETTER or WIDTHxHEIGHT in mm (default A4 portrait)",
    )

    pen_choice = argparse.ArgumentParser(add_help=False)
    pen_choice.add_argument(
        "--pens",
        type=pen_set,
        default=pens.DEFAULT_PENS,
        help="the pens on hand as NAME=#RRGGBB,... (default CMYK)",
    )

    to_svg = commands.add_parser(
        "image",
        parents=[pen_choice],
        help="trace a raster image into an SVG with one layer per pen",
    )
    to_svg.add_argument("image", type=Path)
    to_svg.add_argument("output", type=Path)
    to_svg.add_argument(
        "--resolution",
        type=positive_int,
        default=400,
        help="shrink the image to this many pixels on its long side first",
    )
    to_svg.add_argument(
        "--smoothing",
        type=odd_size,
        default=5,
        help="mode filter size in pixels that votes away stray pixels; 1 turns it off",
    )
    to_svg.add_argument(
        "--speckle",
        type=non_negative,
        default=8.0,
        help="drop regions and holes smaller than this many pixels square",
    )
    to_svg.add_argument(
        "--blend",
        type=positive_int,
        default=2,
        help="mix colours by overlapping up to this many pens; 1 gives each region "
        "a single pen",
    )
    style = to_svg.add_mutually_exclusive_group()
    style.add_argument(
        "--tone",
        action="store_true",
        help="shade with crossing pen lines instead of filled regions, for photos; "
        "the lines are a pixel apart, so set --resolution to the drawing's width "
        "over the pen width (--smoothing, --speckle and --blend are ignored)",
    )
    style.add_argument(
        "--one-line",
        action="store_true",
        help="draw each pen as one smooth line that wanders more densely where more "
        "of that pen is wanted, for photos; the line is a pixel wide, so set "
        "--resolution to the drawing's width over the pen width (--smoothing, "
        "--speckle and --blend are ignored)",
    )
    style.add_argument(
        "--centerline",
        action="store_true",
        help="draw each region once along its middle instead of filling it, for line "
        "art and lettering; keep --resolution near the image's own size so thin "
        "lines survive (--blend is ignored)",
    )
    to_svg.add_argument(
        "--breaks",
        type=non_negative_int,
        default=3,
        help="with --one-line, lift each pen at most this many times, cutting its "
        "line at the longest jumps",
    )
    to_svg.set_defaults(handler=run_image)

    to_gcode = commands.add_parser(
        "gcode",
        parents=[paper, pen_choice],
        help="convert an SVG to G-code fitted to the paper, one pen at a time",
    )
    to_gcode.add_argument("svg", type=Path)
    to_gcode.add_argument("output", type=Path)
    to_gcode.add_argument("--margin", type=non_negative, default=10.0, help="mm")
    to_gcode.add_argument(
        "--tolerance",
        type=positive,
        default=0.1,
        help="max distance between a curve and its line segments, mm",
    )
    to_gcode.add_argument("--feed", type=feed_rate, default=1500.0, help="mm/min")
    to_gcode.add_argument(
        "--lift", type=positive, default=5.0, help="pen-up height above the paper, mm"
    )
    to_gcode.add_argument(
        "--hatch",
        type=hatch_spacing,
        default=0.8,
        help="spacing of the lines that fill shapes, mm; 0 leaves fills empty",
    )
    to_gcode.add_argument(
        "--hatch-angle",
        type=finite,
        default=45.0,
        help="direction of the first pen's fill lines from the paper's X edge, "
        "degrees; each later pen turns 180/pens further so overlaps cross-hatch",
    )
    to_gcode.add_argument(
        "--outline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="trace the edge of every filled shape",
    )
    to_gcode.set_defaults(handler=run_gcode)

    pen_offset = argparse.ArgumentParser(add_help=False)
    pen_offset.add_argument(
        "--pen-offset",
        type=finite,
        required=True,
        help="pen tip distance beyond the gripper's target_frame along its +X, mm; "
        "0 when the controller's TCP is the pen tip",
    )

    teach = commands.add_parser(
        "teach",
        parents=[pen_offset],
        help="print --origin for the pen tip touching the paper's bottom-left corner",
    )
    add_endpoint_arguments(teach)
    teach.set_defaults(handler=run_teach)

    placement = argparse.ArgumentParser(add_help=False)
    paper_pose = placement.add_mutually_exclusive_group(required=True)
    paper_pose.add_argument(
        "--origin",
        type=finite,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="paper bottom-left corner on its surface, base frame, mm",
    )
    paper_pose.add_argument(
        "--corners",
        type=finite,
        nargs=12,
        metavar=CORNER_METAVARS,
        help="the paper's bottom-left, bottom-right, top-right and top-left corners "
        "on its surface, base frame, mm; the drawing is stretched to land on each",
    )
    placement.add_argument(
        "--yaw",
        type=finite,
        help="with --origin, the angle of the paper's X edge from base X, degrees "
        "(default 0)",
    )

    draw = commands.add_parser(
        "draw", parents=[pen_offset, placement], help="draw a G-code file on the robot"
    )
    draw.add_argument("gcode", type=Path)
    draw.add_argument(
        "--z-offset",
        type=finite,
        default=0.0,
        help="height trim added to every Z, mm (negative presses harder)",
    )
    draw.add_argument(
        "--rapid", type=feed_rate, default=3000.0, help="pen-up travel speed, mm/min"
    )
    draw.add_argument(
        "--tilt",
        type=tilt_angle,
        default=15.0,
        help="lean the pen back from its direction of travel, degrees; "
        "0 holds the start orientation",
    )
    draw.add_argument(
        "--swap-wait",
        type=non_negative,
        metavar="SECONDS",
        help="wait this long at each pen change instead of asking for Enter",
    )
    draw.add_argument(
        "--tool",
        type=positive_int,
        help="draw only this tool's strokes, with its pen already in the gripper: no "
        f"pen change, and the arm ends held {robot.PEN_CHANGE_HEIGHT_MM:g} mm above "
        "the paper for the next swap",
    )
    draw.add_argument(
        "--dry-run",
        action="store_true",
        help="print the pen-tip bounds and duration without moving",
    )
    add_endpoint_arguments(draw)
    draw.set_defaults(handler=run_draw)

    view = commands.add_parser(
        "view",
        parents=[pen_offset, placement, paper],
        help="watch the arm in MuJoCo and see the ink on the paper (run under mjpython)",
    )
    add_endpoint_arguments(view)
    view.set_defaults(handler=run_view)

    live = commands.add_parser(
        "live",
        parents=[paper],
        help="doodle from local speech or a transcript; preview by default",
    )
    live.add_argument(
        "output", type=Path, help="new session directory; must not already exist"
    )
    source = live.add_mutually_exclusive_group()
    source.add_argument(
        "--transcript", type=Path, help="replay one spoken phrase per line"
    )
    source.add_argument("--wav", type=Path, help="replay a 16 kHz mono PCM16 WAV")
    live.add_argument("--device", help="microphone device name or numeric index")
    live.add_argument("--whisper-cli", default="whisper-cli")
    live.add_argument(
        "--whisper-model", type=Path, help="local whisper.cpp GGML model file"
    )
    live.add_argument("--language", default="en", help="Whisper language code, or auto")
    live.add_argument(
        "--chunk-seconds",
        type=positive,
        default=4.0,
        help="audio window or transcript line interval, 1–30 seconds",
    )
    live.add_argument(
        "--model",
        default="keywords",
        help="Ollama model name, e.g. qwen2.5:7b; default keywords needs no LLM",
    )
    live.add_argument("--margin", type=non_negative, default=10.0, help="mm")
    live.add_argument(
        "--cell", type=positive, default=40.0, help="canvas cell size, mm"
    )
    live.add_argument(
        "--feed", type=feed_rate, default=1000.0, help="ink speed, mm/min"
    )
    live.add_argument(
        "--rapid", type=feed_rate, default=3000.0, help="pen-up speed, mm/min"
    )
    live.add_argument("--lift", type=positive, default=2.0, help="pen-up height, mm")
    live.add_argument(
        "--ink-seconds",
        type=positive,
        default=8.0,
        help="maximum ink motion per doodle; travel and lifts add time",
    )
    live.add_argument(
        "--max-age",
        type=positive,
        default=20.0,
        help="discard older ideas, seconds since audio capture",
    )
    live.add_argument(
        "--preview-speed",
        type=positive,
        default=1.0,
        help="preview motion speed multiplier; does not affect robot",
    )
    live.add_argument(
        "--execute",
        action="store_true",
        help="send motion to the configured controller; load a black pen first",
    )
    live.add_argument("--origin", nargs=3, type=finite, metavar=("X", "Y", "Z"))
    live.add_argument("--yaw", type=finite, default=0.0, help="paper yaw, degrees")
    live.add_argument("--pen-offset", type=finite, help="required with --execute, mm")
    add_endpoint_arguments(live)
    live.set_defaults(handler=run_live)

    talk = commands.add_parser(
        "talk",
        parents=[paper],
        help="every few minutes, have Codex doodle what the talk just said; "
        "preview by default",
    )
    talk.add_argument(
        "output", type=Path, help="new session directory; must not already exist"
    )
    heard = talk.add_mutually_exclusive_group()
    heard.add_argument(
        "--transcript",
        type=Path,
        help="rehearse on a text transcript, windowed at --wpm without waiting",
    )
    heard.add_argument(
        "--wav", type=Path, help="replay a 16 kHz mono PCM16 WAV in real time"
    )
    talk.add_argument("--device", help="microphone device name or numeric index")
    talk.add_argument("--whisper-cli", default="whisper-cli")
    talk.add_argument(
        "--whisper-model", type=Path, help="local whisper.cpp GGML model file"
    )
    talk.add_argument("--language", default="en", help="Whisper language code, or auto")
    talk.add_argument(
        "--chunk-seconds",
        type=positive,
        default=4.0,
        help="audio per transcription, 1–30 seconds",
    )
    talk.add_argument(
        "--interval-minutes",
        type=positive,
        default=10.0,
        help="minutes of speech behind each batch of doodles",
    )
    talk.add_argument(
        "--wpm",
        type=positive,
        default=150.0,
        help="speaking rate that sets --transcript window boundaries",
    )
    talk.add_argument(
        "--doodles",
        type=int,
        choices=range(1, 4),
        default=1,
        help="most doodles per window (default 1)",
    )
    talk.add_argument(
        "--cell", type=positive, default=60.0, help="doodle square size on paper, mm"
    )
    talk.add_argument(
        "--planner-model", help="Codex model for the doodles (default: Codex's own)"
    )
    talk.add_argument(
        "--feed", type=feed_rate, default=1000.0, help="ink speed, mm/min"
    )
    talk.add_argument(
        "--rapid", type=feed_rate, default=3000.0, help="pen-up speed, mm/min"
    )
    talk.add_argument("--lift", type=positive, default=2.0, help="pen-up height, mm")
    talk.add_argument(
        "--swap-wait",
        type=non_negative,
        metavar="SECONDS",
        help="wait this long at each pen change instead of asking for Enter",
    )
    talk.add_argument(
        "--execute",
        action="store_true",
        help="draw each batch on the configured controller, swapping pens by hand",
    )
    talk.add_argument("--origin", nargs=3, type=finite, metavar=("X", "Y", "Z"))
    talk.add_argument("--yaw", type=finite, default=0.0, help="paper yaw, degrees")
    talk.add_argument("--pen-offset", type=finite, help="required with --execute, mm")
    add_endpoint_arguments(talk)
    talk.set_defaults(handler=run_talk)

    storyboard = commands.add_parser(
        "storyboard",
        parents=[paper],
        help="replay Codex-authored plans in ten-minute transcript windows",
    )
    storyboard.add_argument("transcript", type=Path)
    storyboard.add_argument("output", type=Path, help="new output directory")
    storyboard.add_argument(
        "--plans", type=Path, default=Path("examples/keynote/plans")
    )
    storyboard.add_argument("--interval-seconds", type=positive, default=600.0)
    storyboard.add_argument(
        "--wpm",
        type=positive,
        default=150.0,
        help="synthetic speech rate for untimed text",
    )
    storyboard.add_argument("--feed", type=feed_rate, default=1000.0)
    storyboard.add_argument("--rapid", type=feed_rate, default=3000.0)
    storyboard.add_argument("--lift", type=positive, default=2.0)
    storyboard.add_argument(
        "--swap-seconds",
        type=non_negative,
        default=20.0,
        help="estimated manual swap time per pen load",
    )
    storyboard.set_defaults(handler=run_storyboard)
    return parser


def main() -> int:
    logging.basicConfig(format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    try:
        return args.handler(args)
    except (OSError, ParseError, RuntimeError, ValueError, robot.DrawError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
