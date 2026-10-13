# svg-to-gcode

Turns photos and SVG drawings into G-code and draws them on paper with the Owl 6.9 arm holding a marker, one colour at a time. It can also show the arm and the ink in a MuJoCo viewer, driven by the real controller or the controller's simulator.

## Setup

Install [uv](https://docs.astral.sh/uv/), then from this folder run:

```bash
uv sync
```

This installs Python 3.14, the dependencies, and the project itself in editable mode.

## Photo to drawing

These three commands draw the example photo with four Expo markers on a 250 mm square canvas:

```bash
PENS="red=#D52B1E,green=#00843D,blue=#1B4F9C,black=#231F20"
mkdir -p output
uv run main.py image examples/golden-gate.jpg output/golden-gate.svg --one-line --resolution 120 --pens "$PENS"
uv run main.py gcode output/golden-gate.svg output/golden-gate.gcode --paper 250x250 --margin 12 --lift 2 --feed 1000 --pens "$PENS"
uv run main.py draw output/golden-gate.gcode --pen-offset 40 --origin -125 -460 30.6 --tilt 0 --dry-run
```

1. `image` turns the photo into an SVG with one layer per pen.
2. `gcode` fits the SVG inside the paper's margin and writes G-code that draws one pen after another, with a tool change (`T<n> M6`) between pens.
3. `draw --dry-run` prints where the pen tip will go and how long the motion takes, without moving the arm. Drop `--dry-run` to draw.

Every command has `--help`.

## Doodle a talk every few minutes

`talk` starts from the transcript. Every `--interval-minutes` of speech (default 10),
Codex reads only that window, sketches up to `--doodles` line drawings in black, blue,
green and red, and local checks keep the strokes that quote the window, stay inside
their square and fit the paper. Each batch is new G-code drawn once; earlier ink never
moves. It runs the signed-in Codex CLI (`codex login`); no API key is needed.

Rehearse on a transcript without waiting, preview only:

```bash
uv run main.py talk output/rehearsal --transcript test.txt --paper 250x250
```

Open `output/rehearsal/index.html` to step through the batches. At the talk, listen
to the microphone ([setup](docs/live-doodles.md#microphone-setup-on-macos)) and draw:

```bash
uv run --extra live main.py talk output/keynote --whisper-model output/models/ggml-base.en.bin --paper 250x250 --execute --pen-offset 40 --origin -125 -460 30.6
```

The arm asks for Enter at each pen change; `--swap-wait SECONDS` waits instead.
Ctrl-C while it is listening ends the talk and doodles the last part; Ctrl-C at any
other time stops.

## Live voice doodles

For the **ten-minute, text-free, red/green/blue/black design**, see
[the incremental architecture and keyless replay](docs/ten-minute-doodle-architecture.md).
Run the Codex-authored `test.txt` example with:

```bash
uv run main.py storyboard test.txt output/my-ten-minute-replay --paper 250x250
```

Open the generated `index.html` to step through additions. This command exports
individual batch G-code and does not move the arm.

`live` listens to a microphone, a recorded WAV, or a transcript and adds small
vector doodles to a persistent canvas. It runs locally with whisper.cpp and an
optional Ollama model; no paid API or API key is required. Preview is the default.

Try the included transcript without downloading any models:

```bash
uv run main.py live output/live-demo --transcript examples/talk.txt --chunk-seconds 1 --preview-speed 100
```

Open `output/live-demo/preview.html` to see the canvas. Use a new output directory
for each session. See [the live setup guide](docs/live-doodles.md) for microphone
setup, free local models, recorded audio, and robot/simulator output.

### Styles

`image` draws in one of three styles:

- **Filled regions** (the default) reduce the image to flat areas of the pens' colours, mixing up to `--blend` pens by overlap. `gcode` then hatches each area (`--hatch`, `--hatch-angle`) and traces its outline. This suits logos and flat artwork.
- **`--tone`** shades along rows one pen width apart, inking more of each row, in long strokes, where more of a pen's ink is wanted. Each pen's rows run at a different angle, so the colours cross. This suits photos.
- **`--one-line`** draws each pen as one smooth, continuous line that wanders more densely where more of that pen is wanted. The pen lifts at most `--breaks` times per colour (default 3), at the longest jumps. This also suits photos.

For `--tone` and `--one-line`, set `--resolution` to the drawing's width divided by the pen's width. For a 2 mm Expo tip on a 250 mm canvas with 12 mm margins, that is 226 / 2 ≈ 113. The example uses 120, so neighbouring lines overlap slightly.

### Pens

`--pens` lists the pens on hand as `NAME=#RRGGBB`, separated by commas. The default is CMYK: `yellow=#FFF200,cyan=#00AEEF,magenta=#EC008C,black=#231F20`. Pass the same list to `image` and `gcode`.

Pens are always drawn lightest first, whatever order you list them in. Colours are made by laying pens over each other, so a colour that no combination reaches comes out as the nearest one. For example, the Expo set above draws skin as red.

## Drawing on the arm

First, find where the paper is. Touch the pen tip to the paper's bottom-left corner, then run:

```bash
uv run main.py teach --pen-offset 40 --host <robot-ip>
```

It prints the `--origin` to pass to `draw`. `--pen-offset` is how far, in mm, the pen tip sticks out past the gripper's `target_frame`.

Then draw:

```bash
uv run main.py draw output/golden-gate.gcode --pen-offset 40 --origin <X> <Y> <Z> --tilt 0 --host <robot-ip>
```

At each pen change, the arm lifts the pen and holds still. If the gripper is active, `draw` asks you to press Enter before it opens and again before it grips the new pen. Otherwise, it waits while you swap the pen by hand. `--swap-wait SECONDS` replaces each Enter with a timed wait.

Things learned on the controller simulator:

- `--tilt` leans the pen back from its direction of travel, 15° by default. On a dense drawing, re-tilting at the end of every line pushed the tip error past the controller's 10 mm limit, and the drawing aborted. `--tilt 0` keeps the marker upright, and the error stayed under 3 mm.
- One-line drawings tracked within about 4 mm at `--feed 1000`, but 7.3 mm at 1500.
- The arm reaches a 250 mm canvas whose bottom-left corner is at `--origin -125 -460 30.6`. For any other placement, check the pen-tip bounds that `--dry-run` prints.
- The dry run counts planned motion only. Including the approach and pen swaps, simulated runs took about 1.1× that time for one-line drawings and 1.5× for hatched ones.

## Simulator

To try a drawing without the arm, run the control repo's firmware emulator with its MuJoCo plant, and run the controller against it. These commands assume the `control` repo sits next to this one.

```bash
# 1. In control/tools: start the emulator. It prints ENDPOINT=127.0.0.1:<PORT>.
uv run --no-sync python ../../svg-to-gcode/tools/realtime_emulator.py firmware_emulator.py --mujoco controller

# 2. In control/build: start the controller on that endpoint.
<controller> 127.0.0.1:<PORT> --listen-pub tcp://127.0.0.1:5555 --listen-cmd tcp://127.0.0.1:5556

# 3. Here: open the viewer, then draw.
uv run mjpython main.py view --pen-offset 40 --origin -125 -460 30.6 --paper 250x250
uv run main.py draw output/golden-gate.gcode --pen-offset 40 --origin -125 -460 30.6 --tilt 0 --swap-wait 3
```

`<controller>` is the `controller` target built from the control repo's current source.

The emulator ticks every 5 ms, and the controller faults when feedback is more than 25 ms old. Ordinary macOS scheduling can pause the tick thread long enough to cause that. `realtime_emulator.py` runs the tick thread on macOS real-time scheduling to prevent it.

Keep the machine otherwise idle while a simulated drawing runs. Under heavy load, the controller can drop out of its hold during a pen change.

## Layout

```text
main.py           command line: image, gcode, teach, draw, view
svg_to_gcode/     the converter and the arm driver
  pens.py           pen colours, colour mixing and ink coverage
  image.py          photo to SVG in the three styles
  one_line.py       stippling, route finding and curve fitting for --one-line
  hatch.py          fill lines for shapes
  gcode.py          SVG to G-code, one pen at a time
  robot.py          G-code to arm motion through the controller, and pen changes
  tilt.py           the trailing pen tilt
  viewer.py         MuJoCo scene with the paper and the ink
tools/            copies of control/tools, plus realtime_emulator.py
robots/           the arm's URDF with the Hand-E gripper
examples/         sample photo
output/           generated SVGs, G-code, previews and videos (git-ignored)
tests/            pytest suite
```

`tools/motion.py`, `tools/robot_interface.py` and `tools/prepare_owl69_model.py` are verbatim copies of the files in `control/tools`. Ruff skips them. To update them, copy the new versions over instead of editing them here. They import each other by module name, so the editable install puts `tools/` on the import path.

## Development

```bash
uv run pytest
uv run ruff format
uv run ruff check .
```
