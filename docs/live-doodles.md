# Live voice doodles

The `live` command turns speech into small vector drawings on one sheet. It uses
whisper.cpp for local transcription and optionally Ollama for interpreting speech.
There are no API charges. Hardware, power, and the initial model downloads are
still required. After installation and model downloads, inference works offline.

## Try it without models

```bash
uv run main.py live output/live-demo --transcript examples/talk.txt --chunk-seconds 1 --preview-speed 100
```

Open `output/live-demo/preview.html`. The page refreshes its SVG every second.
Each line of the transcript arrives at the configured interval. Preview normally
runs at estimated drawing speed; `--preview-speed 100` accelerates only preview.
It never accelerates the robot.

Each session requires a new output directory. It contains:

- `preview.html` and `canvas.svg`: the cumulative **planned** drawing.
- `canvas.gcode`: the complete planned drawing, in fixed paper coordinates.
- Numbered G-code files: one drawing per accepted idea.
- `events.jsonl`: planned, previewed, drawn, skipped, or interrupted events, with
  source phrases and timing. Planned ink is not proof that the robot drew it.

## Microphone setup on macOS

Install the optional microphone dependency and [whisper.cpp](https://github.com/ggml-org/whisper.cpp).
The [Homebrew formula](https://formulae.brew.sh/formula/whisper.cpp) installs its CLI:

```bash
uv sync --extra live
brew install whisper.cpp
mkdir -p output/models
curl -fL https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin -o output/models/ggml-base.en.bin
```

Use the default microphone:

```bash
uv run --extra live main.py live output/live-mic --whisper-model output/models/ggml-base.en.bin
```

Allow microphone access for the terminal when macOS requests it. For a stage,
select the sound desk's audio interface rather than the laptop microphone:

```bash
uv run --extra live python -m sounddevice
uv run --extra live main.py live output/live-stage --whisper-model output/models/ggml-base.en.bin --device 2
```

Replace `2` with the input device index from the device listing. The input must
support 16 kHz mono capture. `--whisper-cli /path/to/whisper-cli` overrides the
executable. Other platforms can build whisper.cpp from its upstream instructions;
sounddevice also needs PortAudio, which may require a separate system package.

The default `base.en` model is English-only. For other languages, download a
multilingual Whisper model and use `--language auto` or a language code. Keyword
selection remains English-only, so use a suitable Ollama model for other languages.

## Add a local language model

Default `--model keywords` matches English words to twelve motifs. It needs no LLM
and skips unrelated speech. For contextual interpretation, install and start
[Ollama](https://ollama.com), then download a local model:

```bash
ollama pull qwen2.5:7b
uv run --extra live main.py live output/live-ai --whisper-model output/models/ggml-base.en.bin --model qwen2.5:7b
```

If using the Ollama CLI without the desktop app, run `ollama serve` in another
terminal first. `qwen2.5:1.5b` is a smaller alternative for limited memory, with
less reliable interpretation. Both sizes use Apache 2.0; the Qwen2.5 3B and 72B
variants have different licenses. See the [model listing](https://ollama.com/library/qwen2.5).

The application sends only transcript text to `http://127.0.0.1:11434/api/chat`.
Ollama's [structured outputs](https://ollama.com/blog/structured-outputs) constrain
the answer to a known motif or `skip`. The model cannot provide robot commands or
coordinates. Invalid answers, unavailable models, and request timeouts produce a
warning and fall back to keyword selection.

## Recorded audio

Convert audio to 16 kHz, mono, 16-bit PCM WAV, then replay it:

```bash
ffmpeg -i talk.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output/talk.wav
uv run main.py live output/live-recording --wav output/talk.wav --whisper-model output/models/ggml-base.en.bin --model qwen2.5:7b
```

WAV input does not require sounddevice. Replay reads windows at speech speed,
plus transcription and interpretation time; it is not a timing-accurate benchmark
of live microphone performance.

## Draw through the existing controller

Start the controller and MuJoCo viewer as described in the main README. Match
paper size and placement between the viewer and this command:

```bash
uv run main.py live output/live-simulator --transcript examples/talk.txt --paper 250x250 --execute --pen-offset 40 --origin -125 -460 30.6
```

`--execute` sends motion to the configured controller. The default is localhost;
`--host` selects a different controller. For the physical arm, use its measured
paper origin and pen offset, load a black pen, and pass its host. Microphone and
Ollama options work with `--execute` too. The example origin above is only the
documented simulator placement, not a calibration for another setup.

The live command uses the current pen orientation (`tilt=0`), the existing robot
driver's error checks, and a lift between strokes. It does not open the gripper or
request pen swaps. Ctrl-C during motion invokes the driver's abort lift and stop.
An error stops the session; it does not automatically retry a partially drawn motif.
Do not replay the entire cumulative G-code over an already drawn session.

## Timing and limits

- Capture continues while drawing. Microphone windows are four seconds by default,
  with half a second of overlap. Only the newest waiting audio window and newest
  waiting drawing are retained. Slow inference can therefore skip speech.
- whisper.cpp runs as a subprocess per window and reloads its model. This keeps
  installation simple but adds latency. It is chunked transcription, not word-level
  streaming. Quiet windows are skipped by an energy threshold; noisy rooms can
  still produce transcription errors. Benchmark a direct stage feed before use.
- `--ink-seconds 8` bounds pen-down path length at the configured feed. Travel,
  pen lifts, settling, transcription, and inference add time. It is not a guaranteed
  speech-to-drawing latency. `--max-age 20` discards ideas more than twenty seconds
  old when the robot is ready to start them.
- `--feed 1000`, `--lift 2`, and `--rapid 3000` are the defaults, in mm/min and mm.
  The model never changes them. Paper bounds are checked before every drawing.
- Twelve fixed motifs cover trees, sun, city, person, heart, lightbulb, robot,
  rocket, book, cloud, growth, and globe. This version selects motifs; it does not
  generate arbitrary illustrations or connect objects into a shared scene.
- Cells fill in alternating rows, without repainting earlier cells. Default A4
  paper holds 24 cells; `--paper 250x250` holds 25. `--cell` and `--margin` change
  capacity. A full canvas ends the session. Replace the paper and start a new
  directory; automatic paper advancement and session resumption are not supported.
- Keep local AI load away from a timing-sensitive controller when possible. The
  robot driver runs separately from the Whisper subprocess and Ollama server,
  but they still compete for hardware resources on the same computer.

Start with transcript preview, then recorded speech, then microphone preview,
and finally the controller simulator before a live stage run.
