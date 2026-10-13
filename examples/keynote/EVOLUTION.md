# Illustrated transcript replay

Open `output/keynote-evolution/index.html`. Play streams all 10,658 cleaned spoken
words at the selected speed. The original file, including its wrapper, is also
available in the transcript panel. Timing is synthetic at 150 words per minute.

Twelve illustrations appear individually after their corresponding ten-minute
windows close. The final partial window flushes at the end. Playback pauses for
1.4 seconds per addition so each change is visible. Previous/Next doodle allows
manual inspection, including individual additions within the same batch.

The artwork was generated as one composed sheet with the built-in image tool.
The browser reveals fixed regions of that image without regenerating prior art.
This demonstrates visual sequencing, not independent live generation, stroke
execution, or vector output. These illustrations have not been converted to
robot paths. Existing G-code from the earlier vector fixture does not represent
this artwork.

Rebuild after changing the viewer template:

```sh
.venv/bin/python examples/keynote/build_evolution.py
```

The script verifies the transcript hashes against the source plans. Artwork and
the generation prompt are in `output/keynote-evolution/`.
