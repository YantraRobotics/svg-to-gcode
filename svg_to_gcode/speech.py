"""Offline speech input through whisper.cpp; no audio leaves this process's machine."""

import logging
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
SAMPLE_RATE = 16000


def put_latest(slot: queue.Queue, item) -> None:
    """Single producer: replace pending work when the consumer falls behind."""
    try:
        slot.put_nowait(item)
    except queue.Full:
        try:
            slot.get_nowait()
        except queue.Empty:
            pass
        slot.put_nowait(item)


class Whisper:
    def __init__(self, executable: str, model: Path, language: str):
        self.executable = shutil.which(executable)
        if self.executable is None:
            raise ValueError(
                "whisper-cli not found; install whisper.cpp or pass --whisper-cli"
            )
        if not model.is_file():
            raise ValueError(f"Whisper model not found: {model}")
        self.model = model.resolve()
        self.language = language

    def transcribe(self, pcm: bytes, stop: threading.Event) -> str:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        if not len(samples) or np.sqrt(np.mean(samples * samples)) < 0.008:
            return ""
        with tempfile.TemporaryDirectory(prefix="doodle-speech-") as directory:
            audio = Path(directory) / "speech.wav"
            output = Path(directory) / "transcript"
            with wave.open(str(audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(pcm)
            with (Path(directory) / "stderr.txt").open("w+") as errors:
                process = subprocess.Popen(
                    [
                        self.executable,
                        "-m",
                        str(self.model),
                        "-f",
                        str(audio),
                        "-l",
                        self.language,
                        "-otxt",
                        "-of",
                        str(output),
                        "-nt",
                        "-t",
                        "2",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=errors,
                )
                deadline = time.monotonic() + 60
                try:
                    while process.poll() is None:
                        if stop.wait(0.1):
                            return ""
                        if time.monotonic() > deadline:
                            raise RuntimeError("whisper-cli timed out after 60 seconds")
                    if process.returncode:
                        errors.seek(0)
                        raise RuntimeError(
                            f"whisper-cli failed: {errors.read()[-1500:]}"
                        )
                    return output.with_suffix(".txt").read_text().strip()
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()


def microphone_chunks(device: str | None, seconds: float, stop: threading.Event):
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError("Microphone support needs: uv sync --extra live") from None
    slot = queue.Queue(maxsize=1)
    buffer = bytearray()
    chunk_bytes = int(seconds * SAMPLE_RATE) * 2
    overlap_bytes = SAMPLE_RATE  # Half a second protects words at chunk boundaries.

    def capture(data, frames, timing, status):
        if status:
            log.warning("microphone: %s", status)
        buffer.extend(data)
        while len(buffer) >= chunk_bytes:
            put_latest(slot, (bytes(buffer[:chunk_bytes]), time.monotonic()))
            del buffer[: chunk_bytes - overlap_bytes]

    selected = int(device) if device is not None and device.isdecimal() else device
    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=4000,
            device=selected,
            callback=capture,
        ) as stream:
            while not stop.is_set():
                try:
                    yield slot.get(timeout=0.25)
                except queue.Empty:
                    if not stream.active:
                        raise RuntimeError("microphone stream stopped")
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Cannot capture microphone: {exc}") from exc


def wav_chunks(path: Path, seconds: float, stop: threading.Event):
    """Replay PCM WAV at speech speed, including a final partial chunk."""
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (
            1,
            2,
            SAMPLE_RATE,
        ):
            raise ValueError("--wav requires 16 kHz, mono, 16-bit PCM WAV")
        frames = int(seconds * SAMPLE_RATE)
        while not stop.is_set():
            pcm = wav.readframes(frames)
            if not pcm:
                break
            if stop.wait(len(pcm) / (2 * SAMPLE_RATE)):
                break
            yield pcm, time.monotonic()


def transcripts(args, stop: threading.Event):
    if args.transcript is not None:
        with args.transcript.open() as source:
            for line in source:
                if stop.is_set():
                    break
                if text := line.strip():
                    yield text[:4000], time.monotonic()
                    if stop.wait(args.chunk_seconds):
                        break
        return
    whisper = Whisper(args.whisper_cli, args.whisper_model, args.language)
    source = (
        wav_chunks(args.wav, args.chunk_seconds, stop)
        if args.wav
        else microphone_chunks(args.device, args.chunk_seconds, stop)
    )
    try:
        for pcm, captured in source:
            if text := whisper.transcribe(pcm, stop):
                yield text[:4000], captured
    finally:
        source.close()
