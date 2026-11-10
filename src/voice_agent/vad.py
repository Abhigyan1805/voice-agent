"""Streaming voice-activity detection.

Audio arrives in arbitrary-sized chunks (a browser sends ~20-100 ms at a time). ``VadSegmenter``
cuts it into 32 ms frames, asks a *frame scorer* for a speech probability per frame, and runs a
small state machine with hysteresis:

* **silence -> speech** once ``min_speech_ms`` of consecutive frames score above ``start_threshold``
  (emits ``SpeechStart``; this is what triggers barge-in);
* **speech -> silence** once ``end_silence_ms`` of consecutive frames score below ``end_threshold``
  (emits ``SpeechEnd`` with the utterance audio, padded with ``pre_roll_ms`` of audio from before
  the start so the first syllable is not clipped).

The scorer is pluggable: ``SileroScorer`` wraps the Silero VAD v6 ONNX model that ships with
faster-whisper, and ``EnergyScorer`` is a dependency-free RMS fallback used in tests.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Protocol

import numpy as np

SAMPLE_RATE = 16_000
FRAME = 512  # samples per frame at 16 kHz = 32 ms (what Silero expects)
FRAME_MS = FRAME * 1000 / SAMPLE_RATE


class FrameScorer(Protocol):
    def __call__(self, frame: np.ndarray) -> float: ...
    def reset(self) -> None: ...


class EnergyScorer:
    """Map frame RMS (dBFS) to a pseudo-probability. Deterministic; useful for tests and as a fallback."""

    def __init__(self, floor_db: float = -50.0, ceil_db: float = -30.0):
        self.floor_db, self.ceil_db = floor_db, ceil_db

    def __call__(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12))
        db = 20.0 * np.log10(rms + 1e-12)
        return float(np.clip((db - self.floor_db) / (self.ceil_db - self.floor_db), 0.0, 1.0))

    def reset(self) -> None:
        pass


def silero_model_path() -> Path:
    import faster_whisper

    return Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx"


class SileroScorer:
    """Stateful, frame-at-a-time Silero VAD v6 (ONNX, ~2 MB, well under 1 ms per frame on CPU)."""

    CONTEXT = 64

    def __init__(self, path: str | Path | None = None):
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(
            str(path or silero_model_path()), providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros(self.CONTEXT, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.context, frame.astype(np.float32)])[None, :]
        out, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.context = frame[-self.CONTEXT :].astype(np.float32)
        return float(np.asarray(out).reshape(-1)[0])


@dataclass
class VadConfig:
    start_threshold: float = 0.5
    end_threshold: float = 0.35
    min_speech_ms: float = 192.0  # 6 frames
    end_silence_ms: float = 544.0  # 17 frames of silence ends the turn
    pre_roll_ms: float = 320.0
    tail_ms: float = 96.0  # silence kept after the last speech frame
    max_utterance_s: float = 20.0


@dataclass
class SpeechStart:
    t: float  # seconds of stream time at which speech began (start of the first speech frame)


@dataclass
class SpeechEnd:
    t_start: float
    t_end: float  # stream time of the end of the last speech frame (before the hangover)
    t_detected: float  # stream time at which the end was declared (t_end + hangover)
    audio: np.ndarray = field(repr=False)
    forced: bool = False  # True when max_utterance_s cut the turn

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


VadEvent = SpeechStart | SpeechEnd


class VadSegmenter:
    """Feed audio, get ``SpeechStart`` / ``SpeechEnd`` events. Not thread-safe; one per stream."""

    def __init__(self, scorer: FrameScorer | Callable[[np.ndarray], float], config: VadConfig | None = None):
        self.scorer = scorer
        self.cfg = config or VadConfig()
        self._ms = lambda ms: max(1, int(round(ms / FRAME_MS)))
        self.reset()

    def reset(self) -> None:
        if hasattr(self.scorer, "reset"):
            self.scorer.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._frames_seen = 0
        self._pre = deque(maxlen=self._ms(self.cfg.pre_roll_ms))
        self._in_speech = False
        self._candidate: list[np.ndarray] = []  # speech frames not yet confirmed as a start
        self._utt: list[np.ndarray] = []
        self._silence_run = 0
        self._speech_start_frame = 0
        self._last_speech_frame = 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def feed(self, audio: np.ndarray) -> Iterator[VadEvent]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        buf = np.concatenate([self._pending, audio]) if len(self._pending) else audio
        n = len(buf) // FRAME
        for i in range(n):
            yield from self._frame(buf[i * FRAME : (i + 1) * FRAME])
        self._pending = buf[n * FRAME :].copy()

    def flush(self) -> Iterator[VadEvent]:
        """End of stream: close an open utterance."""
        if self._in_speech:
            yield self._close(forced=False)

    def _t(self, frame_idx: int) -> float:
        return frame_idx * FRAME / SAMPLE_RATE

    def _frame(self, frame: np.ndarray) -> Iterator[VadEvent]:
        idx = self._frames_seen
        self._frames_seen += 1
        p = self.scorer(frame)
        cfg = self.cfg
        if not self._in_speech:
            if p >= cfg.start_threshold:
                self._candidate.append(frame)
                if len(self._candidate) >= self._ms(cfg.min_speech_ms):
                    self._in_speech = True
                    self._speech_start_frame = idx - len(self._candidate) + 1
                    self._last_speech_frame = idx
                    self._utt = list(self._pre) + self._candidate
                    self._candidate = []
                    self._silence_run = 0
                    self._pre.clear()
                    yield SpeechStart(t=self._t(self._speech_start_frame))
                return
            # A blip shorter than min_speech_ms: push it into the pre-roll and keep listening.
            for f in self._candidate:
                self._pre.append(f)
            self._candidate = []
            self._pre.append(frame)
            return
        self._utt.append(frame)
        if p >= cfg.end_threshold:
            self._silence_run = 0
            self._last_speech_frame = idx
        else:
            self._silence_run += 1
        if self._silence_run >= self._ms(cfg.end_silence_ms):
            yield self._close(forced=False)
        elif len(self._utt) * FRAME / SAMPLE_RATE >= cfg.max_utterance_s:
            yield self._close(forced=True)

    def _close(self, forced: bool) -> SpeechEnd:
        tail = self._ms(self.cfg.tail_ms)
        keep = len(self._utt) - max(0, self._silence_run - tail)
        audio = np.concatenate(self._utt[:keep]) if keep > 0 else np.zeros(0, dtype=np.float32)
        ev = SpeechEnd(
            t_start=self._t(self._speech_start_frame),
            t_end=self._t(self._last_speech_frame + 1),
            t_detected=self._t(self._frames_seen),
            audio=audio,
            forced=forced,
        )
        self._in_speech = False
        self._utt = []
        self._silence_run = 0
        return ev


def segment(audio: np.ndarray, scorer, config: VadConfig | None = None, chunk: int = 1600) -> list[VadEvent]:
    """Run a whole array through a fresh segmenter in ``chunk``-sample pieces (100 ms by default)."""
    seg = VadSegmenter(scorer, config)
    events: list[VadEvent] = []
    for i in range(0, len(audio), chunk):
        events.extend(seg.feed(audio[i : i + chunk]))
    events.extend(seg.flush())
    return events
