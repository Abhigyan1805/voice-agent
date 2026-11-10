"""One conversational turn: utterance audio -> transcript -> streamed LLM reply -> sentence-by-sentence audio.

The LLM runs in a producer thread and pushes tokens and finished sentences into a queue; the
caller's thread pulls from the queue and synthesises each sentence as soon as it arrives, so the
first sentence is being spoken while later ones are still being generated. Setting the ``cancel``
event (barge-in) stops both sides at the next token or sentence boundary.

Timings are wall-clock seconds measured with ``time.perf_counter`` from the moment the turn
starts (for live audio, the moment the VAD declared end-of-speech).
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from .llm import TextDelta, ToolEvent
from .text import SentenceStreamer


@dataclass
class Transcript:
    text: str
    seconds: float


@dataclass
class Token:
    text: str


@dataclass
class ToolUse:
    name: str
    args: dict
    result: dict


@dataclass
class SpeechAudio:
    index: int
    text: str
    lang: str
    audio: np.ndarray = field(repr=False)
    sample_rate: int = 22_050


class SpeculativeSTT:
    """Transcribe in a background thread, started at a VAD pause before the turn is confirmed.

    If the VAD then ends the turn with the same audio (``SpeechEnd.pause_id`` matches), the turn
    reuses this transcript and only waits for whatever decode time is left; if the user kept
    talking, the job is simply dropped.
    """

    def __init__(self, stt, audio: np.ndarray, pause_id: int, language: str | None = "auto"):
        self.pause_id = pause_id
        self.text: str | None = None
        self.language: str | None = None
        self.compute_s = 0.0
        self.error: BaseException | None = None
        self._done = threading.Event()

        def work() -> None:
            t = time.perf_counter()
            try:
                self.text = stt.transcribe(audio, language=language)
                self.language = getattr(stt, "last_language", None)
            except BaseException as e:  # re-raised in result()
                self.error = e
            finally:
                self.compute_s = time.perf_counter() - t
                self._done.set()

        threading.Thread(target=work, name="speculative-stt", daemon=True).start()

    def result(self) -> str:
        self._done.wait()
        if self.error is not None:
            raise self.error
        return self.text or ""


@dataclass
class TurnMetrics:
    stt_s: float | None = None  # STT time on the critical path (after end of turn)
    stt_compute_s: float | None = None  # full decode time (differs from stt_s when speculative)
    stt_speculative: bool = False
    llm_first_token_s: float | None = None  # from LLM request start
    llm_first_sentence_s: float | None = None  # from LLM request start
    tts_first_audio_s: float | None = None  # synthesis time of the first sentence
    time_to_first_audio_s: float | None = None  # from turn start (STT + LLM + TTS on the critical path)
    time_to_filler_audio_s: float | None = None  # only when the filler option spoke during a tool call
    llm_total_s: float | None = None
    total_s: float | None = None
    tools: list[str] = field(default_factory=list)
    sentences: int = 0
    reply_audio_s: float = 0.0
    cached: bool = False
    llm_requests: int = 0
    llm_model: str | None = None
    llm_retries: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    interrupted: bool = False
    error: str | None = None
    reply: str = ""
    transcript: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class TurnDone:
    metrics: TurnMetrics


Event = Transcript | Token | ToolUse | SpeechAudio | TurnDone

_SENTENCE, _TOKEN, _TOOL, _END, _ERROR = range(5)


class VoicePipeline:
    def __init__(
        self,
        stt,
        agent,
        tts,
        stt_language: str | None = "auto",
        streamer_kw: dict | None = None,
        filler: bool = False,
    ):
        self.stt, self.agent, self.tts = stt, agent, tts
        self.stt_language = stt_language
        self.streamer_kw = streamer_kw or {}
        # Speak a short "one moment" while a tool round-trip is in flight. Off by default so the
        # headline time-to-first-audio is always time to the first words of the actual answer.
        self.filler = filler

    def run_turn(
        self,
        audio: np.ndarray | None = None,
        text: str | None = None,
        cancel: threading.Event | None = None,
        speculative: SpeculativeSTT | None = None,
    ) -> Iterator[Event]:
        cancel = cancel or threading.Event()
        m = TurnMetrics()
        t0 = time.perf_counter()

        if speculative is not None:
            text = speculative.result()
            m.stt_s = time.perf_counter() - t0
            m.stt_compute_s = speculative.compute_s
            m.stt_speculative = True
        elif text is None:
            if audio is None:
                raise ValueError("give audio or text")
            text = self.stt.transcribe(audio, language=self.stt_language)
            m.stt_s = m.stt_compute_s = time.perf_counter() - t0
        else:
            m.stt_s = 0.0
        m.transcript = text
        yield Transcript(text, m.stt_s)
        if not text.strip():
            m.total_s = time.perf_counter() - t0
            yield TurnDone(m)
            return

        q: queue.Queue = queue.Queue()
        t_llm = time.perf_counter()

        def produce() -> None:
            streamer = SentenceStreamer(**self.streamer_kw)
            try:
                for ev in self.agent.respond(text, cancel):
                    if cancel.is_set():
                        break
                    if isinstance(ev, ToolEvent):
                        q.put((_TOOL, ev, time.perf_counter()))
                    elif isinstance(ev, TextDelta):
                        q.put((_TOKEN, ev.text, time.perf_counter()))
                        for s in streamer.feed(ev.text):
                            q.put((_SENTENCE, s, time.perf_counter()))
                if not cancel.is_set():
                    for s in streamer.flush():
                        q.put((_SENTENCE, s, time.perf_counter()))
            except Exception as e:  # surfaced to the consumer, which speaks a short apology
                q.put((_ERROR, f"{type(e).__name__}: {e}", time.perf_counter()))
            finally:
                q.put((_END, None, time.perf_counter()))

        worker = threading.Thread(target=produce, name="llm-producer", daemon=True)
        worker.start()
        reply: list[str] = []
        index = 0
        while True:
            try:
                kind, payload, t_ev = q.get(timeout=0.05)
            except queue.Empty:
                if cancel.is_set():
                    break
                continue
            if cancel.is_set():
                break
            if kind == _END:
                m.llm_total_s = t_ev - t_llm
                break
            if kind == _ERROR:
                m.error = payload
                payload = "माफ़ कीजिए, अभी जवाब नहीं मिल पाया।" if _is_hindi(text) else "Sorry, I could not get an answer."
                kind = _SENTENCE
            if kind == _TOOL:
                m.tools.append(payload.name)
                yield ToolUse(payload.name, payload.args, payload.result)
                if self.filler and index == 0 and m.time_to_filler_audio_s is None:
                    phrase = "एक सेकंड।" if _is_hindi(text) else "One moment."
                    wav = self.tts.synthesize(phrase)
                    m.time_to_filler_audio_s = time.perf_counter() - t0
                    yield SpeechAudio(-1, phrase, self.tts.voice_for(phrase), wav, self.tts.sample_rate)
            elif kind == _TOKEN:
                if m.llm_first_token_s is None and payload.strip():
                    m.llm_first_token_s = t_ev - t_llm
                reply.append(payload)
                yield Token(payload)
            elif kind == _SENTENCE:
                if m.llm_first_sentence_s is None:
                    m.llm_first_sentence_s = t_ev - t_llm
                lang = self.tts.voice_for(payload)
                wav = self.tts.synthesize(payload, lang)
                if cancel.is_set():
                    break
                if m.time_to_first_audio_s is None:
                    m.tts_first_audio_s = self.tts.last_seconds
                    m.time_to_first_audio_s = time.perf_counter() - t0
                m.sentences += 1
                m.reply_audio_s += len(wav) / self.tts.sample_rate
                yield SpeechAudio(index, payload, lang, wav, self.tts.sample_rate)
                index += 1
        m.interrupted = cancel.is_set()
        info = getattr(self.agent.client, "info", None)
        m.cached = bool(info and info.cached)
        m.llm_requests = info.requests if info else 0
        m.llm_retries = info.retries if info else 0
        m.llm_model = getattr(info, "model", None)
        m.llm_prompt_tokens = getattr(info, "prompt_tokens", 0)
        m.llm_completion_tokens = getattr(info, "completion_tokens", 0)
        m.reply = "".join(reply).strip()
        m.total_s = time.perf_counter() - t0
        yield TurnDone(m)


def _is_hindi(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)
