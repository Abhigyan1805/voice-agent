"""FastAPI app: a browser page streams microphone audio over a WebSocket and plays the replies.

Protocol on ``/ws``:

* browser -> server: binary messages of 16 kHz mono little-endian int16 PCM; JSON text messages
  ``{"type": "text", "text": ...}`` (typed question), ``{"type": "playback_done"}`` (the browser's
  audio queue has drained) and ``{"type": "reset"}`` (forget the conversation).
* server -> browser: JSON events (``vad``, ``transcript``, ``token``, ``tool``, ``sentence``,
  ``metrics``, ``interrupt``, ``error``). Each ``sentence`` event with ``"audio": true`` is followed
  by one binary message holding that sentence's int16 PCM at ``sample_rate``.

Barge-in: the server runs the VAD on the incoming audio all the time. If speech starts while a
reply is being generated or is still playing in the browser, the current turn is cancelled and an
``interrupt`` event tells the browser to drop its queued audio.
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .audio import float_to_pcm16, pcm16_to_float
from .pipeline import SpeculativeSTT, SpeechAudio, Token, ToolUse, Transcript, TurnDone
from .vad import SileroScorer, SpeechEnd, SpeechPause, SpeechStart, VadSegmenter

STATIC = Path(__file__).parent / "static"


def create_app(pipeline=None, filler: bool = False, scorer_factory=SileroScorer) -> FastAPI:
    state: dict = {"pipeline": pipeline}
    model_lock = threading.Lock()  # one turn at a time across connections: one STT + TTS instance

    @asynccontextmanager
    async def lifespan(_app):
        if state["pipeline"] is None:
            from .factory import build_pipeline

            state["pipeline"] = build_pipeline(filler=filler)
        yield

    app = FastAPI(title="Hinglish voice agent", lifespan=lifespan)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        p = state["pipeline"]
        return {"ok": p is not None, "llm": getattr(p.agent.client, "name", None) if p else None}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        pipe = state["pipeline"]
        loop = asyncio.get_running_loop()
        out: asyncio.Queue = asyncio.Queue()
        vad = VadSegmenter(scorer_factory())
        session = {"cancel": None, "busy": False, "speaking": False, "turn": 0, "spec": None}
        pipe.agent.reset()

        def emit(msg) -> None:
            try:
                loop.call_soon_threadsafe(out.put_nowait, msg)
            except RuntimeError:  # the socket's event loop is gone (client left mid-turn)
                pass

        def run_turn(turn_id, cancel, audio=None, text=None, endpoint_delay=None, speculative=None) -> None:
            with model_lock:
                if cancel.is_set():
                    return
                try:
                    for ev in pipe.run_turn(audio=audio, text=text, cancel=cancel, speculative=speculative):
                        if cancel.is_set() and not isinstance(ev, TurnDone):
                            continue
                        if isinstance(ev, Transcript):
                            lang = speculative.language if speculative else getattr(pipe.stt, "last_language", None)
                            emit({"type": "transcript", "turn": turn_id, "text": ev.text,
                                  "lang": lang if text is None else None})
                        elif isinstance(ev, Token):
                            emit({"type": "token", "turn": turn_id, "text": ev.text})
                        elif isinstance(ev, ToolUse):
                            emit({"type": "tool", "turn": turn_id, "name": ev.name, "args": ev.args, "result": ev.result})
                        elif isinstance(ev, SpeechAudio):
                            session["speaking"] = True
                            emit({"type": "sentence", "turn": turn_id, "index": ev.index, "text": ev.text,
                                  "lang": ev.lang, "audio": True, "sample_rate": ev.sample_rate})
                            emit(float_to_pcm16(ev.audio))
                        elif isinstance(ev, TurnDone):
                            m = ev.metrics.as_dict()
                            m["endpoint_delay_s"] = endpoint_delay
                            emit({"type": "metrics", "turn": turn_id, **m})
                except Exception as e:  # keep the socket alive
                    emit({"type": "error", "turn": turn_id, "message": f"{type(e).__name__}: {e}"})
                finally:
                    if session["turn"] == turn_id:
                        session["busy"] = False

        def start_turn(audio=None, text=None, endpoint_delay=None, speculative=None) -> None:
            interrupt()
            session["turn"] += 1
            cancel = threading.Event()
            session.update(cancel=cancel, busy=True)
            threading.Thread(
                target=run_turn, args=(session["turn"], cancel, audio, text, endpoint_delay, speculative), daemon=True
            ).start()

        def interrupt() -> bool:
            active = session["busy"] or session["speaking"]
            if session["cancel"] is not None:
                session["cancel"].set()
            if active:
                emit({"type": "interrupt", "turn": session["turn"]})
            session.update(busy=False, speaking=False)
            return active

        async def sender() -> None:
            while True:
                msg = await out.get()
                if isinstance(msg, bytes):
                    await ws.send_bytes(msg)
                else:
                    await ws.send_text(json.dumps(msg, ensure_ascii=False))

        send_task = asyncio.create_task(sender())
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    for ev in vad.feed(pcm16_to_float(msg["bytes"])):
                        if isinstance(ev, SpeechStart):
                            barged = interrupt()
                            emit({"type": "vad", "state": "speech", "barge_in": barged})
                        elif isinstance(ev, SpeechPause):
                            # start transcribing now; reused if the turn really ends here
                            if ev.audio.size >= 0.3 * 16_000:
                                session["spec"] = SpeculativeSTT(pipe.stt, ev.audio, ev.pause_id, pipe.stt_language)
                        elif isinstance(ev, SpeechEnd):
                            emit({"type": "vad", "state": "silence", "duration": round(ev.duration, 2)})
                            spec, session["spec"] = session["spec"], None
                            if spec is not None and spec.pause_id != ev.pause_id:
                                spec = None
                            if ev.duration >= 0.3:
                                start_turn(audio=ev.audio, endpoint_delay=round(ev.t_detected - ev.t_end, 3),
                                           speculative=spec)
                elif msg.get("text"):
                    data = json.loads(msg["text"])
                    kind = data.get("type")
                    if kind == "text" and data.get("text", "").strip():
                        start_turn(text=data["text"].strip())
                    elif kind == "playback_done":
                        session["speaking"] = False
                    elif kind == "reset":
                        interrupt()
                        vad.reset()
                        pipe.agent.reset()
                    elif kind == "stop":
                        interrupt()
        except WebSocketDisconnect:
            pass
        finally:
            interrupt()
            send_task.cancel()

    return app
