"""Command line: ask a recorded question, chat by typing, or start the web UI.

    voice-agent ask question.wav --out reply.wav     # speech in, speech out, with per-stage timings
    voice-agent chat                                 # type questions, hear (saved) spoken answers
    voice-agent serve                                # browser UI with live mic and barge-in
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def _print_turn(pipe, events_iter, out: Path | None, play: bool) -> dict:
    from .audio import write_wav
    from .pipeline import SpeechAudio, Token, ToolUse, Transcript, TurnDone

    chunks: list[np.ndarray] = []
    sr = pipe.tts.sample_rate
    metrics = {}
    for ev in events_iter:
        if isinstance(ev, Transcript):
            lang = getattr(pipe.stt, "last_language", "") if pipe.stt else ""
            print(f"you  [{lang}] : {ev.text}" if lang else f"you  : {ev.text}")
            print("agent: ", end="", flush=True)
        elif isinstance(ev, ToolUse):
            print(f"\n  [tool {ev.name}({json.dumps(ev.args, ensure_ascii=False)}) -> "
                  f"{json.dumps(ev.result, ensure_ascii=False)[:120]}]\nagent: ", end="", flush=True)
        elif isinstance(ev, Token):
            print(ev.text, end="", flush=True)
        elif isinstance(ev, SpeechAudio):
            chunks.append(ev.audio)
            chunks.append(np.zeros(int(0.15 * sr), dtype=np.float32))
        elif isinstance(ev, TurnDone):
            metrics = ev.metrics.as_dict()
    print()
    keys = ("stt_s", "llm_first_token_s", "llm_first_sentence_s", "tts_first_audio_s", "time_to_first_audio_s",
            "time_to_filler_audio_s", "total_s")
    print("timings: " + "  ".join(f"{k}={metrics[k]:.2f}" for k in keys if metrics.get(k) is not None)
          + f"  tools={metrics.get('tools')}" + ("  (LLM cache hit)" if metrics.get("cached") else ""))
    if metrics.get("error"):
        print(f"LLM error: {metrics['error']}", file=sys.stderr)
    if chunks and out:
        write_wav(out, np.concatenate(chunks), sr)
        print(f"reply audio -> {out}")
        if play and shutil.which("aplay"):
            subprocess.run(["aplay", "-q", str(out)], check=False)
    return metrics


def cmd_ask(args) -> None:
    from .audio import load_audio
    from .factory import build_pipeline
    from .vad import SileroScorer, SpeechEnd, segment

    pipe = build_pipeline(filler=args.filler)
    audio = load_audio(args.audio)
    ends = [e for e in segment(audio, SileroScorer()) if isinstance(e, SpeechEnd)]
    if not ends:
        print("no speech detected by the VAD; using the whole file", file=sys.stderr)
        utt = audio
    else:
        utt = np.concatenate([e.audio for e in ends])
    _print_turn(pipe, pipe.run_turn(audio=utt), Path(args.out) if args.out else None, args.play)


def cmd_chat(args) -> None:
    from .factory import build_pipeline

    pipe = build_pipeline(need_stt=False, filler=args.filler)
    print(f"LLM: {pipe.agent.client.name} ({pipe.agent.client.model}). Empty line to quit.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    turn = 0
    while True:
        try:
            text = input("\n> ").strip()
        except EOFError:
            break
        if not text:
            break
        turn += 1
        _print_turn(pipe, pipe.run_turn(text=text), out_dir / f"reply-{turn:02d}.wav", args.play)


def cmd_serve(args) -> None:
    import uvicorn

    from .server import create_app

    print(f"open http://{args.host}:{args.port}  (models load on startup)")
    uvicorn.run(create_app(filler=args.filler), host=args.host, port=args.port, log_level="warning")


def cmd_demo_service(args) -> None:
    from .demo_service import serve

    print(f"order service (local stand-in) on http://127.0.0.1:{args.port}; set VOICE_AGENT_ORDERS_URL to use it")
    serve(port=args.port, api_key=args.key)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="voice-agent", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="answer a recorded question (wav/flac/ogg)")
    a.add_argument("audio")
    a.add_argument("--out", default="reply.wav")
    a.add_argument("--play", action="store_true", help="play the reply with aplay if available")
    a.add_argument("--filler", action="store_true", help="say 'one moment' while a tool call runs")
    a.set_defaults(fn=cmd_ask)
    c = sub.add_parser("chat", help="type questions, get spoken answers as wav files")
    c.add_argument("--out-dir", default="results/audio/chat")
    c.add_argument("--play", action="store_true")
    c.add_argument("--filler", action="store_true")
    c.set_defaults(fn=cmd_chat)
    s = sub.add_parser("serve", help="web UI with live microphone, streaming replies and barge-in")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--filler", action="store_true")
    s.set_defaults(fn=cmd_serve)
    d = sub.add_parser("demo-service", help="run the local order-status REST service the order tools call")
    d.add_argument("--port", type=int, default=8090)
    d.add_argument("--key", default="demo-key", help="API key the service expects in X-API-Key")
    d.set_defaults(fn=cmd_demo_service)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
