"""Smoke test against a running server: a typed turn, a spoken turn streamed in real time, and a barge-in.

Speaks the same WebSocket protocol as the browser page (see src/voice_agent/server.py).

    voice-agent serve --port 8765 &
    python scripts/smoke_ws.py [ws://127.0.0.1:8765/ws]
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_agent.audio import float_to_pcm16, load_audio  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765/ws"


async def recv_until(ws, kinds: set[str], log: list, timeout: float = 60) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = await asyncio.wait_for(ws.recv(), timeout)
        if isinstance(msg, bytes):
            log.append(("audio", len(msg)))
            continue
        data = json.loads(msg)
        log.append((data["type"], data))
        if data["type"] in kinds:
            return data
    raise TimeoutError(kinds)


async def stream(ws, audio) -> None:
    """Send 32 ms frames at real-time pace, like the browser's audio worklet."""
    for i in range(0, len(audio), 512):
        await ws.send(float_to_pcm16(audio[i : i + 512]))
        await asyncio.sleep(0.032)


async def main() -> None:
    async with websockets.connect(URL, max_size=None) as ws:
        log: list = []
        await ws.send(json.dumps({"type": "text", "text": "आज कौन सा दिन है?"}))
        m = await recv_until(ws, {"metrics"}, log)
        print("typed :", [k for k, _ in log], "| reply:", m["reply"], "| first audio", m["time_to_first_audio_s"], "s")
        await ws.send(json.dumps({"type": "playback_done"}))

        log = []
        await stream(ws, load_audio(ROOT / "prompts" / "audio" / "hx-kb-2.wav"))
        m = await recv_until(ws, {"metrics"}, log)
        heard = next(d for k, d in log if k == "transcript")
        print(f"spoken: heard [{heard['lang']}] {heard['text']} | reply: {m['reply'][:100]}")
        print(f"        stt {m['stt_s']} s (speculative {m['stt_speculative']}), llm first token "
              f"{m['llm_first_token_s']} s, first audio {m['time_to_first_audio_s']} s after the "
              f"{m['endpoint_delay_s']} s end-of-turn wait" + (" (LLM cache hit)" if m["cached"] else ""))

        log = []
        await ws.send(json.dumps({"type": "text", "text": "Tell me about the history of Delhi in five sentences."}))
        await recv_until(ws, {"sentence"}, log)  # the agent has started speaking
        await stream(ws, load_audio(ROOT / "prompts" / "audio" / "hi-time-1.wav")[8000:20000])  # talk over it
        await recv_until(ws, {"interrupt"}, log, timeout=10)
        m = await recv_until(ws, {"metrics"}, log)
        print(f"barge-in: interrupted={m['interrupted']} after {m['sentences']} sentence(s)")
        assert m["interrupted"]


if __name__ == "__main__":
    asyncio.run(main())
