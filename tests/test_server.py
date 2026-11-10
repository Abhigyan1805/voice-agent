"""WebSocket protocol and barge-in, with fake STT/TTS/LLM and the energy VAD."""

import json
import time

import numpy as np
from fastapi.testclient import TestClient

from voice_agent.agent import Agent
from voice_agent.audio import float_to_pcm16
from voice_agent.pipeline import VoicePipeline
from voice_agent.server import create_app
from voice_agent.vad import EnergyScorer
from test_agent_pipeline import FakeSTT, FakeTTS, SlowClient

SR = 16_000


def tone(s):
    t = np.arange(int(s * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(s):
    return np.zeros(int(s * SR), dtype=np.float32)


def send_audio(ws, audio, chunk=512):
    for i in range(0, len(audio), chunk):
        ws.send_bytes(float_to_pcm16(audio[i : i + chunk]))


def recv_until(ws, kind, limit=200):
    seen = []
    for _ in range(limit):
        msg = ws.receive()
        if msg.get("bytes") is not None:
            seen.append(("bytes", len(msg["bytes"])))
            continue
        data = json.loads(msg["text"])
        seen.append((data["type"], data))
        if data["type"] == kind:
            return seen
    raise AssertionError(f"no {kind} in {seen}")


def make_client(delay=0.02, tts_delay=0.0):
    pipe = VoicePipeline(FakeSTT(), Agent(SlowClient(delay=delay)), FakeTTS(delay=tts_delay))
    app = create_app(pipeline=pipe, scorer_factory=EnergyScorer)
    return TestClient(app)


def test_spoken_turn_round_trip():
    with make_client() as client, client.websocket_connect("/ws") as ws:
        send_audio(ws, np.concatenate([silence(0.3), tone(1.0), silence(0.8)]))
        seen = recv_until(ws, "metrics")
        kinds = [k for k, _ in seen]
        assert kinds[:2] == ["vad", "vad"]
        assert "transcript" in kinds and "tool" in kinds and "token" in kinds
        # every sentence event is immediately followed by its audio bytes
        for i, k in enumerate(kinds):
            if k == "sentence":
                assert kinds[i + 1] == "bytes"
        metrics = seen[-1][1]
        assert metrics["sentences"] == 3 and metrics["endpoint_delay_s"] > 0.4


def test_typed_turn():
    with make_client() as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "text", "text": "समय?"}))
        seen = recv_until(ws, "metrics")
        assert seen[0][1]["text"] == "समय?"


def test_barge_in_interrupts_reply():
    with make_client(delay=0.2) as client, client.websocket_connect("/ws") as ws:
        send_audio(ws, np.concatenate([silence(0.3), tone(1.0), silence(0.8)]))
        recv_until(ws, "sentence")  # the agent has started talking
        send_audio(ws, np.concatenate([tone(0.5)]))  # user talks over it
        seen = recv_until(ws, "interrupt")
        seen += recv_until(ws, "vad")
        vad = [d for k, d in seen if k == "vad" and d["state"] == "speech"]
        assert vad and vad[-1]["barge_in"] is True
        metrics = recv_until(ws, "metrics")[-1][1]
        assert metrics["interrupted"] and metrics["sentences"] < 3


def test_health():
    with make_client() as client:
        assert client.get("/health").json()["ok"] is True
        assert "Hinglish voice agent" in client.get("/").text
