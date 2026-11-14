"""Tool-call loops (with a fake HTTP stream) and the turn pipeline (with fake STT/TTS). No network, no models."""

import json
import threading
import time

import numpy as np

from voice_agent.agent import Agent
from voice_agent.llm import (
    GeminiClient,
    OfflineClient,
    OpenAICompatClient,
    Overloaded,
    TextDelta,
    ToolEvent,
    TurnInfo,
    _SSE,
)
from voice_agent.pipeline import SpeculativeSTT, SpeechAudio, Token, ToolUse, Transcript, TurnDone, VoicePipeline


class FakeSSE:
    """Replays scripted responses, one list of chunks per request, and records request bodies."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.bodies = []

    def stream(self, url, headers, body, info, cancel, max_retries=None):
        info.requests += 1
        self.bodies.append(json.loads(json.dumps(body)))
        self.urls = getattr(self, "urls", []) + [url]
        chunks = self.responses.pop(0)
        if isinstance(chunks, Exception):
            raise chunks
        for chunk in chunks:
            if cancel is not None and cancel.is_set():
                return
            yield chunk


def gem(*parts):
    return {"candidates": [{"content": {"role": "model", "parts": list(parts)}}]}


def test_gemini_tool_round_trip_replays_signature_and_id():
    client = GeminiClient(api_key="test", cache=False)
    call = {"functionCall": {"name": "calculator", "args": {"expression": "23*47"}, "id": "c1"}, "thoughtSignature": "sig"}
    client.http = FakeSSE([[gem(call)], [gem({"text": "जवाब 1081 "}), gem({"text": "है।"})]])
    events = list(client.turn("sys", [], "23 गुणा 47?"))
    tool = [e for e in events if isinstance(e, ToolEvent)][0]
    assert tool.result["result"] == 1081
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "जवाब 1081 है।"
    second = client.http.bodies[1]["contents"]
    assert second[1]["parts"][0]["thoughtSignature"] == "sig"  # model turn replayed verbatim
    fr = second[2]["parts"][0]["functionResponse"]
    assert fr["id"] == "c1" and fr["response"]["result"] == 1081
    assert client.info.requests == 2 and client.info.tool_calls == ["calculator"]


def test_gemini_falls_back_when_primary_overloaded_and_keeps_model_for_tool_round():
    client = GeminiClient(model="m-primary", fallbacks=("m-backup",), api_key="test", cache=False)
    call = {"functionCall": {"name": "calculator", "args": {"expression": "2+2"}}}
    client.http = FakeSSE([Overloaded("503"), [gem(call)], [gem({"text": "4."})]])
    text = "".join(e.text for e in client.turn("sys", [], "2+2?") if isinstance(e, TextDelta))
    assert text == "4."
    assert ["m-primary" in u for u in client.http.urls] == [True, False, False]
    assert all("m-backup" in u for u in client.http.urls[1:])
    assert client.info.model == "m-backup" and client.model == "m-backup"  # sticky for the next turn


def test_gemini_history_roles():
    client = GeminiClient(api_key="test", cache=False)
    client.http = FakeSSE([[gem({"text": "ok"})]])
    list(client.turn("sys", [{"role": "user", "text": "a"}, {"role": "assistant", "text": "b"}], "c"))
    roles = [c["role"] for c in client.http.bodies[0]["contents"]]
    assert roles == ["user", "model", "user"]


def oa(content=None, tool=None):
    delta = {}
    if content:
        delta["content"] = content
    if tool:
        delta["tool_calls"] = [tool]
    return {"choices": [{"delta": delta}]}


def test_openai_compatible_streamed_tool_call_arguments_are_assembled():
    client = OpenAICompatClient(api_key="test", cache=False)
    client.http = FakeSSE(
        [
            [
                oa(tool={"index": 0, "id": "t1", "function": {"name": "calculator", "arguments": '{"expre'}}),
                oa(tool={"index": 0, "function": {"arguments": 'ssion": "2+3"}'}}),
            ],
            [oa("Five."), oa(" Done.")],
        ]
    )
    events = list(client.turn("sys", [], "2+3?"))
    assert [e.result["result"] for e in events if isinstance(e, ToolEvent)] == [5]
    msgs = client.http.bodies[1]["messages"]
    assert msgs[-2]["tool_calls"][0]["function"]["arguments"] == '{"expression": "2+3"}'
    assert msgs[-1] == {"role": "tool", "tool_call_id": "t1", "content": '{"expression": "2+3", "result": 5}'}


def test_sse_cache_replays_without_network(tmp_path):
    sse = _SSE(cache=True, cache_dir=tmp_path)
    key = sse._key("https://x/y?key=secret", {"a": 1})
    (tmp_path / f"{key}.json").write_text(json.dumps([{"n": 1}, {"n": 2}]))
    info = TurnInfo()
    assert list(sse.stream("https://x/y?key=other", {}, {"a": 1}, info, None)) == [{"n": 1}, {"n": 2}]
    assert info.cached


def test_offline_client_uses_tools():
    c = OfflineClient()
    events = list(c.turn("", [], "23 गुणा 47 कितना है"))
    assert isinstance(events[0], ToolEvent) and events[0].result["result"] == 1081
    assert "1081" in "".join(e.text for e in events if isinstance(e, TextDelta))


# ------------------------------------------------------------------ pipeline


class FakeSTT:
    def transcribe(self, audio, language=None):
        return "हैलो"


class FakeTTS:
    sample_rate = 1000

    def __init__(self, delay=0.0):
        self.delay, self.spoken, self.last_seconds = delay, [], 0.0

    def voice_for(self, text):
        return "hi" if any("ऀ" <= c <= "ॿ" for c in text) else "en"

    def synthesize(self, text, lang=None):
        time.sleep(self.delay)
        self.spoken.append(text)
        self.last_seconds = self.delay
        return np.zeros(100, dtype=np.float32)


class SlowClient:
    """Streams a three-sentence reply token by token with a delay, like a real LLM."""

    info = TurnInfo()

    def __init__(self, delay=0.02):
        self.delay = delay
        self.produced = 0

    def turn(self, system, history, user_text, cancel=None):
        yield ToolEvent("get_datetime", {}, {"time_24h": "09:00"}, 0.0)
        for word in "पहला वाक्य है। दूसरा वाक्य है। तीसरा वाक्य है।".split(" "):
            if cancel is not None and cancel.is_set():
                return
            time.sleep(self.delay)
            self.produced += 1
            yield TextDelta(word + " ")


def test_pipeline_streams_sentences_in_order_with_metrics():
    tts = FakeTTS()
    agent = Agent(SlowClient())
    pipe = VoicePipeline(FakeSTT(), agent, tts)
    events = list(pipe.run_turn(audio=np.zeros(16000, dtype=np.float32)))
    assert isinstance(events[0], Transcript) and events[0].text == "हैलो"
    assert any(isinstance(e, ToolUse) for e in events)
    audio = [e for e in events if isinstance(e, SpeechAudio)]
    assert [a.text for a in audio] == ["पहला वाक्य है।", "दूसरा वाक्य है।", "तीसरा वाक्य है।"]
    m = events[-1].metrics
    assert isinstance(events[-1], TurnDone) and m.sentences == 3 and m.tools == ["get_datetime"]
    assert m.llm_first_token_s <= m.llm_first_sentence_s <= m.time_to_first_audio_s <= m.total_s
    # first audio arrives well before the whole reply has been generated
    assert m.time_to_first_audio_s < m.llm_total_s
    assert agent.history[-1]["text"] == "पहला वाक्य है। दूसरा वाक्य है। तीसरा वाक्य है।"


def test_first_sentence_audio_is_ready_before_llm_finishes():
    pipe = VoicePipeline(None, Agent(SlowClient(delay=0.05)), FakeTTS())
    seen_tokens_at_first_audio = None
    tokens = 0
    for e in pipe.run_turn(text="hi"):
        if isinstance(e, Token):
            tokens += 1
        if isinstance(e, SpeechAudio) and seen_tokens_at_first_audio is None:
            seen_tokens_at_first_audio = tokens
    assert seen_tokens_at_first_audio is not None and seen_tokens_at_first_audio < 9


def test_barge_in_cancels_generation_and_marks_history():
    client = SlowClient(delay=0.05)
    agent = Agent(client)
    tts = FakeTTS()
    pipe = VoicePipeline(None, agent, tts)
    cancel = threading.Event()
    events = []
    for e in pipe.run_turn(text="hi", cancel=cancel):
        events.append(e)
        if isinstance(e, SpeechAudio):
            cancel.set()  # user starts talking while the first sentence plays
    time.sleep(0.2)
    m = events[-1].metrics
    assert m.interrupted and m.sentences == 1
    assert client.produced < 9  # generation stopped early
    assert agent.history[-1]["text"].endswith("[interrupted]")


def test_llm_error_becomes_spoken_apology():
    class Broken:
        info = TurnInfo()

        def turn(self, *a, **k):
            raise RuntimeError("boom")
            yield

    tts = FakeTTS()
    events = list(VoicePipeline(None, Agent(Broken()), tts).run_turn(text="नमस्ते"))
    m = events[-1].metrics
    assert m.error and "boom" in m.error and m.sentences == 1 and tts.spoken[0].startswith("माफ़")


def test_filler_spoken_during_tool_call():
    tts = FakeTTS()
    events = list(VoicePipeline(None, Agent(SlowClient()), tts, filler=True).run_turn(text="समय?"))
    assert tts.spoken[0] == "एक सेकंड।"
    m = events[-1].metrics
    assert m.time_to_filler_audio_s is not None and m.time_to_filler_audio_s <= m.time_to_first_audio_s
    assert m.sentences == 3  # filler is not counted as an answer sentence


def test_speculative_transcript_is_reused():
    class SlowSTT(FakeSTT):
        calls = 0

        def transcribe(self, audio, language=None):
            SlowSTT.calls += 1
            time.sleep(0.2)
            return "हैलो"

    stt = SlowSTT()
    job = SpeculativeSTT(stt, np.zeros(10, dtype=np.float32), pause_id=1)
    time.sleep(0.25)  # the VAD hangover elapses while the decode runs
    pipe = VoicePipeline(stt, Agent(SlowClient(delay=0)), FakeTTS())
    events = list(pipe.run_turn(speculative=job))
    m = events[-1].metrics
    assert SlowSTT.calls == 1 and m.transcript == "हैलो" and m.stt_speculative
    assert m.stt_s < 0.1 <= m.stt_compute_s


def test_auto_prefers_deepseek_then_gemini_then_offline(monkeypatch):
    from voice_agent.llm import make_client

    for k in ("DEEPSEEK_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "VOICE_AGENT_LLM_THINKING",
              "DEEPSEEK_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    assert isinstance(make_client("auto"), OfflineClient)
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    assert isinstance(make_client("auto"), GeminiClient)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    assert isinstance(make_client("auto"), OpenAICompatClient)


def test_deepseek_thinking_defaults_off_and_can_be_overridden(monkeypatch):
    monkeypatch.delenv("VOICE_AGENT_LLM_THINKING", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    client = OpenAICompatClient(api_key="k", cache=False)
    client.http = FakeSSE([[{"choices": [{"delta": {"content": "hi"}}]}]])
    list(client.turn("sys", [], "hello"))
    assert client.http.bodies[0]["thinking"] == {"type": "disabled"}
    monkeypatch.setenv("VOICE_AGENT_LLM_THINKING", "on")
    assert OpenAICompatClient(api_key="k", cache=False).thinking == "on"
    monkeypatch.delenv("VOICE_AGENT_LLM_THINKING")
    other = OpenAICompatClient(api_key="k", cache=False, base_url="http://localhost:8000/v1")
    other.http = FakeSSE([[{"choices": [{"delta": {"content": "hi"}}]}]])
    list(other.turn("sys", [], "hello"))
    assert "thinking" not in other.http.bodies[0]


def test_tool_section_names_every_tool_and_the_no_tool_case():
    from voice_agent.agent import SYSTEM_PROMPT, SYSTEM_PROMPT_V1

    for name in ("calculator", "get_datetime", "search_knowledge"):
        assert name in SYSTEM_PROMPT and name in SYSTEM_PROMPT_V1
    assert "Never answer these from memory" in SYSTEM_PROMPT and "No tool for general knowledge" in SYSTEM_PROMPT
    assert SYSTEM_PROMPT_V1.startswith(SYSTEM_PROMPT[:200])
