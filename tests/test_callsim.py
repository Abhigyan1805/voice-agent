"""Call-suite scoring and failure attribution, with the offline LLM and a fake STT. No models, no network."""

from datetime import date

import numpy as np

from voice_agent.agent import Agent
from voice_agent.audio import write_wav
from voice_agent.callsim import (
    CallRunner,
    attribute,
    check_turn,
    expand,
    reply_language,
    spoken_digits,
    summarise,
    wer_cer,
)
from voice_agent.demo_service import ServiceThread
from voice_agent.llm import OfflineClient
from voice_agent.pipeline import VoicePipeline
from voice_agent.vad import EnergyScorer

TODAY = date(2026, 11, 14)
WORDS = {"shipped": "shipped|भेज"}


class ScriptedSTT:
    def __init__(self, texts):
        self.texts = list(texts)
        self.last_language = "hi"

    def transcribe(self, audio, language=None):
        return self.texts.pop(0) if self.texts else ""


class FakeTTS:
    sample_rate = 16_000
    last_seconds = 0.0

    def voice_for(self, text):
        return "hi"

    def synthesize(self, text, lang=None):
        return np.zeros(1600, np.float32)


def test_normalisation_helpers():
    assert expand(r"\b{d+2}\b @shipped {iso-3}", TODAY, WORDS) == r"\b16\b (?:shipped|भेज) 2026-11-11"
    assert spoken_digits("मेरा नंबर चार आठ दो 1 ३ है") == "48213"
    assert reply_language("आपका order 48213 shipped है") == "hi" and reply_language("It is shipped.") == "en"
    assert wer_cer("नंबर चार आठ दो", "नंबर 482") == (0.0, 0.0)


def test_check_turn_tools_forbid_reply_and_language():
    tools = [{"name": "get_order_status", "args": {"order_id": "४८२१३"}}]
    expect = {"tools": [{"name": "get_order_status", "args": {"order_id": "48213"}}], "forbid": ["reschedule_delivery"],
              "reply_any": ["@shipped"], "reply_none": ["delivered"], "reply_lang": "hi"}
    assert all(check_turn(expect, tools, "आपका ऑर्डर भेज दिया गया है, 16 नवंबर तक आएगा।", TODAY, WORDS).values())
    bad = check_turn(expect, tools + [{"name": "reschedule_delivery", "args": {}}], "It was delivered.", TODAY, WORDS)
    assert bad == {"tools": True, "forbid": False, "reply_any": False, "reply_none": False, "reply_lang": False}
    assert check_turn({"tools": []}, tools, "", TODAY, WORDS) == {"tools": False}


def _turn(passed, **kw):
    t = {"passed": passed, "text_passed": passed, "checks": {"tools": True}, "unplanned_tool_error": False,
         "tool_over_budget": False, "vad_no_speech": False, "vad_split": False, "tts_roundtrip_ok": None}
    t.update(kw)
    return t


def _call(turns, final_ok=True):
    return {"turns": turns, "final_state_ok": final_ok}


def test_attribution_rules_one_per_stage():
    ok = _call([_turn(True), _turn(True)])
    cases = {
        "tool": _call([_turn(False, unplanned_tool_error=True)]),
        "vad": _call([_turn(False, vad_no_speech=True)]),
        "tts": _call([_turn(False, text_passed=True, tts_roundtrip_ok=False)]),
        "stt": _call([_turn(False, checks={"tools": False})]),
    }
    for stage, spoken in cases.items():
        assert attribute(spoken, ok)["stage"] == stage
    assert attribute(_call([_turn(False)]), _call([_turn(False)]))["stage"] == "llm"
    spoken = _call([_turn(True), _turn(False), _turn(False)])
    assert attribute(spoken, _call([_turn(True), _turn(True), _turn(True)])) == {"stage": "stt", "turn": 1}
    assert [t["attribution"] for t in spoken["turns"]] == [None, "stt", "cascade"]
    assert attribute(_call([_turn(True)], final_ok=False), _call([_turn(True)], final_ok=False))["stage"] == "llm"
    assert attribute(ok, ok) == {"stage": None, "turn": None}


def test_runner_spoken_and_oracle_tracks(tmp_path):
    tone = (0.3 * np.sin(np.arange(16_000) / 16_000 * 2 * np.pi * 220)).astype(np.float32)
    audio = np.concatenate([np.zeros(8000, np.float32), tone, np.zeros(16_000, np.float32)])
    write_wav(tmp_path / "t-0.wav", audio, 16_000)
    write_wav(tmp_path / "t-1.wav", np.zeros(16_000, np.float32), 16_000)  # silence: VAD finds no speech
    call = {"id": "t", "lang": "en", "turns": [
        {"user": "what is 6 times 7", "expect": {"tools": [{"name": "calculator"}], "reply_any": ["42"]}},
        {"user": "what is 2 plus 2", "expect": {"reply_any": ["4"]}},
    ]}
    stt = ScriptedSTT(["what is 6 times 7", "the answer is 42"])  # turn 0, then its TTS round trip
    pipe = VoicePipeline(stt, Agent(OfflineClient()), FakeTTS())
    with ServiceThread() as svc:
        runner = CallRunner(pipe, svc, scorer=EnergyScorer(), audio_dir=tmp_path, today=TODAY, words=WORDS)
        spoken = runner.run_call(call, "spoken")
        oracle = runner.run_call(call, "oracle")
    t0 = spoken["turns"][0]
    assert t0["passed"] and t0["wer"] == 0.0, {k: t0[k] for k in ("checks", "tts_roundtrip_ok", "transcript", "reply")}
    assert spoken["turns"][1]["vad_no_speech"] and not spoken["turns"][1]["passed"]
    assert all(t["passed"] for t in oracle["turns"])
    spoken["attribution"] = attribute(spoken, oracle)
    assert spoken["attribution"] == {"stage": "vad", "turn": 1}
    oracle["attribution"] = {"stage": None, "turn": None}
    s = summarise([spoken, oracle])
    assert s["calls_passed_spoken"] == "0/1" and s["calls_passed_oracle"] == "1/1" and s["failures_by_stage"] == {"vad": 1}
