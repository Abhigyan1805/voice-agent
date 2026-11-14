"""Spoken-number normalisation of recogniser output."""

import pytest

from voice_agent.numbers import normalize_numbers as n


@pytest.mark.parametrize("heard, want", [
    ("मेरा ऑर्डर नंबर है चार आठ दो एक तीन।", "मेरा ऑर्डर नंबर है 48213।"),
    ("order number is six zero three one eight.", "order number is 60318."),
    ("my number is nine one two three four five six seven eight zero", "my number is 9123456780"),
    ("ऑडर 6, 8, 3, 4, 6 ने वी", "ऑडर 68346 ने वी"),
    ("मेरे AUD 4881-3 की", "मेरे AUD 48813 की"),
    ("order 4 8 2 1 3 please", "order 48213 please"),
    ("double five two oh one", "55201"),
    ("डबल पाँच दो शून्य एक", "55201"),
    ("नंबर ४८२१३ है", "नंबर ४८२१३ है"),  # a single group is already canonical
    ("phone 98765 43210", "phone 9876543210"),
])
def test_digit_runs_are_joined(heard, want):
    assert n(heard).translate(str.maketrans("०१२३४५६७८९", "0123456789")) == want.translate(
        str.maketrans("०१२३४५६७८९", "0123456789"))


@pytest.mark.parametrize("text", [
    "अगर 1200 रुपये के बैग पर 15 प्रतिशत डिस्काउंट मिले",
    "एक बार फिर से कोशिश कीजिए",
    "दो दिन बाद आना",
    "आप गोमा के आस-पास घूमने के लिए",
    "What is the square root of 144 plus 17?",
    "one moment please",
    "20 नवंबर को कर दो",
])
def test_ordinary_text_is_untouched(text):
    assert n(text) == text


def test_digit_prompt_only_after_a_number_request(monkeypatch):
    from voice_agent.numbers import DIGIT_PROMPTS, stt_prompt

    monkeypatch.delenv("VOICE_AGENT_DIGIT_FIX", raising=False)
    assert stt_prompt([]) is None
    asked_hi = [{"role": "user", "text": "स्टेटस"}, {"role": "assistant", "text": "अपना 5 अंकों का ऑर्डर नंबर बताइए?"}]
    assert stt_prompt(asked_hi) == DIGIT_PROMPTS["hi"]
    asked_en = [{"role": "assistant", "text": "Could you tell me your order number?"}]
    assert stt_prompt(asked_en) == DIGIT_PROMPTS["en"]
    assert stt_prompt([{"role": "assistant", "text": "आपका ऑर्डर शिप हो चुका है।"}]) is None
    monkeypatch.setenv("VOICE_AGENT_DIGIT_FIX", "0")
    assert stt_prompt(asked_hi) is None


def test_pipeline_normalises_numbers_before_the_llm(monkeypatch):
    import numpy as np

    from voice_agent.agent import Agent
    from voice_agent.pipeline import TurnDone, VoicePipeline

    seen = []

    class EchoClient:
        from voice_agent.llm import TurnInfo

        info = TurnInfo()

        def turn(self, system, history, user_text, cancel=None):
            seen.append(user_text)
            return iter(())

    class STT:
        last_language = "hi"

        def transcribe(self, audio, language=None, prompt=None):
            return "ऑर्डर चार आठ दो एक तीन"

    monkeypatch.delenv("VOICE_AGENT_DIGIT_FIX", raising=False)
    events = list(VoicePipeline(STT(), Agent(EchoClient()), None).run_turn(audio=np.zeros(16000, np.float32)))
    m = events[-1].metrics
    assert isinstance(events[-1], TurnDone) and seen == ["ऑर्डर 48213"]
    assert m.transcript == "ऑर्डर चार आठ दो एक तीन" and m.llm_input == "ऑर्डर 48213"
