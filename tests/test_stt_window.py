"""Short encoder window arithmetic and the fallback decision, with a fake model. No model files."""

import numpy as np

from voice_agent.stt import FULL_FRAMES, TOKEN_LIMIT, WhisperSTT, needs_fallback, window_frames


def test_window_is_utterance_plus_pad_and_even():
    assert window_frames(3 * 16_000, 5.0, 12.0) == 800
    assert window_frames(int(2.345 * 16_000), 5.0, 12.0) % 2 == 0
    assert window_frames(int(2.345 * 16_000), 5.0, 12.0) >= 734


def test_long_utterances_and_pad_zero_use_full_window():
    assert window_frames(12 * 16_000, 5.0, 12.0) == FULL_FRAMES
    assert window_frames(20 * 16_000, 5.0, 12.0) == FULL_FRAMES
    assert window_frames(2 * 16_000, 0.0, 12.0) == FULL_FRAMES
    assert window_frames(11 * 16_000, 25.0, 30.0) == FULL_FRAMES  # never more than 30 s


def test_fallback_decision():
    assert not needs_fallback("आज कौन सा दिन है", 8, 1.7)
    assert needs_fallback("नमस्ते " * 10, 30, 3.0)  # loop collapsed by collapse_repeats
    assert needs_fallback("a b c d e f g", 14 * 2 + 21, 2.0)  # more tokens than 2 s can hold
    assert needs_fallback("x", TOKEN_LIMIT, 30.0)  # stopped at the generation limit


class FakeFE:
    n_samples = 480_000
    nb_max_frames = 3000

    def __call__(self, audio):
        return np.zeros((80, len(audio) // 160 + 1), np.float32)


class FakeModel:
    feature_extractor = FakeFE()


def make_stt(decodes):
    stt = object.__new__(WhisperSTT)
    stt.model = FakeModel()
    stt.pad_s, stt.short_window_max_s = 5.0, 12.0
    seen = []

    def _decode(features, language):
        seen.append(features.shape[-1])
        return decodes.pop(0)

    stt._decode = _decode
    return stt, seen


def test_short_window_kept_when_clean():
    stt, seen = make_stt([("आज कौन सा दिन है", [1] * 8, "hi")])
    assert stt.transcribe(np.zeros(2 * 16_000, np.float32)) == "आज कौन सा दिन है"
    assert seen == [700] and not stt.last_fallback and stt.last_window_s == 7.0


def test_runaway_short_decode_is_redone_at_30s():
    stt, seen = make_stt([("हाँ " * 60, [1] * 120, "hi"), ("हाँ ठीक है", [1] * 5, "hi")])
    assert stt.transcribe(np.zeros(2 * 16_000, np.float32)) == "हाँ ठीक है"
    assert seen == [700, FULL_FRAMES] and stt.last_fallback and stt.last_window_s == 30.0
