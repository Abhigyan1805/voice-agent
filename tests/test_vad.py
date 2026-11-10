import numpy as np

from voice_agent.vad import FRAME, SAMPLE_RATE, EnergyScorer, SpeechEnd, SpeechStart, VadConfig, VadSegmenter, segment


def tone(seconds: float, amp: float = 0.3, f: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def kinds(events):
    return [type(e).__name__ for e in events]


def test_single_utterance_start_and_end():
    audio = np.concatenate([silence(1.0), tone(1.5), silence(1.0)])
    events = segment(audio, EnergyScorer())
    assert kinds(events) == ["SpeechStart", "SpeechEnd"]
    start, end = events
    assert abs(start.t - 1.0) < 0.05
    assert abs(end.t_end - 2.5) < 0.05
    # end is declared only after end_silence_ms of quiet
    assert abs((end.t_detected - end.t_end) - VadConfig().end_silence_ms / 1000) < 0.05
    # utterance = pre-roll + speech + short tail
    expected = 1.5 + VadConfig().pre_roll_ms / 1000 + VadConfig().tail_ms / 1000
    assert abs(end.duration - expected) < 0.07


def test_two_utterances_separated_by_long_pause():
    audio = np.concatenate([silence(0.5), tone(1.0), silence(1.2), tone(0.8), silence(1.0)])
    events = segment(audio, EnergyScorer())
    assert kinds(events) == ["SpeechStart", "SpeechEnd", "SpeechStart", "SpeechEnd"]


def test_short_pause_does_not_split_turn():
    audio = np.concatenate([silence(0.5), tone(0.8), silence(0.3), tone(0.8), silence(1.0)])
    events = segment(audio, EnergyScorer())
    assert kinds(events) == ["SpeechStart", "SpeechEnd"]
    assert events[1].duration > 1.8


def test_click_shorter_than_min_speech_is_ignored():
    audio = np.concatenate([silence(0.5), tone(0.06), silence(1.0)])
    assert segment(audio, EnergyScorer()) == []


def test_chunk_size_does_not_change_result():
    audio = np.concatenate([silence(0.7), tone(1.1), silence(0.9)])
    a = segment(audio, EnergyScorer(), chunk=160)
    b = segment(audio, EnergyScorer(), chunk=7777)
    assert kinds(a) == kinds(b)
    assert a[1].t_end == b[1].t_end and len(a[1].audio) == len(b[1].audio)


def test_flush_closes_open_utterance():
    seg = VadSegmenter(EnergyScorer())
    events = list(seg.feed(np.concatenate([silence(0.3), tone(1.0)])))
    assert kinds(events) == ["SpeechStart"]
    assert seg.in_speech
    tail = list(seg.flush())
    assert kinds(tail) == ["SpeechEnd"] and not seg.in_speech


def test_max_utterance_forces_end():
    cfg = VadConfig(max_utterance_s=2.0)
    events = segment(np.concatenate([tone(5.0), silence(1.0)]), EnergyScorer(), cfg)
    ends = [e for e in events if isinstance(e, SpeechEnd)]
    assert ends[0].forced and ends[0].duration <= 2.0 + FRAME / SAMPLE_RATE


def test_custom_scorer_and_hysteresis():
    # scores: 10 frames speech (0.9), 5 frames of 0.4 (between thresholds -> still speech), then silence
    scores = iter([0.0] * 5 + [0.9] * 10 + [0.4] * 5 + [0.0] * 40)
    seg = VadSegmenter(lambda frame: next(scores))
    events = list(seg.feed(np.zeros(FRAME * 60, dtype=np.float32)))
    assert isinstance(events[0], SpeechStart) and isinstance(events[1], SpeechEnd)
    assert abs(events[1].t_end - 20 * FRAME / SAMPLE_RATE) < 1e-9
