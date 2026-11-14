"""Telephone-channel simulation: mu-law error bound, band-pass, SNR, determinism, codec length."""

import numpy as np
import pytest

from voice_agent import degrade
from voice_agent.degrade import SR, add_noise, bandpass, make_condition, mulaw_decode, mulaw_encode, packet_loss, telephone


def tone(hz, seconds=1.0, amp=0.5):
    return (amp * np.sin(2 * np.pi * hz * np.arange(int(SR * seconds)) / SR)).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x[2000:-2000], dtype=np.float64))))


def test_mulaw_round_trip_error_bound():
    x = np.linspace(-1, 1, 10_001).astype(np.float32)
    err = np.abs(mulaw_decode(mulaw_encode(x)) - x)
    assert err.max() < 0.035 and np.mean(err[np.abs(x) < 0.05]) < 0.002  # fine steps near zero


def test_bandpass_keeps_voice_band_and_cuts_outside():
    assert rms(bandpass(tone(1000))) / rms(tone(1000)) > 0.9
    assert rms(bandpass(tone(100))) / rms(tone(100)) < 0.1
    assert rms(bandpass(tone(6000))) / rms(tone(6000)) < 0.05


def test_telephone_removes_high_band_and_keeps_length():
    y = telephone(tone(1000) + tone(6000))
    assert len(y) == SR and abs(rms(y) - rms(tone(1000))) / rms(tone(1000)) < 0.15


def test_snr_within_half_db(monkeypatch):
    monkeypatch.setattr(degrade, "speech_power", lambda a: float(np.mean(np.square(a, dtype=np.float64))))
    clean = tone(440, 2.0, 0.3)
    noisy = add_noise(clean, "white", 10.0)
    snr = 10 * np.log10(np.mean(clean.astype(np.float64) ** 2) / np.mean((noisy - clean).astype(np.float64) ** 2))
    assert abs(snr - 10.0) < 0.5


def test_deterministic_and_loss_rate(monkeypatch):
    monkeypatch.setattr(degrade, "speech_power", lambda a: float(np.mean(np.square(a, dtype=np.float64))))
    a = tone(300, 2.0)
    assert np.array_equal(add_noise(a, "white", 5), add_noise(a, "white", 5))
    lost = packet_loss(np.ones(SR * 20, np.float32), 0.05)
    frames = lost.reshape(-1, 320)
    assert 0.03 < np.mean(frames.max(axis=1) == 0) < 0.07
    assert np.array_equal(make_condition("tel+loss5")(a), make_condition("tel+loss5")(a))


def test_opus_round_trip_keeps_length():
    pytest.importorskip("av")
    y = make_condition("tel+opus8")(tone(800, 1.5, 0.3))
    assert len(y) == int(1.5 * SR) and rms(y) > 0.05
