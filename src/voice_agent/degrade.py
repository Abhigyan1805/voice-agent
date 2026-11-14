"""Simulated telephone channel: band-limit, 8 kHz, G.711 mu-law, low-bitrate codec, noise, packet loss.

All functions take and return 16 kHz mono float32 and are deterministic (noise and loss are
seeded from the input audio itself, so the same clip always gets the same degradation).
Filters are windowed-sinc FIRs in numpy; the codec goes through PyAV (libopus). GSM and AMR
encoders are not available in this PyAV build, so Opus at 8 kbit/s narrowband stands in for a
low-bitrate mobile codec. Noise is added at the caller's side, before the channel.

Conditions (``make_condition``), as listed in evals/conditions.json:
  clean          no change
  tel            300-3400 Hz band-pass, 8 kHz, mu-law, back to 16 kHz
  tel+opus8      tel, then Opus at 8 kbit/s
  tel+babbleN    babble noise (6 FLEURS speakers from outside the evaluation slice) at N dB SNR, then tel
  tel+whiteN     white noise at N dB SNR, then tel
  tel+lossN      tel, then N % of 20 ms frames zeroed
"""

from __future__ import annotations

import io
import re
import zlib
from functools import lru_cache
from typing import Callable

import numpy as np

SR = 16_000
TEL_SR = 8_000


# ------------------------------------------------------------------ filters and rates


def lowpass_kernel(cutoff_hz: float, sr: int, taps: int = 129) -> np.ndarray:
    """Blackman-windowed sinc low-pass with unit DC gain."""
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * cutoff_hz / sr * n) * np.blackman(taps)
    return (h / h.sum()).astype(np.float64)


def bandpass_kernel(lo_hz: float, hi_hz: float, sr: int, taps: int = 257) -> np.ndarray:
    return lowpass_kernel(hi_hz, sr, taps) - lowpass_kernel(lo_hz, sr, taps)


def _filter(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.convolve(x.astype(np.float64), h, mode="same")


def bandpass(audio: np.ndarray, lo_hz: float = 300.0, hi_hz: float = 3400.0, sr: int = SR) -> np.ndarray:
    return _filter(audio, bandpass_kernel(lo_hz, hi_hz, sr)).astype(np.float32)


def down2(audio: np.ndarray) -> np.ndarray:
    """16 kHz -> 8 kHz with an anti-alias low-pass (the repo's linear resampler has none)."""
    return _filter(audio, lowpass_kernel(3_700, SR))[::2].astype(np.float32)


def up2(audio: np.ndarray) -> np.ndarray:
    """8 kHz -> 16 kHz: zero-stuff and low-pass (gain 2 restores the level)."""
    y = np.zeros(len(audio) * 2, dtype=np.float64)
    y[::2] = audio
    return (2 * _filter(y, lowpass_kernel(3_900, SR))).astype(np.float32)


# ------------------------------------------------------------------ G.711 mu-law


def mulaw_encode(x: np.ndarray, mu: int = 255) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.round((y + 1) / 2 * mu).astype(np.uint8)


def mulaw_decode(q: np.ndarray, mu: int = 255) -> np.ndarray:
    y = q.astype(np.float64) / mu * 2 - 1
    return (np.sign(y) * np.expm1(np.abs(y) * np.log1p(mu)) / mu).astype(np.float32)


def telephone(audio: np.ndarray) -> np.ndarray:
    """Band-pass 300-3400 Hz, 8 kHz, mu-law round trip, back to 16 kHz."""
    narrow = down2(bandpass(audio))
    return up2(mulaw_decode(mulaw_encode(narrow)))[: len(audio)]


# ------------------------------------------------------------------ codec (PyAV)


def opus(audio: np.ndarray, bitrate: int = 8_000) -> np.ndarray:
    """Encode and decode at 8 kHz with libopus (narrowband) at ``bitrate`` bit/s; returns 16 kHz."""
    import av

    pcm = (np.clip(down2(audio), -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with av.open(buf, "w", format="ogg") as out:
        st = out.add_stream("libopus", rate=TEL_SR)
        st.bit_rate = bitrate
        st.layout = "mono"
        frame_len = 160  # 20 ms
        pad = (-len(pcm)) % frame_len
        pcm = np.concatenate([pcm, np.zeros(pad, np.int16)])
        for i in range(0, len(pcm), frame_len):
            fr = av.AudioFrame.from_ndarray(pcm[None, i : i + frame_len], format="s16", layout="mono")
            fr.sample_rate = TEL_SR
            fr.pts = i
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)
    buf.seek(0)
    chunks = []
    with av.open(buf, "r") as inp:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=TEL_SR)
        for frame in inp.decode(audio=0):
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
    dec = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    y = up2(dec)
    y = np.concatenate([y, np.zeros(max(0, len(audio) - len(y)), np.float32)])
    return y[: len(audio)].astype(np.float32)


# ------------------------------------------------------------------ noise and loss


def _rng(audio: np.ndarray, salt: str) -> np.random.Generator:
    seed = zlib.crc32(np.asarray(audio, np.float32).tobytes()) ^ zlib.crc32(salt.encode())
    return np.random.default_rng(seed)


def speech_power(audio: np.ndarray, threshold: float = 0.5) -> float:
    """Mean power over the frames the Silero VAD scores as speech (all frames if it finds none)."""
    from .vad import FRAME, SileroScorer

    scorer = _scorer()
    scorer.reset()
    frames = [audio[i : i + FRAME] for i in range(0, len(audio) - FRAME + 1, FRAME)]
    speech = [f for f in frames if scorer(f) >= threshold]
    x = np.concatenate(speech) if speech else audio
    return float(np.mean(np.square(x, dtype=np.float64)) + 1e-12)


@lru_cache(maxsize=1)
def _scorer():
    from .vad import SileroScorer

    return SileroScorer()


@lru_cache(maxsize=1)
def babble_source(n_speakers: int = 6) -> np.ndarray:
    """Babble: n FLEURS hi_in test clips from outside every evaluation slice (300-307), each scaled
    to the same level and summed, looped to a long buffer."""
    from .fleurs import load_clips

    clips = load_clips(300, 308)[:n_speakers]
    n = max(len(c.audio) for c in clips)
    mix = np.zeros(n, np.float64)
    for k, c in enumerate(clips):
        a = c.audio / (np.sqrt(np.mean(np.square(c.audio, dtype=np.float64))) + 1e-9)
        mix += np.roll(np.resize(a, n), k * n // n_speakers)  # offset speakers so pauses do not line up
    return (mix / np.sqrt(np.mean(mix**2))).astype(np.float32)


def add_noise(audio: np.ndarray, kind: str, snr_db: float) -> np.ndarray:
    rng = _rng(audio, f"{kind}{snr_db}")
    if kind == "white":
        noise = rng.standard_normal(len(audio)).astype(np.float32)
    elif kind == "babble":
        src = babble_source()
        start = int(rng.integers(0, len(src)))
        noise = np.resize(np.roll(src, -start), len(audio))
    else:
        raise ValueError(f"unknown noise {kind!r}")
    p_noise = float(np.mean(np.square(noise, dtype=np.float64)) + 1e-12)
    gain = np.sqrt(speech_power(audio) / (p_noise * 10 ** (snr_db / 10)))
    return np.clip(audio + gain * noise, -1.0, 1.0).astype(np.float32)


def packet_loss(audio: np.ndarray, rate: float, frame_ms: float = 20.0) -> np.ndarray:
    rng = _rng(audio, f"loss{rate}")
    n = int(SR * frame_ms / 1000)
    y = audio.copy()
    for i in range(0, len(y), n):
        if rng.random() < rate:
            y[i : i + n] = 0.0
    return y


# ------------------------------------------------------------------ conditions


def make_condition(name: str) -> Callable[[np.ndarray], np.ndarray]:
    steps: list[Callable[[np.ndarray], np.ndarray]] = []
    for part in name.split("+"):
        if part == "clean":
            continue
        if part == "tel":
            steps.append(telephone)
        elif part == "opus8":
            steps.append(lambda a: opus(a, 8_000))
        elif m := re.fullmatch(r"(babble|white)(\d+)", part):
            kind, snr = m.group(1), float(m.group(2))
            steps.insert(0, lambda a, kind=kind, snr=snr: add_noise(a, kind, snr))  # caller's side
        elif m := re.fullmatch(r"loss(\d+)", part):
            rate = int(m.group(1)) / 100
            steps.append(lambda a, rate=rate: packet_loss(a, rate))
        else:
            raise ValueError(f"unknown condition part {part!r}")

    def apply(audio: np.ndarray) -> np.ndarray:
        for f in steps:
            audio = f(audio)
        return audio.astype(np.float32)

    apply.__name__ = name
    return apply
