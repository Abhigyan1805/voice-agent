"""Small audio helpers: loading, resampling, format conversion."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.dtype == np.int16:
        return (audio.astype(np.float32) / 32768.0).astype(np.float32)
    return audio.astype(np.float32, copy=False)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interpolation resampler; good enough for speech going into Whisper or a speaker."""
    if sr_in == sr_out or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    n_out = int(round(len(audio) * sr_out / sr_in))
    x_old = np.arange(len(audio), dtype=np.float64)
    x_new = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def load_audio(path_or_bytes, sr: int = 16_000) -> np.ndarray:
    """Read a wav/flac/ogg file (path or bytes) as mono float32 at ``sr``."""
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    audio, sr_in = sf.read(src, dtype="float32", always_2d=False)
    return resample(to_mono_float32(audio), sr_in, sr)


def float_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def write_wav(path, audio: np.ndarray, sr: int) -> None:
    sf.write(str(path), audio, sr, subtype="PCM_16")
