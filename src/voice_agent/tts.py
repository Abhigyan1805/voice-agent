"""Text-to-speech with Piper (VITS exported to ONNX, ~63 MB per voice, faster than real time on CPU).

One Hindi and one English voice are loaded; each sentence goes to the voice that matches its
script. espeak-ng's Hindi phonemiser reads digits in Hindi and pronounces Latin-script English
words inside a Hindi sentence in English, so code-mixed sentences stay on the Hindi voice.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .text import detect_language


class PiperTTS:
    def __init__(self, hindi_voice: str | Path, english_voice: str | Path | None = None, length_scale: float = 1.0):
        from piper import PiperVoice, SynthesisConfig

        self._cfg = SynthesisConfig(length_scale=length_scale)
        self.voices = {"hi": PiperVoice.load(str(hindi_voice))}
        if english_voice and Path(english_voice).exists():
            self.voices["en"] = PiperVoice.load(str(english_voice))
        rates = {v.config.sample_rate for v in self.voices.values()}
        if len(rates) != 1:
            raise ValueError(f"voices must share a sample rate, got {rates}")
        self.sample_rate = rates.pop()
        self.last_seconds = 0.0

    def voice_for(self, text: str) -> str:
        lang = detect_language(text)
        return lang if lang in self.voices else "hi"

    def synthesize(self, text: str, lang: str | None = None) -> np.ndarray:
        """Synthesise one sentence; returns mono float32 at ``self.sample_rate``."""
        t0 = time.perf_counter()
        voice = self.voices[lang or self.voice_for(text)]
        chunks = [c.audio_float_array for c in voice.synthesize(text, syn_config=self._cfg)]
        self.last_seconds = time.perf_counter() - t0
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
