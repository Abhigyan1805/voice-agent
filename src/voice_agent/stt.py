"""Speech-to-text with the fine-tuned Hindi Whisper-small, int8 CTranslate2 build, via faster-whisper.

The model was fine-tuned only with the Hindi language token. Forced to Hindi it writes English
speech as Devanagari gibberish, while letting faster-whisper auto-detect costs a second encoder
pass. So ``transcribe`` runs the encoder **once**, reads Whisper's language-ID head from that
encoder output, picks Hindi or English (the only two the agent supports), and decodes from the
same encoder output.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

# Devanagari costs roughly one Whisper token per character, so a 15 s Hindi sentence can need
# 300+ tokens. Same cap as the fine-tuning repo; generation stops at the end token long before
# this for a short spoken question, so the cap costs nothing there.
MAX_NEW_TOKENS = 440
MAX_LENGTH = 448  # Whisper's decoder context
LANGS = ("hi", "en")


def collapse_repeats(text: str, max_repeat: int = 3) -> str:
    """Cut runaway greedy loops ("पर्णी पर्णी पर्णी ...") to at most ``max_repeat`` copies of an n-gram."""
    words = text.split()
    for n in range(1, 5):
        out: list[str] = []
        i = 0
        while i < len(words):
            gram = words[i : i + n]
            reps = 1
            while words[i + reps * n : i + (reps + 1) * n] == gram:
                reps += 1
            out.extend(gram * min(reps, max_repeat) if len(gram) == n else gram)
            i += n * reps if len(gram) == n else len(gram)
        words = out
    return " ".join(words)


class WhisperSTT:
    def __init__(self, model_dir: str | Path, threads: int = 4):
        from faster_whisper import WhisperModel

        model_dir = Path(model_dir)
        if not (model_dir / "model.bin").exists():
            raise FileNotFoundError(
                f"no CTranslate2 model at {model_dir}; run `python scripts/build_stt_model.py` (see README)"
            )
        self.model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", cpu_threads=threads)
        self._tokenizers: dict = {}
        self.last_seconds = 0.0
        self.last_language = "hi"
        self.last_language_probs: dict[str, float] = {}

    def _tokenizer(self, lang: str):
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import get_suppressed_tokens

        if lang not in self._tokenizers:
            tok = Tokenizer(self.model.hf_tokenizer, True, task="transcribe", language=lang)
            self._tokenizers[lang] = (tok, get_suppressed_tokens(tok, [-1]))
        return self._tokenizers[lang]

    def transcribe(self, audio: np.ndarray, language: str | None = "auto") -> str:
        """Greedy decode of one VAD-cut utterance (16 kHz mono float32, up to 30 s).

        ``language``: "hi" or "en" to force, "auto"/None to choose between them per utterance.
        """
        from faster_whisper.audio import pad_or_trim

        t0 = time.perf_counter()
        fe = self.model.feature_extractor
        features = pad_or_trim(fe(audio[: fe.n_samples])[..., : fe.nb_max_frames])
        enc = self.model.encode(features)
        if language in (None, "auto"):
            probs = {tok[2:-2]: p for tok, p in self.model.model.detect_language(enc)[0]}
            self.last_language_probs = {k: round(probs.get(k, 0.0), 4) for k in LANGS}
            lang = max(LANGS, key=lambda k: probs.get(k, 0.0))
        else:
            lang = language
        tok, suppress = self._tokenizer(lang)
        prompt = list(tok.sot_sequence) + [tok.no_timestamps]
        result = self.model.model.generate(
            enc,
            [prompt],
            beam_size=1,
            max_length=min(MAX_LENGTH, len(prompt) + MAX_NEW_TOKENS),
            suppress_blank=True,
            suppress_tokens=suppress,
        )[0]
        text = collapse_repeats(tok.decode(result.sequences_ids[0]).strip())
        self.last_language = lang
        self.last_seconds = time.perf_counter() - t0
        return re.sub(r"\s+", " ", text).strip()
