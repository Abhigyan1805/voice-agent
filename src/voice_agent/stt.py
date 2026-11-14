"""Speech-to-text with the fine-tuned Hindi Whisper-small, int8 CTranslate2 build, via faster-whisper.

The model was fine-tuned only with the Hindi language token. Forced to Hindi it writes English
speech as Devanagari gibberish, while letting faster-whisper auto-detect costs a second encoder
pass. So ``transcribe`` runs the encoder **once**, reads Whisper's language-ID head from that
encoder output, picks Hindi or English (the only two the agent supports), and decodes from the
same encoder output.

Encoder window: Whisper is trained on 30 s windows, and encoding 30 s costs about 1.3 s on this
CPU even for a 2 s question. For utterances shorter than ``short_window_max_s`` the features are
padded only to the utterance length plus ``pad_s`` seconds. With too little trailing padding the
decoder can miss the end token and loop, so a short-window decode that loops or produces more
tokens than the audio can plausibly hold is decoded again with the full 30 s window (the
"fallback"). ``pad_s = 0`` restores the plain 30 s behaviour.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import numpy as np

# Devanagari costs roughly one Whisper token per character. We request up to 444 tokens, but
# CTranslate2 stops Whisper generation at half of the 448-token decoder context: every runaway
# decode ends at exactly TOKEN_LIMIT generated tokens. At a median ~13 tokens per second of Hindi
# speech, sentences longer than about 17 s can be cut; the WER evaluation counts how often.
MAX_NEW_TOKENS = 440
MAX_LENGTH = 448  # Whisper's decoder context
TOKEN_LIMIT = 222  # generated tokens at which decoding is observed to stop
LANGS = ("hi", "en")
FRAMES_PER_S = 100  # mel frames per second (hop 160 at 16 kHz)
FULL_FRAMES = 3000  # 30 s window
# Fallback bound: Hindi on this model needs a median ~13 tokens per second of speech.
MAX_TOKENS_PER_S = 14.0
MAX_TOKENS_SLACK = 20


def window_frames(n_samples: int, pad_s: float, short_window_max_s: float, sr: int = 16_000) -> int:
    """Mel frames to encode for an utterance of ``n_samples``: utterance + ``pad_s`` when short, else 30 s."""
    dur = n_samples / sr
    if pad_s <= 0 or dur >= short_window_max_s:
        return FULL_FRAMES
    n = math.ceil((dur + pad_s) * FRAMES_PER_S)
    n += n % 2  # the encoder's strided conv halves the length
    return min(FULL_FRAMES, n)


def needs_fallback(raw_text: str, n_tokens: int, duration_s: float) -> bool:
    """A short-window decode is suspect if it looped, hit the token limit, or is implausibly long."""
    looped = collapse_repeats(raw_text) != raw_text
    too_long = n_tokens > MAX_TOKENS_PER_S * duration_s + MAX_TOKENS_SLACK
    return looped or too_long or n_tokens >= TOKEN_LIMIT


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
    def __init__(self, model_dir: str | Path, threads: int = 4, pad_s: float = 5.0, short_window_max_s: float = 12.0):
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
        self.pad_s = pad_s
        self.short_window_max_s = short_window_max_s
        self.last_window_s = 30.0
        self.last_fallback = False
        self.last_tokens = 0
        self.last_hit_token_limit = False

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
        audio = audio[: fe.n_samples]
        feats = fe(audio)[..., : fe.nb_max_frames]
        n = window_frames(len(audio), self.pad_s, self.short_window_max_s)
        raw, ids, lang = self._decode(pad_or_trim(feats, n), language)
        fallback = False
        if n < FULL_FRAMES and needs_fallback(raw, len(ids), len(audio) / 16_000):
            fallback, n = True, FULL_FRAMES
            raw, ids, lang = self._decode(pad_or_trim(feats, n), language)
        self.last_window_s = n / FRAMES_PER_S
        self.last_fallback = fallback
        self.last_tokens = len(ids)
        self.last_hit_token_limit = len(ids) >= TOKEN_LIMIT
        self.last_language = lang
        self.last_seconds = time.perf_counter() - t0
        return re.sub(r"\s+", " ", collapse_repeats(raw)).strip()

    def _decode(self, features: np.ndarray, language: str | None) -> tuple[str, list[int], str]:
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
        ids = result.sequences_ids[0]
        return tok.decode(ids).strip(), ids, lang
