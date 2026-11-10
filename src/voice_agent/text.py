"""Text plumbing between the LLM and the TTS: sentence streaming, cleanup and script detection.

``SentenceStreamer`` is what lets the agent start talking before the reply is finished: LLM
token deltas go in, and complete sentences come out as soon as their end is seen, so the first
sentence can be synthesised while the rest of the reply is still being generated.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Iterator

# Sentence enders: Latin . ! ?, Devanagari danda / double danda, and the full-width forms.
_ENDERS = ".!?।॥？！"
# Soft split points used only when a sentence grows past max_chars.
_SOFT = ",;:—–،"
_ABBREV = {"dr", "mr", "mrs", "ms", "st", "vs", "etc", "e.g", "i.e", "no", "prof", "sr", "jr", "approx"}
_MARKDOWN_RE = re.compile(r"[*_`#>|~]+")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_SPACE_RE = re.compile(r"\s+")
_BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+")


def clean_for_speech(text: str) -> str:
    """Remove markdown, emoji and other things a TTS voice should not read out."""
    text = _BULLET_RE.sub("", _LINK_RE.sub(r"\1", text))
    text = _MARKDOWN_RE.sub(" ", text)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "So" or cat == "Cs" or ch in "‍️":
            continue  # emoji and pictographs
        out.append(ch)
    return _SPACE_RE.sub(" ", "".join(out)).strip()


def _is_boundary(buf: str, i: int) -> bool:
    """Is buf[i] (an ender) the end of a sentence? Needs whitespace (or a closing quote) after it."""
    ch = buf[i]
    nxt = buf[i + 1] if i + 1 < len(buf) else ""
    if not nxt:
        return False  # can't tell yet: "3." might become "3.14"
    if nxt in "\"'”’)]":
        nxt = buf[i + 2] if i + 2 < len(buf) else ""
        if not nxt:
            return False
    if not nxt.isspace():
        return False
    if ch == ".":
        word = re.split(r"\s", buf[:i])[-1].lower().lstrip("(\"'")
        if word in _ABBREV or (len(word) == 1 and word.isalpha()):
            return False  # "Dr. Rao", "A. P. J."
    return True


class SentenceStreamer:
    """Accumulate streamed text and yield speakable sentences.

    * A sentence ends at ``. ! ? । ॥`` followed by whitespace, or at a newline.
    * Pieces shorter than ``min_chars`` are held and merged with the next sentence (no "Ok." blips),
      except the very first one when ``eager_first`` is set (it gets the voice started sooner).
    * A sentence longer than ``max_chars`` is split at the last comma-like mark so a run-on
      reply does not delay the first audio.
    """

    def __init__(self, min_chars: int = 12, max_chars: int = 160, eager_first: bool = True):
        self.min_chars, self.max_chars, self.eager_first = min_chars, max_chars, eager_first
        self._buf = ""
        self._emitted = 0

    def feed(self, delta: str) -> Iterator[str]:
        self._buf += delta
        while True:
            cut = self._find_cut()
            if cut is None:
                return
            piece, self._buf = self._buf[:cut], self._buf[cut:]
            piece = clean_for_speech(piece)
            if piece:
                self._emitted += 1
                yield piece

    def flush(self) -> Iterator[str]:
        piece = clean_for_speech(self._buf)
        self._buf = ""
        if piece:
            self._emitted += 1
            yield piece

    def _min(self) -> int:
        return 1 if (self.eager_first and self._emitted == 0) else self.min_chars

    def _find_cut(self) -> int | None:
        buf = self._buf
        start = 0
        for i, ch in enumerate(buf):
            end = None
            if ch == "\n":
                end = i + 1
            elif ch in _ENDERS and _is_boundary(buf, i):
                end = i + 1
                while end < len(buf) and buf[end] in "\"'”’)]":
                    end += 1
            if end is not None:
                if len(clean_for_speech(buf[:end])) >= self._min():
                    return end
                start = end
        if len(buf) > self.max_chars:
            soft = max(buf.rfind(c, start, self.max_chars) for c in _SOFT)
            if soft > start:
                return soft + 1
            space = buf.rfind(" ", start, self.max_chars)
            if space > start:
                return space + 1
        return None


def split_sentences(text: str, **kw) -> list[str]:
    s = SentenceStreamer(**kw)
    return list(s.feed(text)) + list(s.flush())


def stream_sentences(deltas: Iterable[str], **kw) -> Iterator[str]:
    s = SentenceStreamer(**kw)
    for d in deltas:
        yield from s.feed(d)
    yield from s.flush()


def devanagari_ratio(text: str) -> float:
    """Share of letters that are Devanagari (U+0900-U+097F)."""
    letters = [ch for ch in text if ch.isalpha() or "ऀ" <= ch <= "ॿ"]
    if not letters:
        return 0.0
    return sum("ऀ" <= ch <= "ॿ" for ch in letters) / len(letters)


def detect_language(text: str) -> str:
    """'hi' when the text is mostly Devanagari, else 'en'. Used to pick a TTS voice per sentence."""
    return "hi" if devanagari_ratio(text) >= 0.5 else "en"

