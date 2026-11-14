"""Spoken-number normalisation for recogniser output, before the LLM and the tools see it.

Callers read order ids and phone numbers digit by digit, and the recogniser writes them in many
ways: digit words in English or Hindi ("four eight", "चार आठ"), "double five" / "डबल पाँच",
Devanagari numerals, and digit groups split by spaces, commas or hyphens ("4 8 2 1 3",
"4881-3", "6, 8, 3, 4, 6"). ``normalize_numbers`` rewrites each such run as one plain digit
string ("48213"). Ordinary numbers are left alone ("1200 रुपये पर 15 प्रतिशत"), and digit words
outside a run stay words, because "एक", "दो" and "one" are also everyday words ("एक बार", "दो दिन").

A run is joined when it holds at least ``MIN_RUN_TOKENS`` digit tokens, or when separate digit
groups together make at least ``MIN_GROUP_DIGITS`` digits and each group is at most 4 digits long
(or two 5-digit halves of a 10-digit phone number).
"""

from __future__ import annotations

import os
import re

_DEV = str.maketrans("०१२३४५६७८९", "0123456789")
DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9",
    "शून्य": "0", "जीरो": "0", "ज़ीरो": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पाँच": "5", "पांच": "5",
    "छह": "6", "छः": "6", "छे": "6", "छै": "6", "सात": "7", "आठ": "8", "नौ": "9",
}
REPEAT_WORDS = {"double": 2, "triple": 3, "डबल": 2, "ट्रिपल": 3}
# Hint for the recogniser on the turn after the agent asked for a number (see stt_prompt).
DIGIT_PROMPTS = {
    "hi": "मेरा ऑर्डर नंबर 4 8 2 1 3 है। मेरा फ़ोन नंबर 9 8 7 6 5 4 3 2 1 0 है।",
    "en": "My order number is 4 8 2 1 3. My phone number is 9 8 7 6 5 4 3 2 1 0.",
}
_ASKS_NUMBER = re.compile(r"नंबर|अंक|आईडी|number|digit|\bid\b", re.I)


def digit_fix_enabled() -> bool:
    """``VOICE_AGENT_DIGIT_FIX=0`` turns off number normalisation, the recogniser hint and the read-back rule."""
    return os.environ.get("VOICE_AGENT_DIGIT_FIX", "1") != "0"


def stt_prompt(history: list[dict]) -> str | None:
    """A decoder prompt that primes Whisper for spoken digits, only when the agent's last reply asked
    for a number (or read one back and asked if it is right). Otherwise None: the default path."""
    if not digit_fix_enabled() or not history or history[-1]["role"] != "assistant":
        return None
    last = history[-1]["text"]
    if "?" not in last or not _ASKS_NUMBER.search(last):
        return None
    hindi = sum("ऀ" <= ch <= "ॿ" for ch in last) > len(last) / 4
    return DIGIT_PROMPTS["hi" if hindi else "en"]


MIN_RUN_TOKENS = 3
MIN_GROUP_DIGITS = 4
_PUNCT = ".?!।;:\"'“”‘’()"


def _digits_of(tok: str) -> str | None:
    """Digit string a single token stands for, or None."""
    core = tok.strip(_PUNCT).lower().translate(_DEV)
    if core.isdigit():
        return core
    return DIGIT_WORDS.get(core)


def normalize_numbers(text: str) -> str:
    toks = list(re.finditer(r"[^\s,\-–]+", text))
    out, pos, i = [], 0, 0
    while i < len(toks):
        run: list[str] = []
        repeat, k, end_tok = 0, i, None
        while k < len(toks):
            t = toks[k].group()
            core = t.strip(_PUNCT).lower()
            nxt = _digits_of(toks[k + 1].group()) if k + 1 < len(toks) else None
            if core in REPEAT_WORDS and nxt is not None and len(nxt) == 1 and t == t.strip(_PUNCT):
                repeat, k = REPEAT_WORDS[core], k + 1
                continue
            d = _digits_of(t)
            if d is None or (k > i and not re.fullmatch(r"[\s,\-–]*", text[toks[k - 1].end() : toks[k].start()])):
                break
            run.append(d * repeat if repeat else d)
            repeat, end_tok, k = 0, k, k + 1
            if t != t.rstrip(_PUNCT):  # sentence punctuation ends the run
                break
        joined = "".join(run)
        singles = sum(len(d) == 1 for d in run)
        is_run = len(run) >= 2 and (singles >= MIN_RUN_TOKENS or len(run) >= MIN_RUN_TOKENS
                                    or (len(joined) >= MIN_GROUP_DIGITS and max(map(len, run)) <= 4)
                                    or (len(joined) == 10 and max(map(len, run)) <= 5))  # phone read as 5 + 5
        if is_run:
            first, last = toks[i], toks[end_tok]
            lead = first.group()[: len(first.group()) - len(first.group().lstrip(_PUNCT))]
            trail = last.group()[len(last.group().rstrip(_PUNCT)) :]
            out.append(text[pos : first.start()] + lead + joined + trail)
            pos, i = last.end(), end_tok + 1
        else:
            i += 1
    out.append(text[pos:])
    return "".join(out)
