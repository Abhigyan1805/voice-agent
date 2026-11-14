"""Call-quality suite: run scripted multi-turn calls through the agent and score every turn.

A call (``evals/calls/calls.json``) is a list of user turns, each with a reference text, the voice
that recorded it, and expectations:

  tools      list of {name, args}: each must be called with those arguments (digits normalised);
             [] means "no tool at all"
  forbid     tool names that must not be called in this turn
  reply_any  regexes, at least one must match the reply (after digit/script normalisation)
  reply_none regexes, none may match (an invented status, a wrong number)
  reply_lang "hi" or "en", from the reply's script

Call level: ``final_state`` (order id -> expected delivery date on the service after the call),
``reply_any_in_call`` (regexes that some reply in the call must match, for facts the agent may give
in any turn) and ``faults`` injected into the order service for the whole call. ``final_state_ok``
in the results means every call-level check passed.

Two tracks run the same call:
  spoken  each turn's audio -> VAD -> STT -> LLM + tools -> TTS, history kept across turns
  oracle  the reference text goes straight to the LLM (no audio), same history rules

Failure attribution, applied to the first failing turn of a spoken call (later failing turns are
``cascade``), first rule that matches wins:
  tool  the turn's tool calls were right, but the service or client failed in a way the script did
        not plan for (timeout or unavailable with no injected fault), or the client broke its time
        budget. (not_found and rejected are answers about the data, not tool failures.)
  vad   the VAD found no speech in the turn, or split it into several turns
  stt   the turn failed in the spoken track and passed in the oracle track
  llm   the turn failed in the oracle track too
  tts   the reply text passed every check, but a required number or date was lost when the spoken
        reply was transcribed back by the agent's own recogniser ("TTS suspect": recogniser errors
        are in that loop too)
A call whose turns all pass but whose final state is wrong is attributed to ``llm`` if the oracle
call ended in the wrong state too, otherwise to ``stt``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np

_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_DIGIT_WORDS = {
    "शून्य": "0", "जीरो": "0", "ज़ीरो": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पाँच": "5", "पांच": "5",
    "छह": "6", "छः": "6", "छे": "6", "सात": "7", "आठ": "8", "नौ": "9", "zero": "0", "oh": "0", "one": "1", "two": "2",
    "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_PLACEHOLDER = re.compile(r"\{(d|iso)([+-]\d+)\}")


# ------------------------------------------------------------------ normalisation and checks


def expand(pattern: str, today: date, words: dict[str, str]) -> str:
    """Fill {d+N} / {iso+N} from the run date and @name from the status-word table."""

    def ph(m):
        day = today + timedelta(days=int(m.group(2)))
        return str(day.day) if m.group(1) == "d" else day.isoformat()

    pattern = _PLACEHOLDER.sub(ph, pattern)
    return re.sub(r"@(\w+)", lambda m: f"(?:{words[m.group(1)]})", pattern)


def norm_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_DEV_DIGITS).lower()
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)


def digits_only(value: Any) -> str:
    return re.sub(r"\D", "", str(value).translate(_DEV_DIGITS))


def spoken_digits(text: str) -> str:
    """Digits a transcript contains, counting digit words ("चार आठ", "four eight") as digits."""
    out = []
    for tok in re.split(r"[\s,.;:!?।()\-]+", norm_text(text)):
        out.append(_DIGIT_WORDS.get(tok, digits_only(tok)))
    return "".join(out)


def reply_language(text: str) -> str:
    # by words, so a Hindi reply that keeps English nouns ("आपका order shipped है") still counts as Hindi
    words = text.split()
    dev = sum(1 for w in words if any("ऀ" <= ch <= "ॿ" for ch in w))
    lat = sum(1 for w in words if any(ch.isascii() and ch.isalpha() for ch in w) and not any("ऀ" <= ch <= "ॿ" for ch in w))
    return "hi" if dev >= lat and dev > 0 else "en"


def _arg_equal(key: str, want: Any, got: Any) -> bool:
    if key in ("order_id", "phone"):
        return digits_only(want) == digits_only(got)
    return str(want).strip().lower() == str(got).strip().lower()


def check_turn(expect: dict, tools: list[dict], reply: str, today: date, words: dict[str, str]) -> dict[str, bool]:
    """Every check that the turn's expectations define, as name -> passed."""
    checks: dict[str, bool] = {}
    names = [t["name"] for t in tools]
    if "tools" in expect:
        if expect["tools"] == []:
            checks["tools"] = not tools
        else:
            checks["tools"] = all(
                any(t["name"] == e["name"] and all(_arg_equal(k, v, t["args"].get(k)) for k, v in e.get("args", {}).items())
                    for t in tools)
                for e in expect["tools"]
            )
    if expect.get("forbid"):
        checks["forbid"] = not any(n in names for n in expect["forbid"])
    text = norm_text(reply)
    if expect.get("reply_any"):
        checks["reply_any"] = any(re.search(expand(p, today, words), text) for p in expect["reply_any"])
    if expect.get("reply_none"):
        checks["reply_none"] = not any(re.search(expand(p, today, words), text) for p in expect["reply_none"])
    if expect.get("reply_lang"):
        checks["reply_lang"] = reply_language(reply) == expect["reply_lang"]
    return checks


def call_checks(call: dict, result: dict, today: date, words: dict[str, str]) -> dict[str, bool]:
    checks = {}
    if call.get("final_state"):
        want = {oid: expand(v, today, {}) for oid, v in call["final_state"].items()}
        checks["final_state"] = result["final_state"] == want
    if call.get("reply_any_in_call"):
        replies = " ".join(norm_text(t["reply"]) for t in result["turns"])
        checks["reply_any_in_call"] = any(re.search(expand(p, today, words), replies) for p in call["reply_any_in_call"])
    return checks


def rescore(calls: list[dict], spec: dict, today: date) -> None:
    """Recompute every check and attribution of stored call results against the current call specs
    (used when a check is corrected, so all prompt versions are scored by the same rules). Stored
    transcripts, replies, tool calls, final service state and TTS round-trip results are reused."""
    by_id = {c["id"]: c for c in spec["calls"]}
    words = spec["status_words"]
    for res in calls:
        call = by_id[res["id"]]
        for t, turn_spec in zip(res["turns"], call["turns"]):
            t["checks"] = check_turn(turn_spec["expect"], t["tools"], t["reply"], today, words)
            t["text_passed"] = all(t["checks"].values())
            if res["track"] == "spoken":
                pats = numeric_patterns(turn_spec["expect"])
                if not (t["text_passed"] and pats):
                    t["tts_roundtrip_ok"] = None
                elif t.get("tts_roundtrip_text") is not None:
                    normed = norm_text(t["tts_roundtrip_text"])
                    t["tts_roundtrip_ok"] = any(re.search(expand(p, today, words), normed) for p in pats)
                t["passed"] = t["text_passed"] and t.get("tts_roundtrip_ok") is not False
            else:
                t["passed"] = t["text_passed"]
        res["call_checks"] = call_checks(call, res, today, words)
        res["final_state_ok"] = all(res["call_checks"].values())
    oracle = {(c["id"], c["rep"]): c for c in calls if c["track"] == "oracle"}
    for res in calls:
        if res["track"] == "spoken":
            res["attribution"] = attribute(res, oracle.get((res["id"], res["rep"])))


def numeric_patterns(expect: dict) -> list[str]:
    """reply_any patterns that require a number or date: the entities checked in the TTS round trip."""
    return [p for p in expect.get("reply_any", []) if re.search(r"\\d|\{d|\d", p)]


def _numbers_as_digits(text: str) -> str:
    """Score "चार आठ दो", "4 8 2" and "482" alike: digit words -> digits, digits spaced one per token."""
    from .fleurs import normalize

    toks = [_DIGIT_WORDS.get(t, t) for t in normalize(text.translate(_DEV_DIGITS)).split()]
    return " ".join(" ".join(t) if t.isdigit() else t for t in toks)


def wer_cer(ref: str, hyp: str) -> tuple[float | None, float | None]:
    """Per-turn WER/CER with the repo normaliser, after writing every digit as its own token."""
    import jiwer

    r, h = _numbers_as_digits(ref), _numbers_as_digits(hyp)
    if not r:
        return None, None
    return round(100 * jiwer.wer(r, h), 2), round(100 * jiwer.cer(r, h), 2)


# ------------------------------------------------------------------ attribution


def attribute(spoken: dict, oracle: dict | None) -> dict:
    """Assign a stage to the first failing turn of a spoken call; see the module docstring."""
    turns = spoken["turns"]
    first = next((i for i, t in enumerate(turns) if not t["passed"]), None)
    for i, t in enumerate(turns):
        t["attribution"] = None if t["passed"] else ("cascade" if i != first else None)
    if first is None:
        if spoken["final_state_ok"]:
            return {"stage": None, "turn": None}
        stage = "llm" if oracle is not None and not oracle["final_state_ok"] else "stt"
        return {"stage": stage, "turn": "final_state"}
    t = turns[first]
    o = oracle["turns"][first] if oracle is not None and first < len(oracle["turns"]) else None
    if t["checks"].get("tools", True) and (t["unplanned_tool_error"] or t["tool_over_budget"]):
        stage = "tool"
    elif t["vad_no_speech"] or t["vad_split"]:
        stage = "vad"
    elif t["text_passed"] and t.get("tts_roundtrip_ok") is False:
        stage = "tts"
    elif o is not None and o["passed"]:
        stage = "stt"
    else:
        stage = "llm"
    t["attribution"] = stage
    return {"stage": stage, "turn": first}


# ------------------------------------------------------------------ runner


class CallRunner:
    """Runs calls against a pipeline whose agent has the order tools, and a running order service."""

    def __init__(self, pipe, service, scorer=None, condition: Callable[[np.ndarray], np.ndarray] | None = None,
                 audio_dir: Path | None = None, today: date | None = None, words: dict[str, str] | None = None):
        self.pipe, self.service, self.scorer = pipe, service, scorer
        self.condition = condition
        self.audio_dir = audio_dir
        self.today = today
        self.words = words or {}

    def _reset_service(self, faults: dict) -> None:
        self.service.store.reset()
        if faults:
            self.service.store.faults.update(faults)

    def _segment(self, audio: np.ndarray):
        from .vad import SpeechEnd, segment

        ends = [e for e in segment(audio, self.scorer) if isinstance(e, SpeechEnd)]
        if not ends:
            return None, True, False
        utt = np.concatenate([e.audio for e in ends]) if len(ends) > 1 else ends[0].audio
        return utt, False, len(ends) > 1

    def run_call(self, call: dict, track: str) -> dict:
        from .audio import load_audio
        from .pipeline import SpeechAudio, ToolUse, TurnDone

        self._reset_service(call.get("faults") or {})
        self.pipe.agent.reset()
        planned_error = bool(call.get("faults"))
        turns = []
        for k, spec in enumerate(call["turns"]):
            row: dict[str, Any] = {"turn": k, "reference": spec["user"], "vad_no_speech": False, "vad_split": False}
            if track == "spoken":
                audio = load_audio(self.audio_dir / f"{call['id']}-{k}.wav")
                if self.condition is not None:
                    audio = self.condition(audio)
                utt, row["vad_no_speech"], row["vad_split"] = self._segment(audio)
                events = self.pipe.run_turn(audio=utt) if utt is not None else iter(())
            else:
                events = self.pipe.run_turn(text=spec["user"])
            tools, speech, metrics = [], [], None
            for ev in events:
                if isinstance(ev, ToolUse):
                    tools.append({"name": ev.name, "args": ev.args, "result": ev.result, "meta": getattr(ev, "meta", None)})
                elif isinstance(ev, SpeechAudio) and ev.index >= 0:
                    speech.append(ev)
                elif isinstance(ev, TurnDone):
                    metrics = ev.metrics
            reply = metrics.reply if metrics else ""
            transcript = metrics.transcript if metrics else ""
            checks = check_turn(spec["expect"], tools, reply, self.today, self.words)
            row.update(
                transcript=transcript, reply=reply, tools=tools, checks=checks,
                text_passed=all(checks.values()),
                unplanned_tool_error=any(t["result"].get("error") in ("timeout", "unavailable")
                                         for t in tools if t["name"] in _ORDER_TOOLS) and not planned_error,
                tool_over_budget=any(t["meta"] and not t["meta"].get("within_budget", True) for t in tools),
                stages={k2: getattr(metrics, k2, None) for k2 in ("stt_s", "llm_first_token_s", "tts_first_audio_s",
                                                                   "time_to_first_audio_s", "total_s")} if metrics else {},
                llm_tokens={"prompt": getattr(metrics, "llm_prompt_tokens", 0),
                            "completion": getattr(metrics, "llm_completion_tokens", 0)},
                error=getattr(metrics, "error", None),
            )
            if track == "spoken":
                row["wer"], row["cer"] = wer_cer(spec["user"], transcript)
                want = [digits_only(v) for e in spec["expect"].get("tools", []) for kk, v in e.get("args", {}).items()
                        if kk in ("order_id", "phone")]
                row["entities_heard"] = all(w in spoken_digits(transcript) for w in want) if want else None
                pats = numeric_patterns(spec["expect"])
                row["tts_roundtrip_ok"] = None
                if row["text_passed"] and pats and speech:
                    row["tts_roundtrip_ok"], row["tts_roundtrip_text"] = self._roundtrip(speech, pats)
                row["passed"] = row["text_passed"] and row["tts_roundtrip_ok"] is not False
            else:
                row["passed"] = row["text_passed"]
            turns.append(row)
        final = {oid: self.service.store.orders[oid]["expected_delivery"] for oid in (call.get("final_state") or {})}
        out = {"id": call["id"], "lang": call["lang"], "held_out": call.get("held_out", False), "track": track,
               "turns": turns, "final_state": final, "writes": self.service.store.writes}
        out["call_checks"] = call_checks(call, out, self.today, self.words)
        out["final_state_ok"] = all(out["call_checks"].values())
        return out

    def _roundtrip(self, speech, patterns) -> tuple[bool, str]:
        from .audio import resample

        audio = np.concatenate([resample(s.audio, s.sample_rate, 16_000) for s in speech])
        text = self.pipe.stt.transcribe(audio)
        normed = norm_text(text)
        ok = any(re.search(expand(p, self.today, self.words), normed) for p in patterns)
        return ok, text


_ORDER_TOOLS = ("get_order_status", "find_orders_by_phone", "reschedule_delivery")


def summarise(calls: list[dict]) -> dict:
    """Counts with denominators: task success, turn checks, WER, failures by stage."""
    spoken = [c for c in calls if c["track"] == "spoken"]
    oracle = [c for c in calls if c["track"] == "oracle"]

    def success(c):
        return all(t["passed"] for t in c["turns"]) and c["final_state_ok"]

    def tool_turns(cs):
        ts = [t for c in cs for t in c["turns"] if "tools" in t["checks"]]
        return f"{sum(t['checks']['tools'] for t in ts)}/{len(ts)}"

    stages: dict[str, int] = {}
    for c in spoken:
        s = c["attribution"]["stage"]
        if s:
            stages[s] = stages.get(s, 0) + 1
    wers = [t["wer"] for c in spoken for t in c["turns"] if t.get("wer") is not None]
    ent = [t["entities_heard"] for c in spoken for t in c["turns"] if t.get("entities_heard") is not None]
    by_lang: dict[str, str] = {}
    for lang in sorted({c["lang"] for c in spoken}):
        xs = [c for c in spoken if c["lang"] == lang]
        by_lang[lang] = f"{sum(map(success, xs))}/{len(xs)}"
    held = [c for c in spoken if c["held_out"]]
    tuned = [c for c in spoken if not c["held_out"]]
    return {
        "calls_passed_spoken": f"{sum(map(success, spoken))}/{len(spoken)}",
        "calls_passed_oracle": f"{sum(map(success, oracle))}/{len(oracle)}",
        "calls_passed_spoken_tuning_set": f"{sum(map(success, tuned))}/{len(tuned)}",
        "calls_passed_spoken_held_out": f"{sum(map(success, held))}/{len(held)}",
        "turns_passed_spoken": f"{sum(t['passed'] for c in spoken for t in c['turns'])}/{sum(len(c['turns']) for c in spoken)}",
        "turns_passed_oracle": f"{sum(t['passed'] for c in oracle for t in c['turns'])}/{sum(len(c['turns']) for c in oracle)}",
        "tool_correct_spoken": tool_turns(spoken),
        "tool_correct_oracle": tool_turns(oracle),
        "by_language_spoken": by_lang,
        "by_call_spoken": {cid: f"{sum(success(c) for c in spoken if c['id'] == cid)}/{sum(c['id'] == cid for c in spoken)}"
                           for cid in sorted({c["id"] for c in spoken})},
        "failures_by_stage": dict(sorted(stages.items())),
        "turn_wer": {"n": len(wers), "median": float(np.median(wers)) if wers else None,
                     "mean": round(float(np.mean(wers)), 2) if wers else None,
                     "zero_error_turns": sum(w == 0 for w in wers)},
        "order_and_phone_digits_heard": f"{sum(ent)}/{len(ent)}",
        "tokens": {k: sum(t["llm_tokens"][k] for c in calls for t in c["turns"]) for k in ("prompt", "completion")},
    }


def load_calls(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
