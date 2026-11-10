"""Streaming LLM clients with tool calling.

Three backends share one interface, ``client.turn(system, history, user_text, cancel)``, which
yields ``TextDelta`` and ``ToolEvent`` objects and handles tool round-trips internally:

* ``GeminiClient`` - Google Gemini over the REST ``streamGenerateContent`` SSE endpoint (free tier).
* ``OpenAICompatClient`` - any OpenAI-style ``/chat/completions`` endpoint (DeepSeek, a local vLLM, ...).
* ``OfflineClient`` - a deterministic rule-based stand-in so the app, tests and demos run with no key.

Network responses are cached on disk keyed by the request body (never the key), and 429 / 5xx
responses are retried with exponential back-off.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from . import tools as toolbox
from .config import CACHE_DIR

MAX_TOOL_ROUNDS = 3


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolEvent:
    name: str
    args: dict[str, Any]
    result: dict[str, Any]
    seconds: float


@dataclass
class TurnInfo:
    """Filled in by the client during a turn; read by the pipeline for metrics."""

    requests: int = 0
    cached: bool = False
    tool_calls: list[str] = field(default_factory=list)
    retries: int = 0
    retry_codes: list[int] = field(default_factory=list)


class LLMError(RuntimeError):
    pass


class RateLimited(LLMError):
    pass


# ------------------------------------------------------------------ HTTP + cache


class _SSE:
    """POST a JSON body and iterate the ``data:`` payloads of a server-sent-event stream, with a disk cache."""

    def __init__(self, cache: bool = True, cache_dir: Path = CACHE_DIR / "llm", max_retries: int = 5):
        import httpx

        self.httpx = httpx
        self.client = httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))
        self.cache = cache
        self.cache_dir = cache_dir
        self.max_retries = max_retries

    def _key(self, url: str, body: dict) -> str:
        blob = json.dumps({"url": url.split("?")[0], "body": body}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def stream(
        self, url: str, headers: dict, body: dict, info: TurnInfo, cancel: threading.Event | None
    ) -> Iterator[dict]:
        key = self._key(url, body)
        path = self.cache_dir / f"{key}.json"
        info.requests += 1
        if self.cache and path.exists():
            info.cached = True
            yield from json.loads(path.read_text(encoding="utf-8"))
            return
        recorded: list[dict] = []
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            with self.client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code in (429, 500, 502, 503, 504):
                    resp.read()
                    if attempt == self.max_retries:
                        cls = RateLimited if resp.status_code == 429 else LLMError
                        raise cls(f"HTTP {resp.status_code} after {attempt} retries")
                    retry_after = resp.headers.get("retry-after")
                    wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
                    info.retries += 1
                    info.retry_codes.append(resp.status_code)
                    time.sleep(min(wait, 30.0))
                    delay *= 2
                    continue
                if resp.status_code != 200:
                    text = resp.read().decode(errors="replace")[:300]
                    raise LLMError(f"HTTP {resp.status_code}: {text}")
                for line in resp.iter_lines():
                    if cancel is not None and cancel.is_set():
                        return  # don't cache a cancelled stream
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    obj = json.loads(data)
                    recorded.append(obj)
                    yield obj
                break
        if self.cache and recorded:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(recorded, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------ Gemini


class GeminiClient:
    name = "gemini"

    def __init__(self, model: str = "gemini-3.1-flash-lite", api_key: str | None = None, cache: bool = True):
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise LLMError("GOOGLE_API_KEY is not set")
        self.http = _SSE(cache=cache)
        self.info = TurnInfo()

    def _body(self, system: str, contents: list[dict]) -> dict:
        gen: dict[str, Any] = {"temperature": 0.4, "maxOutputTokens": 400}
        if self.model.startswith("gemini-3"):
            gen["thinkingConfig"] = {"thinkingLevel": "minimal"}
        elif self.model.startswith("gemini-2.5"):
            gen["thinkingConfig"] = {"thinkingBudget": 0}
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "tools": [{"functionDeclarations": toolbox.gemini_declarations()}],
            "generationConfig": gen,
        }

    def turn(
        self, system: str, history: list[dict], user_text: str, cancel: threading.Event | None = None
    ) -> Iterator[TextDelta | ToolEvent]:
        self.info = TurnInfo()
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["text"]}]} for m in history
        ]
        contents.append({"role": "user", "parts": [{"text": user_text}]})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent?alt=sse"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        for _ in range(MAX_TOOL_ROUNDS + 1):
            model_parts: list[dict] = []
            calls: list[dict] = []
            for chunk in self.http.stream(url, headers, self._body(system, contents), self.info, cancel):
                for cand in chunk.get("candidates", [])[:1]:
                    for part in cand.get("content", {}).get("parts", []):
                        model_parts.append(part)  # replayed verbatim: keeps thoughtSignature
                        if "functionCall" in part:
                            calls.append(part["functionCall"])
                        elif part.get("text") and not part.get("thought"):
                            yield TextDelta(part["text"])
            if cancel is not None and cancel.is_set():
                return
            if not calls:
                return
            contents.append({"role": "model", "parts": model_parts})
            responses = []
            for call in calls:
                ev = _run_tool(call.get("name", ""), call.get("args") or {})
                self.info.tool_calls.append(ev.name)
                yield ev
                fr: dict[str, Any] = {"name": ev.name, "response": ev.result}
                if call.get("id"):
                    fr["id"] = call["id"]
                responses.append({"functionResponse": fr})
            contents.append({"role": "user", "parts": responses})


# ------------------------------------------------------------------ OpenAI-compatible (DeepSeek, vLLM, ...)


class OpenAICompatClient:
    name = "openai"

    def __init__(self, model: str = "deepseek-chat", base_url: str | None = None, api_key: str | None = None, cache=True):
        self.model = model
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMError("DEEPSEEK_API_KEY / OPENAI_API_KEY is not set")
        self.http = _SSE(cache=cache)
        self.info = TurnInfo()

    def turn(self, system, history, user_text, cancel=None):
        self.info = TurnInfo()
        messages: list[dict] = [{"role": "system", "content": system}]
        messages += [{"role": m["role"], "content": m["text"]} for m in history]
        messages.append({"role": "user", "content": user_text})
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for _ in range(MAX_TOOL_ROUNDS + 1):
            body = {
                "model": self.model,
                "messages": messages,
                "tools": toolbox.openai_tools(),
                "stream": True,
                "temperature": 0.4,
                "max_tokens": 400,
            }
            text, calls = "", {}
            for chunk in self.http.stream(url, headers, body, self.info, cancel):
                for choice in chunk.get("choices", [])[:1]:
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        text += delta["content"]
                        yield TextDelta(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
            if (cancel is not None and cancel.is_set()) or not calls:
                return
            ordered = [calls[i] for i in sorted(calls)]
            messages.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                        for c in ordered
                    ],
                }
            )
            for c in ordered:
                try:
                    args = json.loads(c["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                ev = _run_tool(c["name"], args)
                self.info.tool_calls.append(ev.name)
                yield ev
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(ev.result, ensure_ascii=False)})


# ------------------------------------------------------------------ offline


_MATH_RE = re.compile(r"[\d०-९][\d०-९\s.,]*\s*(?:[-+*/×÷^%x]|plus|minus|times|into|गुणा|भाग|जोड़|घटा)")
_TIME_WORDS = ("time", "date", "day", "today", "समय", "टाइम", "तारीख", "दिन", "आज", "बजे", "वार")


class OfflineClient:
    """No network: routes to a tool by keyword and reads the result back. For tests and key-less demos."""

    name = "offline"
    model = "rules"

    def __init__(self, **_):
        self.info = TurnInfo()

    def turn(self, system, history, user_text, cancel=None):
        self.info = TurnInfo()
        hindi = any("ऀ" <= ch <= "ॿ" for ch in user_text)
        low = user_text.lower()
        if _MATH_RE.search(low):
            expr = _words_to_ops(low)
            ev = _run_tool("calculator", {"expression": expr})
            self.info.tool_calls.append(ev.name)
            yield ev
            r = ev.result.get("result")
            reply = (f"इसका जवाब {r} है।" if hindi else f"The answer is {r}.") if r is not None else (
                "माफ़ कीजिए, मैं यह हिसाब नहीं कर पाया।" if hindi else "Sorry, I could not work that out."
            )
        elif any(w in low for w in _TIME_WORDS):
            ev = _run_tool("get_datetime", {})
            self.info.tool_calls.append(ev.name)
            yield ev
            d = ev.result
            reply = (
                f"अभी {d['time_24h']} बजे हैं, आज {d['weekday_hi']} है।"
                if hindi
                else f"It is {d['time_24h']} on {d['weekday_en']}, {d['date']}."
            )
        else:
            ev = _run_tool("search_knowledge", {"query": user_text})
            self.info.tool_calls.append(ev.name)
            yield ev
            hits = ev.result.get("results") or []
            reply = hits[0]["text"] if hits else (
                "माफ़ कीजिए, अभी मैं ऑफ़लाइन हूँ और इसका जवाब नहीं दे सकता।"
                if hindi
                else "Sorry, I am offline right now and cannot answer that."
            )
        for word in re.findall(r"\S+\s*", reply):
            if cancel is not None and cancel.is_set():
                return
            yield TextDelta(word)


def _words_to_ops(text: str) -> str:
    for word, op in (("गुणा", "*"), ("times", "*"), ("into", "*"), ("भाग", "/"), ("plus", "+"), ("जोड़", "+"),
                     ("minus", "-"), ("घटा", "-"), (" x ", " * ")):
        text = text.replace(word, op)
    m = re.findall(r"[\d०-९.,%()+\-*/×÷^ ]+", text)
    return max(m, key=lambda s: len(s.strip())).strip() if m else text


def _run_tool(name: str, args: dict) -> ToolEvent:
    t0 = time.perf_counter()
    result = toolbox.call_tool(name, args)
    return ToolEvent(name=name, args=args, result=result, seconds=time.perf_counter() - t0)


def make_client(provider: str = "auto", model: str | None = None, cache: bool = True):
    """'auto' picks Gemini when GOOGLE_API_KEY is set, else offline. DeepSeek is only used when asked for."""
    if provider == "auto":
        provider = "gemini" if (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")) else "offline"
    if provider == "gemini":
        return GeminiClient(model=model or "gemini-3.1-flash-lite", cache=cache)
    if provider in ("openai", "deepseek"):
        return OpenAICompatClient(model=model or "deepseek-chat", cache=cache)
    if provider == "offline":
        return OfflineClient()
    raise ValueError(f"unknown LLM provider {provider!r}")
