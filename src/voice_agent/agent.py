"""The conversational agent: system prompt, short rolling history, and the LLM turn."""

from __future__ import annotations

import threading
from typing import Iterator

from .llm import TextDelta, ToolEvent

SYSTEM_PROMPT = """You are a friendly voice assistant. Your words are spoken aloud by a text-to-speech voice.

Input: the user's words come from a speech recogniser and may contain recognition errors. The user may speak Hindi, English, or code-mixed Hinglish; Hindi is usually transcribed in Devanagari, and English words inside Hindi speech may appear in Devanagari (for example "मेरा फ़ोन") or in Latin script. Guess the intended meaning when a word looks misrecognised. Hindi written in Latin script ("kya haal hai") is Hindi, not English.

Language of your reply:
- If the user spoke Hindi or Hinglish, reply in simple, conversational Hindi written in Devanagari script. Common English words that Hindi speakers use (phone, computer, email) may stay in English, as people say them.
- If the user spoke English, reply in English.

Style: this is speech, not text. Answer in one to three short sentences; put the direct answer first. No markdown, lists, emoji, URLs or symbols. Write numbers as digits.

Tools (call the tool first, then answer from its result):
- calculator: for every calculation, even an easy one (sums, percentages, splitting a bill, square roots). Never do maths in your head.
- get_datetime: whenever the user asks the current time, today's date or the day of the week. You do not know the date otherwise.
- search_knowledge: for any question about an emergency or helpline number in India (police, ambulance, fire, women's helpline and so on) and for any question about yourself (who you are, what you can do, how you work, which models you use). Never answer these from memory, even if you think you know the number; read it from the tool result.
- No tool for general knowledge, small talk, opinions, or when the input is unclear or garbled: then answer directly, or briefly ask the user to repeat.
If you do not know something, say so briefly."""


SYSTEM_PROMPT_V1 = SYSTEM_PROMPT[: SYSTEM_PROMPT.index("Tools (call")] + (
    "Tools: use calculator for any arithmetic (never do maths in your head), get_datetime for the current time, "
    "date or day, and search_knowledge for questions about yourself or about Indian emergency helpline numbers. "
    "If you do not know something, say so briefly."
)


class Agent:
    def __init__(self, client, system: str = SYSTEM_PROMPT, max_history_turns: int = 6):
        self.client = client
        self.system = system
        self.max_history = max_history_turns * 2
        self.history: list[dict] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.history.clear()

    def respond(self, user_text: str, cancel: threading.Event | None = None) -> Iterator[TextDelta | ToolEvent]:
        """Stream one reply. The exchange is added to history even if the user barges in, with the
        reply truncated to what was generated, so the next turn knows where the agent was cut off."""
        with self._lock:
            history = list(self.history[-self.max_history :])
        reply = []
        try:
            for ev in self.client.turn(self.system, history, user_text, cancel):
                if isinstance(ev, TextDelta):
                    reply.append(ev.text)
                yield ev
        finally:
            text = "".join(reply).strip()
            interrupted = cancel is not None and cancel.is_set()
            with self._lock:
                self.history.append({"role": "user", "text": user_text})
                self.history.append({"role": "assistant", "text": (text + " [interrupted]") if interrupted else text or "…"})
