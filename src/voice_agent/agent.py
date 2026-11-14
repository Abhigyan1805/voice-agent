"""The conversational agent: system prompt, short rolling history, and the LLM turn."""

from __future__ import annotations

import threading
from typing import Iterator

from .llm import TextDelta, ToolEvent, ToolStart

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


ORDER_RULES = """

Order desk: you can look up and reschedule the caller's orders with get_order_status, find_orders_by_phone and reschedule_delivery. Today is {weekday}, {today}.
- Order ids have 5 digits and phone numbers 10 digits. Callers say digits as words in Hindi or English ("चार आठ दो एक तीन" is 48213, "छह शून्य" is 60), and the recogniser may write them as words, digits or Devanagari digits. Convert them to plain digits before calling a tool.
- If the caller does not know the order id, ask for their 10-digit phone number and call find_orders_by_phone.
- When a caller asks to move a delivery, first call get_order_status, so you can tell them at once if the order is already delivered or cancelled. Never call reschedule_delivery in the same turn as the request, even when the caller has already given the date: read the order id back digit by digit, say the new date, and ask them to confirm. A date said together with the request is not a confirmation. Call reschedule_delivery only in a later turn, after a clear yes to your read-back, and then tell them the result the tool returned. If they say no, change nothing. Dates go to the tool as YYYY-MM-DD and must be after today.
- If a tool returns not_found, ask the caller to repeat the number once; if it is still not found, offer to look up their orders by phone number.
- If a tool returns timeout or unavailable, say the order system is slow right now and offer to try again. Never guess or invent an order's status or delivery date.
- If a tool returns rejected, explain the reason simply (for example, the order is already delivered).
- Tell the caller the status and the delivery date in words they understand; for a date say the day and month.
- After every tool result, always say something to the caller; never end a turn silently.
- Reply in the language of the caller's latest turn: English if they just spoke English, Hindi if they just spoke Hindi or Hinglish, even if earlier turns were in the other language. A short English reply such as "yes", "okay" or "thank you" is English: answer it in English."""


def build_system_prompt() -> str:
    """The system prompt for this turn: the order-desk rules (with today's date) are added only when
    the order tools are enabled, so the general assistant's prompt stays unchanged."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from .tools import orders_enabled

    if not orders_enabled():
        return SYSTEM_PROMPT
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    return SYSTEM_PROMPT + ORDER_RULES.format(weekday=now.strftime("%A"), today=now.strftime("%Y-%m-%d"))


class Agent:
    def __init__(self, client, system: str | None = None, max_history_turns: int = 6):
        self.client = client
        self.system = system
        self.max_history = max_history_turns * 2
        self.history: list[dict] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.history.clear()

    def respond(self, user_text: str, cancel: threading.Event | None = None) -> Iterator[TextDelta | ToolEvent | ToolStart]:
        """Stream one reply. The exchange is added to history even if the user barges in, with the
        reply truncated to what was generated, so the next turn knows where the agent was cut off."""
        with self._lock:
            history = list(self.history[-self.max_history :])
        reply = []
        try:
            system = self.system if self.system is not None else build_system_prompt()
            for ev in self.client.turn(system, history, user_text, cancel):
                if isinstance(ev, TextDelta):
                    reply.append(ev.text)
                yield ev
        finally:
            text = "".join(reply).strip()
            interrupted = cancel is not None and cancel.is_set()
            with self._lock:
                self.history.append({"role": "user", "text": user_text})
                self.history.append({"role": "assistant", "text": (text + " [interrupted]") if interrupted else text or "…"})
