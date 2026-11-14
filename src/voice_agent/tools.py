"""The agent's tools: a safe calculator, the current date/time, and a tiny local knowledge base.

Each tool is a plain function plus a JSON-schema declaration, so the same registry can be handed
to Gemini (``functionDeclarations``) or to an OpenAI-compatible API (``tools``).
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# ---------------------------------------------------------------- calculator

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTS = {"pi": math.pi, "e": math.e}
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


class CalcError(ValueError):
    pass


def _normalise_expr(expr: str) -> str:
    expr = expr.translate(_DEVANAGARI_DIGITS)
    expr = expr.replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)  # 1,000 -> 1000
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"\1/100*", expr, flags=re.I)  # 15% of 200
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
    return expr.strip()


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
            raise CalcError("exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise CalcError(f"unsupported expression element: {type(node).__name__}")


def calculator(expression: str) -> dict[str, Any]:
    """Evaluate an arithmetic expression without ``eval``: only numbers, operators and a few math functions."""
    expr = _normalise_expr(str(expression))
    if len(expr) > 200:
        return {"error": "expression too long"}
    try:
        value = _eval(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        return {"expression": expr, "error": "division by zero"}
    except (CalcError, SyntaxError, TypeError, ValueError, OverflowError) as e:
        return {"expression": expr, "error": str(e) or type(e).__name__}
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            value = int(value)
        else:
            value = round(value, 6)
    return {"expression": expr, "result": value}


# ---------------------------------------------------------------- date / time

_HI_DAYS = ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"]
_HI_MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]


def get_datetime(timezone: str = "Asia/Kolkata", now: datetime | None = None) -> dict[str, Any]:
    """Current local date and time, with Hindi day and month names."""
    try:
        tz = ZoneInfo(timezone or "Asia/Kolkata")
    except (ZoneInfoNotFoundError, ValueError):
        return {"error": f"unknown timezone {timezone!r}"}
    now = (now or datetime.now(tz)).astimezone(tz)
    return {
        "timezone": str(tz),
        "iso": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time_24h": now.strftime("%H:%M"),
        "weekday_en": now.strftime("%A"),
        "weekday_hi": _HI_DAYS[now.weekday()],
        "month_hi": _HI_MONTHS[now.month - 1],
    }


# ---------------------------------------------------------------- knowledge base

KB_PATH = Path(__file__).parent / "data" / "knowledge.json"


def _trigrams(text: str) -> set[str]:
    text = re.sub(r"[^\wऀ-ॿ]+", " ", text.lower()).strip()
    grams: set[str] = set()
    for word in text.split():
        w = f" {word} "
        grams.update(w[i : i + 3] for i in range(len(w) - 2))
    return grams


@dataclass
class KnowledgeBase:
    entries: list[dict[str, Any]]

    @classmethod
    def load(cls, path: Path = KB_PATH) -> "KnowledgeBase":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def search(self, query: str, k: int = 2, min_score: float = 0.08) -> list[dict[str, Any]]:
        """Character-trigram overlap: works for Hindi, English and mixed queries without an embedding model."""
        q = _trigrams(query)
        if not q:
            return []
        scored = []
        for e in self.entries:
            doc = _trigrams(" ".join([e["title"], " ".join(e.get("keywords", [])), e["text"]]))
            key = _trigrams(" ".join([e["title"], " ".join(e.get("keywords", []))]))
            score = len(q & doc) / len(q) * 0.5 + len(q & key) / len(q)
            if score >= min_score:
                scored.append((score, e))
        scored.sort(key=lambda x: -x[0])
        return [{"title": e["title"], "text": e["text"], "score": round(s, 3)} for s, e in scored[:k]]


_KB: KnowledgeBase | None = None


def search_knowledge(query: str) -> dict[str, Any]:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase.load()
    hits = _KB.search(query)
    return {"query": query, "results": hits} if hits else {"query": query, "results": [], "note": "no match"}


# ---------------------------------------------------------------- registry


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., dict[str, Any]]


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "calculator",
            "Evaluate an arithmetic expression exactly. Call it for every calculation, even simple ones "
            "(sums, percentages, splitting a bill, square roots), instead of computing in your head. Write the expression with digits and + - * / ** ( ) "
            "and functions sqrt, round, log, sin, cos; '15% of 200' is accepted.",
            {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "e.g. '23*47' or '18% of 1250'"}},
                "required": ["expression"],
            },
            calculator,
        ),
        Tool(
            "get_datetime",
            "Get the current date, time and weekday. Call it whenever the user asks the current time, "
            "today's date or the day of the week; you do not know them otherwise.",
            {
                "type": "object",
                "properties": {"timezone": {"type": "string", "description": "IANA timezone, default Asia/Kolkata"}},
            },
            get_datetime,
        ),
        Tool(
            "search_knowledge",
            "Search a small local knowledge base about this voice assistant itself (what it can do, how it "
            "works, which models it uses) and about India's emergency and helpline numbers (police, ambulance, "
            "fire, women's helpline...). Always call it for those topics instead of answering from memory. "
            "Do not call it for general knowledge or unclear input.",
            {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "the user's question or key words"}},
                "required": ["query"],
            },
            search_knowledge,
        ),
    ]
}


def call_tool(name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """Run a tool by name. Errors are returned as data so the LLM can recover; they never raise."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}"}
    try:
        return tool.fn(**(args or {}))
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}


def gemini_declarations() -> list[dict[str, Any]]:
    return [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in TOOLS.values()]


def openai_tools() -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in TOOLS.values()
    ]
