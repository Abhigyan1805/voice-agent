from datetime import datetime, timezone

import pytest

from voice_agent.tools import TOOLS, KnowledgeBase, calculator, call_tool, gemini_declarations, get_datetime, openai_tools


@pytest.mark.parametrize(
    "expr,result",
    [
        ("23*47", 1081),
        ("23 × 47", 1081),
        ("100/8", 12.5),
        ("2^10", 1024),
        ("18% of 1250", 225),
        ("sqrt(144) + 1", 13),
        ("२३ * ४७", 1081),
        ("1,000 + 1", 1001),
        ("-(3 - 5)", 2),
    ],
)
def test_calculator(expr, result):
    assert calculator(expr)["result"] == result


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('x')", "a + 1", "9**9**9", "[1,2]"])
def test_calculator_rejects_unsafe_or_bad_input(expr):
    out = calculator(expr)
    assert "error" in out and "result" not in out


def test_calculator_division_by_zero():
    assert calculator("1/0")["error"] == "division by zero"


def test_get_datetime_fixed_clock():
    now = datetime(2026, 11, 10, 3, 30, tzinfo=timezone.utc)
    out = get_datetime("Asia/Kolkata", now=now)
    assert out["time_24h"] == "09:00" and out["weekday_en"] == "Tuesday" and out["weekday_hi"] == "मंगलवार"
    assert out["month_hi"] == "नवंबर"
    assert "error" in get_datetime("Mars/Olympus")


def test_knowledge_base_hindi_english_and_hinglish_queries():
    kb = KnowledgeBase.load()
    assert kb.search("cyber fraud helpline")[0]["title"] == "Cyber crime helpline"
    assert kb.search("साइबर ठगी हो गई")[0]["title"] == "Cyber crime helpline"
    assert kb.search("एम्बुलेंस का नंबर")[0]["title"] == "Police, fire and ambulance numbers"
    assert kb.search("तुम क्या कर सकते हो")[0]["title"] == "What this assistant can do"
    assert kb.search("zzqx") == []


def test_call_tool_dispatch_and_errors():
    assert call_tool("calculator", {"expression": "2+2"})["result"] == 4
    assert "error" in call_tool("nope", {})
    assert "error" in call_tool("calculator", {"wrong": 1})
    assert call_tool("search_knowledge", {"query": "112"})["results"]


def test_declarations_cover_every_tool():
    names = set(TOOLS)
    assert {d["name"] for d in gemini_declarations()} == names
    assert {t["function"]["name"] for t in openai_tools()} == names
