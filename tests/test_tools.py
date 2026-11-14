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


def test_declarations_cover_every_tool(monkeypatch):
    monkeypatch.delenv("VOICE_AGENT_ORDERS_URL", raising=False)
    base = {n for n, t in TOOLS.items() if not t.orders}
    assert base == {"calculator", "get_datetime", "search_knowledge"}
    assert {d["name"] for d in gemini_declarations()} == base
    assert {t["function"]["name"] for t in openai_tools()} == base
    monkeypatch.setenv("VOICE_AGENT_ORDERS_URL", "http://127.0.0.1:1")
    assert {d["name"] for d in gemini_declarations()} == set(TOOLS)
    assert {t["function"]["name"] for t in openai_tools()} == set(TOOLS)


def test_order_tools_disabled_without_service(monkeypatch):
    from voice_agent.tools import call_tool

    monkeypatch.delenv("VOICE_AGENT_ORDERS_URL", raising=False)
    assert "unknown tool" in call_tool("get_order_status", {"order_id": "48213"})["error"]


def test_order_tools_against_local_service(monkeypatch):
    from datetime import timedelta

    from voice_agent.demo_service import ServiceThread, today
    from voice_agent.llm import _run_tool

    with ServiceThread() as svc:
        monkeypatch.setenv("VOICE_AGENT_ORDERS_URL", svc.url)
        ev = _run_tool("get_order_status", {"order_id": "४८२१३"})  # Devanagari digits normalised
        assert ev.result["status"] == "shipped" and ev.meta["attempts"] == 1 and "_meta" not in ev.result
        assert _run_tool("get_order_status", {"order_id": "4821"}).result["error"] == "rejected"  # no HTTP call
        assert _run_tool("get_order_status", {"order_id": "11111"}).result["error"] == "not_found"
        phone = _run_tool("find_orders_by_phone", {"phone": "+91 98765 43210"}).result
        assert [o["order_id"] for o in phone["orders"]] == ["48213", "51177"]
        new = (today() + timedelta(days=8)).isoformat()
        r = _run_tool("reschedule_delivery", {"order_id": "48213", "date": new}).result
        assert r["expected_delivery"] == new and svc.store.writes == 1
        assert _run_tool("reschedule_delivery", {"order_id": "51177", "date": new}).result["error"] == "rejected"
