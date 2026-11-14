"""Fault evaluation of the order tools' REST client against the real local service, over HTTP.

No LLM and no audio. The service runs in a thread (uvicorn on a free local port); before each call
its data and faults are reset and the fault mode is applied. Each mode runs ``--calls`` times and
records whether the call ended in the expected outcome (the order data, or the expected
error kind the agent can speak about), attempts, seconds, and whether the 5 s budget held.

    python scripts/eval_rest_tool.py          # writes results/rest-tool.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import timedelta
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_agent.demo_service import ServiceThread, today  # noqa: E402
from voice_agent.orders_client import OrdersClient  # noqa: E402

# mode: (faults, call, expected outcome)
MODES = {
    "none": ({}, "get", "ok"),
    "latency 300 ms": ({"latency_ms": 300}, "get", "ok"),
    "one 503, then success": ({"fail_503": 1}, "get", "ok"),
    "two 503s, then success": ({"fail_503": 2}, "get", "ok"),
    "three 503s (more than the retries)": ({"fail_503": 3}, "get", "unavailable"),
    "429 with Retry-After 1 s": ({"rate_limit_429": 1, "retry_after_s": 1}, "get", "ok"),
    "malformed body once": ({"malformed": 1}, "get", "ok"),
    "hang 4 s (longer than the read timeout)": ({"hang_s": 4}, "get", "timeout"),
    "unknown order id": ({}, "get_unknown", "not_found"),
    "reschedule, one 503 then success": ({"fail_503": 1}, "reschedule", "ok"),
    "reschedule a delivered order": ({}, "reschedule_delivered", "rejected"),
}


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "results" / "rest-tool.json"))
    args = ap.parse_args()
    results = {}
    with ServiceThread() as svc:
        admin = httpx.Client(base_url=svc.url, headers={"X-API-Key": svc.api_key})
        client = OrdersClient(base_url=svc.url, api_key=svc.api_key)
        for mode, (faults, call, expected) in MODES.items():
            rows = []
            for _ in range(args.calls):
                admin.post("/_admin/reset")
                if faults:
                    admin.post("/_admin/faults", json=faults)
                new = (today() + timedelta(days=9)).isoformat()
                if call == "get":
                    r = client.get_order("48213")
                elif call == "get_unknown":
                    r = client.get_order("11111")
                elif call == "reschedule":
                    r = client.reschedule("48213", new)
                else:
                    r = client.reschedule("51177", new)
                meta = r.pop("_meta")
                outcome = r.get("error", "ok")
                writes = svc.store.writes
                ok = outcome == expected and (call != "reschedule" or (writes == 1 and r.get("expected_delivery") == new))
                rows.append({"outcome": outcome, "ok": ok, "writes": writes, **meta})
            secs = [x["seconds"] for x in rows]
            results[mode] = {
                "expected": expected,
                "correct": sum(x["ok"] for x in rows),
                "calls": len(rows),
                "attempts_mean": round(statistics.fmean(x["attempts"] for x in rows), 2),
                "seconds_p50": round(pct(secs, 0.5), 3),
                "seconds_p95": round(pct(secs, 0.95), 3),
                "seconds_max": round(max(secs), 3),
                "within_budget": sum(x["within_budget"] for x in rows),
                "outcomes": {o: sum(x["outcome"] == o for x in rows) for o in sorted({x["outcome"] for x in rows})},
            }
            print(f"{mode:<42} {results[mode]['correct']}/{len(rows)} p50={results[mode]['seconds_p50']} "
                  f"max={results[mode]['seconds_max']} attempts={results[mode]['attempts_mean']}", flush=True)
    out = {
        "service": "local stand-in order service (src/voice_agent/demo_service.py) over HTTP on 127.0.0.1",
        "client": {"connect_timeout_s": client.connect_timeout, "read_timeout_s": client.read_timeout,
                   "budget_s": client.budget_s, "max_retries": client.max_retries, "backoff_s": client.backoff_s},
        "calls_per_mode": args.calls,
        "total": {"correct": sum(r["correct"] for r in results.values()), "calls": sum(r["calls"] for r in results.values()),
                  "within_budget": sum(r["within_budget"] for r in results.values())},
        "modes": results,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["total"]))


if __name__ == "__main__":
    main()
