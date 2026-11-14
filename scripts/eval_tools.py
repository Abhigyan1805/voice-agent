"""Text-only tool-routing evaluation: does the LLM call the right tool (or none) for each input?

No audio, no STT, no TTS: each input text goes straight to the LLM with an empty history, as the
first turn of a conversation. Inputs, each labelled with the tool it should trigger ("none" for
no tool):

  heard      the 28 benchmark transcripts exactly as the recogniser produced them
             (round 0 of results/bench-baseline-deepseek.json), errors included
  reference  the same 28 prompts as written (20 benchmark prompts + 8 FLEURS sentences)
  extra      prompts/tool_routing.json: 21 more utterances, tool and no-tool, some garbled

Each prompt version (v1 = the original one-sentence tool rule and tool descriptions, v2 = the
current tool section) runs every input ``--reps`` times with the response cache off.

    python scripts/eval_tools.py --reps 3          # writes results/tool-routing.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_agent import tools as toolbox  # noqa: E402
from voice_agent.agent import SYSTEM_PROMPT, SYSTEM_PROMPT_V1  # noqa: E402
from voice_agent.config import load_env_file  # noqa: E402
from voice_agent.llm import TextDelta, ToolEvent, make_client  # noqa: E402

TOOL_OF = {"tool:calculator": "calculator", "tool:datetime": "get_datetime", "tool:knowledge": "search_knowledge"}

# Tool descriptions as shipped before the v2 prompt (commit a7c54c2), for an honest before/after.
V1_DESCRIPTIONS = {
    "calculator": "Evaluate an arithmetic expression exactly. Use it for any arithmetic, percentages or unit "
    "maths instead of computing in your head. Write the expression with digits and + - * / ** ( ) "
    "and functions sqrt, round, log, sin, cos; '15% of 200' is accepted.",
    "get_datetime": "Get the current date, time and weekday. Use it whenever the user asks about today, now, "
    "the time, the day or the date.",
    "search_knowledge": "Search a small local knowledge base about this voice assistant itself (what it can do, how it "
    "works, which models it uses) and about India's emergency helpline numbers. Use it for questions "
    "on those topics.",
}


def load_inputs() -> list[dict]:
    prompts = {p["id"]: p for p in json.loads((ROOT / "prompts" / "prompts.json").read_text(encoding="utf-8"))}
    bench = json.loads((ROOT / "results" / "bench-baseline-deepseek.json").read_text(encoding="utf-8"))
    items = []
    for t in bench["turns"]:
        if t["round"] != 0:
            continue
        expect = TOOL_OF.get(t["category"], "none")
        items.append({"set": "heard", "id": t["id"], "lang": t["lang"], "expect": expect, "text": t["transcript"]})
        items.append({"set": "reference", "id": t["id"], "lang": t["lang"], "expect": expect,
                      "text": prompts[t["id"]]["text"] if t["id"] in prompts else t["reference"]})
    for x in json.loads((ROOT / "prompts" / "tool_routing.json").read_text(encoding="utf-8")):
        items.append({"set": "extra", "id": x["id"], "lang": x["lang"], "expect": x["expect"], "text": x["text"],
                      "garbled": x.get("garbled", False)})
    return items


def run_one(provider: str, model: str | None, system: str, item: dict) -> dict:
    client = make_client(provider, model, cache=False)
    t0 = time.perf_counter()
    first, tools, args, reply, error = None, [], [], [], None
    try:
        for ev in client.turn(system, [], item["text"]):
            if isinstance(ev, ToolEvent):
                tools.append(ev.name)
                args.append(ev.args)
            elif isinstance(ev, TextDelta):
                if first is None and ev.text.strip():
                    first = time.perf_counter() - t0
                reply.append(ev.text)
    except Exception as e:  # recorded, not raised: one failed request must not end the run
        error = f"{type(e).__name__}: {e}"[:200]
    info = client.info
    return {
        "tools": tools, "args": args, "reply": "".join(reply).strip(), "error": error,
        "first_token_s": round(first, 3) if first is not None else None,
        "requests": info.requests, "prompt_tokens": info.prompt_tokens, "completion_tokens": info.completion_tokens,
        "model": getattr(client, "model", None), "thinking": getattr(client, "thinking", None),
    }


def summarise(rows: list[dict]) -> dict:
    def correct(r):
        return (not r["tools"]) if r["expect"] == "none" else (r["expect"] in r["tools"])

    out: dict = {}
    for name, sub in [("all", rows)] + [(s, [r for r in rows if r["set"] == s]) for s in ("heard", "reference", "extra")]:
        by_tool = {}
        for tool in ("calculator", "get_datetime", "search_knowledge", "none"):
            xs = [r for r in sub if r["expect"] == tool]
            if xs:
                by_tool[tool] = f"{sum(map(correct, xs))}/{len(xs)}"
        tool_rows = [r for r in sub if r["expect"] != "none"]
        none_rows = [r for r in sub if r["expect"] == "none"]
        out[name] = {
            "correct": f"{sum(map(correct, sub))}/{len(sub)}",
            "tool_used_when_needed": f"{sum(map(correct, tool_rows))}/{len(tool_rows)}",
            "unnecessary_tool_calls": f"{sum(bool(r['tools']) for r in none_rows)}/{len(none_rows)}",
            "by_expected_tool": by_tool,
        }
    confusion = Counter((r["expect"], r["tools"][0] if r["tools"] else "none") for r in rows)
    out["confusion"] = [{"expected": e, "called_first": c, "n": n} for (e, c), n in sorted(confusion.items())]
    ft = [r["first_token_s"] for r in rows if r["first_token_s"] is not None]
    ft_tool = [r["first_token_s"] for r in rows if r["first_token_s"] is not None and r["tools"]]
    ft_none = [r["first_token_s"] for r in rows if r["first_token_s"] is not None and not r["tools"]]
    med = lambda xs: round(statistics.median(xs), 3) if xs else None  # noqa: E731
    out["first_token_p50_s"] = {"all": med(ft), "with_tool": med(ft_tool), "no_tool": med(ft_none)}
    out["errors"] = sum(1 for r in rows if r["error"])
    out["tokens"] = {"prompt": sum(r["prompt_tokens"] for r in rows),
                     "completion": sum(r["completion_tokens"] for r in rows)}
    out["over_triggers"] = [{"id": r["id"], "set": r["set"], "text": r["text"], "tools": r["tools"]}
                            for r in none_rows_all(rows) if r["tools"]]
    out["misses"] = [{"id": r["id"], "set": r["set"], "text": r["text"], "expect": r["expect"], "tools": r["tools"]}
                     for r in rows if r["expect"] != "none" and not correct(r)]
    return out


def none_rows_all(rows):
    return [r for r in rows if r["expect"] == "none"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--model", default=None)
    ap.add_argument("--versions", default="v1,v2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "tool-routing.json"))
    args = ap.parse_args()
    load_env_file()
    items = load_inputs()[: args.limit]
    current = {n: t.description for n, t in toolbox.TOOLS.items()}
    result = {"inputs": {s: sum(i["set"] == s for i in items) for s in ("heard", "reference", "extra")},
              "reps": args.reps, "versions": {}}
    for version in args.versions.split(","):
        system = SYSTEM_PROMPT_V1 if version == "v1" else SYSTEM_PROMPT
        for n, t in toolbox.TOOLS.items():
            t.description = V1_DESCRIPTIONS[n] if version == "v1" else current[n]
        jobs = [(rep, it) for rep in range(args.reps) for it in items]
        with ThreadPoolExecutor(args.workers) as pool:
            outs = list(pool.map(lambda j: run_one(args.provider, args.model, system, j[1]), jobs))
        rows = [{**it, "rep": rep, **o} for (rep, it), o in zip(jobs, outs)]
        summary = summarise(rows)
        result["versions"][version] = {"summary": summary, "rows": rows}
        result["llm"] = {"provider": args.provider, "model": rows[0]["model"], "thinking": rows[0]["thinking"]}
        print(version, json.dumps({k: v for k, v in summary.items() if k not in ("over_triggers", "misses", "confusion")},
                                  ensure_ascii=False), flush=True)
    for n, t in toolbox.TOOLS.items():
        t.description = current[n]
    result["tokens_total"] = {
        k: sum(v["summary"]["tokens"][k] for v in result["versions"].values()) for k in ("prompt", "completion")
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("tokens", result["tokens_total"])


if __name__ == "__main__":
    main()
