"""Run the scripted call suite and score it (see src/voice_agent/callsim.py for the rules).

The local order service starts in a thread on a free port; its data and faults are reset before
every call. Each call runs ``--reps`` times in the spoken track (recorded audio through VAD, STT,
LLM with tools, TTS) and in the oracle-text track (reference text straight to the LLM). Audio is
fed without real-time pacing: latency is the benchmark's job, but stage times are still recorded.

    python scripts/make_call_audio.py                       # once: render the callers' audio
    python scripts/eval_calls.py --reps 3 --tag clean       # writes results/calls-clean.json
    python scripts/eval_calls.py --condition tel --oracle-from results/calls-clean.json --tag tel
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_agent.callsim import CallRunner, attribute, load_calls, summarise  # noqa: E402
from voice_agent.config import Settings, load_env_file  # noqa: E402
from voice_agent.demo_service import ServiceThread, today  # noqa: E402


def rescore_file(path: Path) -> None:
    from voice_agent.callsim import rescore

    r = json.loads(path.read_text(encoding="utf-8"))
    suite = load_calls(ROOT / "evals" / "calls" / "calls.json")
    rescore(r["calls"], suite, date.fromisoformat(r["service_date"]))
    r["summary"] = summarise(r["calls"])
    r["checks_sha1"] = hashlib.sha1((ROOT / "evals" / "calls" / "calls.json").read_bytes()).hexdigest()[:12]
    r["rescored"] = True
    path.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(r["summary"], ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default="clean")
    ap.add_argument("--condition", default="clean", help="audio condition from evals/conditions.json")
    ap.add_argument("--oracle-from", default=None, help="reuse the oracle track of an earlier results file")
    ap.add_argument("--calls", default=None, help="comma-separated call ids (default: all)")
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--rescore", default=None, help="re-check a stored results file against the current call specs")
    args = ap.parse_args()
    if args.rescore:
        return rescore_file(Path(args.rescore))

    load_env_file()
    suite = load_calls(ROOT / "evals" / "calls" / "calls.json")
    calls = [c for c in suite["calls"] if not args.calls or c["id"] in args.calls.split(",")]
    condition = None
    if args.condition != "clean":
        from voice_agent.degrade import make_condition

        condition = make_condition(args.condition)
    reused = {}
    if args.oracle_from:
        prev = json.loads(Path(args.oracle_from).read_text(encoding="utf-8"))
        reused = {(c["id"], c["rep"]): c for c in prev["calls"] if c["track"] == "oracle"}

    with ServiceThread() as svc:
        os.environ["VOICE_AGENT_ORDERS_URL"] = svc.url
        os.environ["VOICE_AGENT_ORDERS_KEY"] = svc.api_key
        from voice_agent.agent import build_system_prompt
        from voice_agent.factory import build_pipeline
        from voice_agent.vad import SileroScorer

        s = Settings()
        s.llm_provider, s.llm_cache = args.provider, False
        pipe = build_pipeline(s)
        runner = CallRunner(pipe, svc, scorer=SileroScorer(), condition=condition,
                            audio_dir=ROOT / "evals" / "calls" / "audio", today=today(), words=suite["status_words"])
        out_calls = []
        t0 = time.time()
        for rep in range(args.reps):
            for call in calls:
                spoken = runner.run_call(call, "spoken")
                if (call["id"], rep) in reused:
                    oracle = reused[(call["id"], rep)]
                else:
                    oracle = runner.run_call(call, "oracle")
                    oracle["rep"] = rep
                spoken["rep"] = rep
                spoken["attribution"] = attribute(spoken, oracle)
                oracle.setdefault("attribution", {"stage": None, "turn": None})
                out_calls += [spoken] if (call["id"], rep) in reused else [spoken, oracle]
                ok = all(t["passed"] for t in spoken["turns"]) and spoken["final_state_ok"]
                print(f"[{rep}] {call['id']:<24} spoken={'PASS' if ok else 'FAIL'} stage={spoken['attribution']['stage']} "
                      f"oracle={'PASS' if all(t['passed'] for t in oracle['turns']) and oracle['final_state_ok'] else 'FAIL'}",
                      flush=True)
                tracks = [("spoken", spoken)] + ([] if (call["id"], rep) in reused else [("oracle", oracle)])
                for name, run in tracks:
                    for t in run["turns"]:
                        if not t["passed"]:
                            print(f"      {name} t{t['turn']} heard: {t['transcript'][:90]}\n         said: {t['reply'][:120]}\n"
                                  f"         checks={t['checks']} tools={[(x['name'], x['args']) for x in t['tools']]}",
                                  flush=True)
        reused_oracle = [c for c in reused.values() if c["id"] in {x["id"] for x in calls}]
        summary = summarise(out_calls + reused_oracle)
    prompt = build_system_prompt()
    result = {
        "tag": args.tag,
        "checks_sha1": hashlib.sha1((ROOT / "evals" / "calls" / "calls.json").read_bytes()).hexdigest()[:12],
        "condition": args.condition,
        "run_date": date.today().isoformat(),
        "service_date": today().isoformat(),
        "reps": args.reps,
        "calls": out_calls,
        "oracle_reused_from": args.oracle_from,
        "llm": {"provider": pipe.agent.client.name, "model": pipe.agent.client.model,
                "thinking": getattr(pipe.agent.client, "thinking", None)},
        "system_prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest()[:12],
        "system_prompt": prompt,
        "wall_s": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "summary": summary,
    }
    path = ROOT / "results" / f"calls-{args.tag}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
