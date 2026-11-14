"""Render results/*.json into markdown: results/RESULTS.md, and the generated block in README.md.

    python scripts/report.py            # RESULTS.md
    python scripts/report.py --readme   # also rewrite README.md between the results markers
    python scripts/report.py --check    # exit 1 if the README block differs from what this script generates

Every number in the README's results block comes from a committed results file through this script;
tests/test_readme_block.py fails when the block is stale.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def f(x, nd=2, unit=" s"):
    return "–" if x is None else f"{x:.{nd}f}{unit}"


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_any(*names):
    for n in names:
        r = load(n)
        if r:
            return r
    return None


BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"


def tool_counts(summary: dict) -> tuple[int, int, str]:
    tools = summary["tool_use"]
    ok = sum(int(v.split("/")[0]) for v in tools.values())
    tot = sum(int(v.split("/")[1]) for v in tools.values())
    return ok, tot, tools.get("tool:knowledge", "–")


def gemini_run():
    r = load("superseded/bench-gemini.json")
    if r:
        return r
    r = load("bench-main.json")  # before the final run replaced it
    return r if r and r["summary"]["llm"]["provider"] == "gemini" else None


def latency_steps() -> list[tuple[str, dict]]:
    steps = [
        ("Gemini 3.5 flash-lite (free tier), original code", gemini_run()),
        ("DeepSeek flash, thinking off, original prompt", load("bench-baseline-deepseek.json")),
        ("+ tool section in the system prompt", load("bench-prompt.json")),
        ("+ short STT window (end-of-turn sweep kept 544 ms)", load("bench-fast.json")),
        ("Final code, shipped defaults", load("bench-final.json")),
    ]
    return [(label, r["summary"]) for label, r in steps if r]


def latency_section() -> list[str]:
    steps = latency_steps()
    if not steps:
        return []
    out = [
        "### Latency: what each change bought",
        "",
        "End of speech to first reply audio, measured by wall clock with the audio fed in real time "
        "(`scripts/bench.py`, 20 spoken questions + 8 FLEURS sentences, 2 rounds = 56 turns). One run per row on a "
        "shared laptop CPU; the LLM is a live API, so gaps under about 0.5 s are within run-to-run noise.",
        "",
        "| Configuration | Turns without error | E2E p50 | E2E p95 | End-of-turn wait p50 | STT left p50 | "
        "LLM first sentence p50 | TTS first p50 | Tool used as expected | Knowledge-base tool |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, x in steps:
        st = x["stages"]
        ok, tot, kb = tool_counts(x)
        out.append(
            f"| {label} | {x['turns_ok']}/{x['turns']} | **{f(st['e2e_ttfa_s']['p50'])}** | {f(st['e2e_ttfa_s']['p95'])} | "
            f"{f(st['endpoint_delay_s']['p50'])} | {f(st['stt_s']['p50'])} | {f(st['llm_first_sentence_s']['p50'])} | "
            f"{f(st['tts_first_audio_s']['p50'])} | {ok}/{tot} | {kb} |"
        )
    last = steps[-1][1]
    out += ["", "Latency percentiles are over the turns without error. Errors are LLM requests that failed or hit the "
            "voice timeout (8 s without data); the agent speaks a short apology on those turns. "
            + "; ".join(f"{label}: {', '.join(x['errors'])}" for label, x in steps if x["errors"]) + ".", ""]
    sw = last.get("stt_window")
    if sw:
        window = ("always 30 s" if not sw["pad_s"] else
                  f"utterance + {sw['pad_s']:g} s below {sw['short_window_max_s']:g} s "
                  f"({sw['fallbacks']} of {last['turns']} turns fell back to 30 s)")
        out.append(
            f"Last row: STT window {window}, VAD end-of-turn "
            f"{last['vad']['end_silence_ms']:g} ms, speculative STT at {last['vad']['pause_ms']:g} ms "
            f"({last['speculative_decodes_wasted']} speculative decodes wasted). "
            f"LLM tokens for the run: {last['llm_tokens_total']['prompt']} prompt, {last['llm_tokens_total']['completion']} completion."
        )
        out.append("")
    return out


def routing_section() -> list[str]:
    r = load("tool-routing.json")
    if not r:
        return []
    inp = r["inputs"]
    out = [
        "### Tool routing, text only",
        "",
        f"`scripts/eval_tools.py`: each input goes straight to the LLM ({r['llm']['model']}, thinking "
        f"{r['llm']['thinking']}) as the first turn of a call, {r['reps']} repetitions, cache off. Inputs: "
        f"{inp['heard']} benchmark transcripts as the recogniser heard them, the same {inp['reference']} as written, and "
        f"{inp['extra']} more utterances (tool and no-tool, 3 deliberately garbled). v1 is the original one-sentence tool "
        "rule; v2 is the current tool section.",
        "",
        "| Prompt | Tool used when needed | Calculator | Date/time | Knowledge base | Unnecessary tool calls | "
        "First token p50, no tool / tool |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for v, d in r["versions"].items():
        a = d["summary"]["all"]
        bt = a["by_expected_tool"]
        ft = d["summary"]["first_token_p50_s"]
        out.append(
            f"| {v} | {a['tool_used_when_needed']} | {bt.get('calculator', '–')} | {bt.get('get_datetime', '–')} | "
            f"{bt.get('search_knowledge', '–')} | {a['unnecessary_tool_calls']} | {f(ft['no_tool'])} / {f(ft['with_tool'])} |"
        )
    v2 = r["versions"].get("v2", {}).get("summary")
    if v2:
        heard = v2["heard"]
        out += ["", f"v2 on the spoken-benchmark transcripts alone: tool used when needed {heard['tool_used_when_needed']}, "
                f"unnecessary tool calls {heard['unnecessary_tool_calls']}. Tokens for the whole evaluation: "
                f"{r['tokens_total']['prompt']} prompt, {r['tokens_total']['completion']} completion."]
    out.append("")
    return out


def wer_section(name: str = "wer-150-250.json") -> list[str]:
    w = load(name)
    if not w:
        return []
    out = [
        f"### Speech recognition: {w['data']}",
        "",
        f"{w['clips']} clips, {w['audio_minutes']} min. `30s` encodes Whisper's full 30 s window; `short` encodes the "
        f"utterance + {w.get('short_window', {}).get('pad_s', '–')} s and falls back to 30 s when the decode loops or "
        f"runs too long. Generation stops at 222 tokens, which long Hindi sentences can reach.",
        "",
        "| Decoding | WER | CER | Latency p50 | Latency p95 | Fallbacks to 30 s | Hit token limit | Detected as English |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, r in w["results"].items():
        out.append(f"| {mode} | {r['wer']:.2f} % | {r['cer']:.2f} % | {f(r['latency_p50_s'])} | {f(r['latency_p95_s'])} | "
                   f"{r.get('fallbacks_to_30s', '–')} | {r.get('hit_token_limit', '–')} | {r['detected_english']} |")
    out.append("")
    return out


def endpoint_section() -> list[str]:
    e = load("endpoint-sweep.json")
    if not e:
        return []
    out = [
        "### End-of-turn detection",
        "",
        f"`scripts/eval_endpoint.py`, VAD only: {e['data']}. A premature cut means the VAD ended the turn at a pause "
        f"before the speaker had finished. Rule: {e['rule']}. Chosen: **{e['chosen_end_silence_ms']} ms**.",
        "",
        "| Silence before end of turn | Questions cut early | FLEURS sentences cut early | Wasted speculative decodes per FLEURS clip |",
        "|---:|---:|---:|---:|",
    ]
    for r in e["sweep"]:
        if r["pause_ms"] != 160:
            continue
        q, fl = r["questions"], r["fleurs"]
        out.append(f"| {r['end_silence_ms']} ms | {q['premature_cut']}/{q['clips']} | {fl['premature_cut']}/{fl['clips']} "
                   f"({fl['premature_cut_pct']:.0f} %) | {fl['wasted_speculations_per_clip']:.2f} |")
    out.append("")
    return out


def readme_block() -> str:
    lines = [BEGIN, "<!-- generated by scripts/report.py --readme; do not edit by hand -->", ""]
    for section in (latency_section, routing_section, wer_section, endpoint_section):
        lines += section()
    lines.append(END)
    return "\n".join(lines)


def update_readme(check: bool = False) -> bool:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    a, b = text.index(BEGIN), text.index(END) + len(END)
    new = text[:a] + readme_block() + text[b:]
    if check:
        return new == text
    path.write_text(new, encoding="utf-8")
    return True


def stage_table(s: dict) -> list[str]:
    st = s["stages"]
    rows = [
        ("VAD, CPU per 32 ms frame", "vad_cpu_ms_per_frame", " ms", 3),
        ("End-of-turn wait (VAD hangover)", "endpoint_delay_s", " s", 2),
        ("Speech-to-text, left on the critical path", "stt_s", " s", 2),
        ("Speech-to-text, full decode", "stt_compute_s", " s", 2),
        ("LLM first token (incl. tool round trips)", "llm_first_token_s", " s", 2),
        ("LLM first complete sentence", "llm_first_sentence_s", " s", 2),
        ("TTS first sentence", "tts_first_audio_s", " s", 2),
        ("**End of speech → first reply audio**", "e2e_ttfa_s", " s", 2),
    ]
    out = ["| Stage | p50 | p95 | mean |", "|---|---:|---:|---:|"]
    for label, key, unit, nd in rows:
        v = st.get(key)
        if not v or not v["n"]:
            continue
        b = "**" if label.startswith("**") else ""
        out.append(f"| {label} | {b}{f(v['p50'], nd, unit)}{b} | {b}{f(v['p95'], nd, unit)}{b} | {f(v['mean'], nd, unit)} |")
    return out


def group_table(s: dict) -> list[str]:
    out = ["| Turns | n | p50 | p95 |", "|---|---:|---:|---:|"]
    for name, v in {"All": s["stages"]["e2e_ttfa_s"], **s["by_group"]}.items():
        if v["n"]:
            out.append(f"| {name} | {v['n']} | {f(v['p50'])} | {f(v['p95'])} |")
    return out


def _spec_line(turns: list[dict]) -> str:
    import statistics

    spec = [t for t in turns if t.get("stt_speculative")]
    if not spec:
        return "Speculative STT was off in this run."
    saved = [t["stt_compute_s"] - t["stt_s"] for t in spec]
    return (f"Speculative STT was used on {len(spec)}/{len(turns)} turns and took a median "
            f"{statistics.median(saved):.3f} s (min {min(saved):.3f}, max {max(saved):.3f}) of decode time off the "
            "critical path, measured within the same turn as full decode time minus the STT time left after the "
            "end of the turn. That is the 384 ms between the VAD's pause (160 ms) and end-of-turn (544 ms) events.")


def main() -> None:
    if "--check" in sys.argv:
        ok = update_readme(check=True)
        print("README results block is up to date" if ok else "README results block is stale: run scripts/report.py --readme")
        sys.exit(0 if ok else 1)
    lines = ["# Results", "", "Generated by `scripts/report.py` from the JSON files in this folder.", ""]
    lines += readme_block().splitlines()[3:-1]
    main_b, nospec = load("bench-main.json"), load("bench-nospec.json")
    if main_b:
        s = main_b["summary"]
        lines += [
            "## End-to-end latency",
            "",
            f"CPU: {s['cpu']} ({s['logical_cpus']} logical CPUs visible in WSL2), STT on {s['threads_stt']} threads. "
            f"LLM: {', '.join(f'{k} ({v} turns)' for k, v in s['llm_models_used'].items())}, live calls, cache off. "
            f"{s['turns']} turns ({s['rounds']} rounds over {s['turns'] // s['rounds']} prompts), {s['turns_ok']} without errors. "
            f"Audio fed in real time; speculative STT {'on' if s['speculative_stt'] else 'off'}. "
            f"Machine load average during turns: median {s['loadavg_1m_during_turns']['p50']:.1f}, "
            f"p95 {s['loadavg_1m_during_turns']['p95']:.1f} (other jobs share this machine). "
            f"Peak RSS of the whole process: {s['peak_rss_mb']:.0f} MB.",
            "",
            *stage_table(s),
            "",
            "### By group (end of speech → first reply audio)",
            "",
            *group_table(s),
            "",
            "LLM first token, no tool call: p50 {} / p95 {}; with a tool call (two LLM requests): p50 {} / p95 {}.".format(
                f(s["llm_by_group"]["no tool call"]["p50"]), f(s["llm_by_group"]["no tool call"]["p95"]),
                f(s["llm_by_group"]["with tool call"]["p50"]), f(s["llm_by_group"]["with tool call"]["p95"])),
            "",
            _spec_line(main_b["turns"]),
            "",
            f"Language ID (Hindi vs English, from the single encoder pass): English prompts {s['language_id_accuracy']['en']}, "
            f"Hindi and Hinglish prompts {s['language_id_accuracy']['hi+hinglish']}. "
            f"Expected tool used: calculator {s['tool_use'].get('tool:calculator')}, date/time {s['tool_use'].get('tool:datetime')}, "
            f"knowledge base {s['tool_use'].get('tool:knowledge')}.",
            "",
        ]
    ds = [("Gemini 3.5 flash-lite (free tier)", main_b), ("DeepSeek flash, provider default (thinking on)", load("bench-deepseek-default.json")),
          ("DeepSeek flash, thinking off", load("bench-baseline-deepseek.json"))]
    ds = [(label, r["summary"]) for label, r in ds if r]
    if len(ds) > 1:
        lines += [
            "## LLM provider comparison",
            "",
            "Same 56 turns, same recorded audio, same code and machine, one run per provider (back to back, so the "
            "free-tier service and the machine load differ between runs; treat gaps under about 0.5 s as noise).",
            "",
            "| LLM | Turns without error | LLM first token p50 / p95 | First sentence p50 / p95 | End of speech → first audio p50 / p95 | "
            "Tokens per turn (prompt / completion, mean) | Tool used as expected |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label, x in ds:
            st = x["stages"]
            tk = x.get("llm_tokens_per_turn")
            tools = x["tool_use"]
            ok = sum(int(v.split("/")[0]) for v in tools.values())
            tot = sum(int(v.split("/")[1]) for v in tools.values())
            lines.append(
                f"| {label} | {x['turns_ok']}/{x['turns']} | {f(st['llm_first_token_s']['p50'])} / {f(st['llm_first_token_s']['p95'])} | "
                f"{f(st['llm_first_sentence_s']['p50'])} / {f(st['llm_first_sentence_s']['p95'])} | "
                f"{f(st['e2e_ttfa_s']['p50'])} / {f(st['e2e_ttfa_s']['p95'])} | "
                f"{'not recorded' if not tk else '%.0f / %.0f' % (tk['prompt']['mean'], tk['completion']['mean'])} | {ok}/{tot} |"
            )
        lines += [
            "",
            "Cost per turn: Gemini ran on the free tier, so USD 0. For DeepSeek the whole experiment (four partial "
            "or full runs, about 224 turns, about 210k prompt and 21k completion tokens) took the balance from "
            "USD 0.68 to USD 0.65, i.e. roughly USD 0.00013 per turn (0.0001 to 0.0002 given that the balance is shown "
            "to the cent). Under USD 0.05 was spent in total.",
            "",
        ]
    runs = [("final (this code)", main_b)] + [
        (label, load(f"superseded/{name}"))
        for label, name in [("earlier, TTS memory arena on", "bench-main-arena-on.json"),
                            ("earlier, arena on, load not logged (7-10 seen by hand)", "bench-contended-arena-on.json")]
    ]
    runs = [(label, r) for label, r in runs if r]
    if len(runs) > 1:
        lines += [
            "## Run-to-run variation",
            "",
            "Three full runs of the same 56 turns on the same machine. The earlier two predate the TTS memory "
            "fix (arena off), which changes memory, not speed. LLM first-token time, which depends on the "
            "free-tier service, moves the most.",
            "",
            "| Run | Load avg (median) | STT p50 | LLM first token p50 | E2E p50 | E2E p95 | Peak RSS |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label, r in runs:
            x = r["summary"]
            st = x["stages"]
            load_med = x.get("loadavg_1m_during_turns", {}).get("p50")
            lines.append(
                f"| {label} | {'–' if load_med is None else f'{load_med:.1f}'} | {f(st['stt_s']['p50'])} | "
                f"{f(st['llm_first_token_s']['p50'])} | {f(st['e2e_ttfa_s']['p50'])} | {f(st['e2e_ttfa_s']['p95'])} | "
                f"{x['peak_rss_mb']:.0f} MB |"
            )
        lines.append("")
    if nospec:
        s2 = nospec["summary"]
        lines += [
            "## Separate run with speculative STT off",
            "",
            f"Same prompts, {s2['turns']} turns, run straight after the main benchmark "
            f"(load average median {s2['loadavg_1m_during_turns']['p50']:.1f}). The end-to-end difference between "
            "two runs mostly reflects free-tier LLM latency drifting between runs, so use the within-turn "
            "measurement above for the effect of speculation; this run is here for completeness.",
            "",
            *stage_table(s2),
            "",
        ]
    for name in sorted(RES.glob("superseded/wer-*.json")):
        w = json.loads(name.read_text(encoding="utf-8"))
        lines += [
            f"## Word error rate, earlier run (superseded): {w['data']}",
            "",
            f"{w['clips']} clips, {w['audio_minutes']} min of audio, model `{w['model']}`, {w['decoding']}. "
            f"Normaliser: {w['normaliser']}. CPU: {w['cpu']}, {w['threads']} threads. Peak RSS {w['peak_rss_mb']:.0f} MB.",
            "",
            "| Decoding | WER | CER | RTF | Latency p50 | Latency p95 | Clips detected as English |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for mode, r in w["results"].items():
            lines.append(f"| {mode} | {r['wer']:.2f} % | {r['cer']:.2f} % | {r['rtf']:.3f} | {f(r['latency_p50_s'])} | "
                         f"{f(r['latency_p95_s'])} | {r['detected_english']} |")
        lines.append("")
    tts = load("tts.json")
    if tts:
        lines += ["## Text-to-speech", "", "| Voice | RTF | Mean synthesis per sentence | Round-trip WER | Round-trip CER |",
                  "|---|---:|---:|---:|---:|"]
        for v, r in tts.items():
            if isinstance(r, dict):
                lines.append(f"| `{v}` | {r['rtf']:.3f} | {f(r['mean_synth_s_per_sentence'])} | {r['roundtrip_wer']:.1f} % | "
                             f"{r['roundtrip_cer']:.1f} % |")
        lines += ["", "Round trip: synthesise fixed sentences, transcribe them with the agent's own recogniser, score "
                  "against the input text. It is a proxy for intelligibility that inherits the recogniser's errors.", ""]
    (RES / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    if "--readme" in sys.argv:
        update_readme()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
