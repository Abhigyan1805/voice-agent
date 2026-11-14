"""End-to-end latency benchmark on a fixed set of recorded prompts.

Each prompt's audio is delivered to the streaming VAD in 32 ms frames **at real-time pace**, as a
microphone would deliver it. With speculative STT on (the default, as in the live app), the
decode starts when the VAD sees 160 ms of silence; the turn is confirmed after 544 ms of silence,
and the turn then runs exactly as in the live app: STT -> streamed LLM (live API call, cache
off) -> sentence-by-sentence TTS.

Reported per turn (all wall clock):
  endpoint_delay   last speech frame delivered -> VAD confirms the end of the turn
  stt              STT time left on the critical path after the turn is confirmed
  stt_compute      full STT decode time (equal to stt when speculation is off)
  llm_first_token  LLM request sent -> first text token (includes any tool round-trip)
  tts_first_audio  synthesis time of the first sentence
  e2e_ttfa         last speech frame delivered -> first reply audio ready to play
                   (= endpoint_delay + stt + LLM to first sentence + first-sentence TTS)

    python scripts/bench.py --rounds 2                    # 20 synthetic prompts + 8 FLEURS clips
    python scripts/bench.py --no-speculative --rounds 1 --tag nospec
    python scripts/bench.py --model gemini-3.5-flash-lite --tag g35
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_agent.audio import load_audio  # noqa: E402
from voice_agent.config import Settings, load_env_file  # noqa: E402
from voice_agent.pipeline import SpeculativeSTT, TurnDone  # noqa: E402
from voice_agent.vad import FRAME, SileroScorer, SpeechEnd, SpeechPause, VadConfig, VadSegmenter  # noqa: E402

def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(np.floor(k)), int(np.ceil(k))
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def fleurs_prompts(n: int, offset: int) -> list[dict]:
    """Real read speech: FLEURS hi_in test clips (CC-BY-4.0), in file-name order, from ``offset``."""
    from voice_agent.fleurs import load_clips

    try:
        clips = load_clips(offset, offset + n)
    except FileNotFoundError as e:
        print(f"{e}; skipping real-speech prompts", file=sys.stderr)
        return []
    pad = lambda a: np.concatenate([np.zeros(8000, np.float32), a, np.zeros(19_200, np.float32)])
    return [{"id": "fleurs-" + c.uid, "lang": "hi", "category": "fleurs-real", "text": c.text, "audio": pad(c.audio)}
            for c in clips]


class Pacer:
    """Keep LLM requests under ``rpm`` per minute (free-tier friendly)."""

    def __init__(self, rpm: int):
        self.rpm, self.times = rpm, deque()

    def wait(self, n_requests: int = 2) -> None:
        now = time.time()
        while self.times and now - self.times[0] > 60:
            self.times.popleft()
        if len(self.times) + n_requests > self.rpm:
            sleep = 60 - (now - self.times[0]) + 0.5
            time.sleep(max(0.0, sleep))
            return self.wait(n_requests)

    def record(self, n: int) -> None:
        self.times.extend([time.time()] * n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--fleurs", type=int, default=8, help="number of real FLEURS clips to add")
    ap.add_argument("--fleurs-offset", type=int, default=300)
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--model", default=None)
    ap.add_argument("--thinking", choices=["on", "off"], default=None, help="DeepSeek reasoning switch")
    ap.add_argument("--rpm", type=int, default=12)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--filler", action="store_true")
    ap.add_argument("--stt-pad-s", type=float, default=None, help="short STT window padding; 0 = always 30 s")
    ap.add_argument("--end-silence-ms", type=float, default=None, help="VAD end-of-turn hangover")
    ap.add_argument("--pause-ms", type=float, default=None, help="VAD pause that starts speculative STT")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-realtime", dest="realtime", action="store_false",
                    help="feed audio as fast as possible and add the VAD hangover arithmetically")
    ap.add_argument("--no-speculative", dest="speculative", action="store_false",
                    help="do not start STT at the VAD pause; transcribe only after end of turn")
    args = ap.parse_args()

    if args.thinking:
        os.environ["VOICE_AGENT_LLM_THINKING"] = args.thinking
    load_env_file()
    from voice_agent.factory import build_pipeline

    s = Settings()
    s.llm_provider, s.llm_model, s.llm_cache = args.provider, args.model, False
    if args.stt_pad_s is not None:
        s.stt_pad_s = args.stt_pad_s
    vad_cfg = VadConfig()
    if args.end_silence_ms is not None:
        vad_cfg.end_silence_ms = args.end_silence_ms
    if args.pause_ms is not None:
        vad_cfg.pause_ms = args.pause_ms
    t_load = time.perf_counter()
    pipe = build_pipeline(s, filler=args.filler)
    load_s = time.perf_counter() - t_load
    scorer = SileroScorer()

    prompts = json.loads((ROOT / "prompts" / "prompts.json").read_text(encoding="utf-8"))
    for p in prompts:
        p["audio"] = load_audio(ROOT / "prompts" / "audio" / f"{p['id']}.wav")
        p["source"] = "synthetic (Piper)"
    real = fleurs_prompts(args.fleurs, args.fleurs_offset)
    for p in real:
        p["source"] = "FLEURS test (real speech)"
    prompts = (prompts + real)[: args.limit]

    # warm-up (not recorded): one STT decode and one TTS sentence
    pipe.stt.transcribe(prompts[0]["audio"][8000:-19200])
    pipe.tts.synthesize("नमस्ते।")

    pacer = Pacer(args.rpm)
    rows = []
    for rnd in range(args.rounds):
        for p in prompts:
            pipe.agent.reset()
            pacer.wait()
            seg = VadSegmenter(scorer, vad_cfg)
            vad_cpu, end, spec, n_spec = 0.0, None, None, 0
            audio = p["audio"]
            frame_wall: list[float] = []
            t_start = time.perf_counter()
            for k, i in enumerate(range(0, len(audio), FRAME)):
                if args.realtime:  # deliver each 32 ms frame when a microphone would
                    delay = t_start + (k + 1) * FRAME / 16_000 - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                frame_wall.append(time.perf_counter())
                t = time.perf_counter()
                evs = list(seg.feed(audio[i : i + FRAME]))
                vad_cpu += time.perf_counter() - t
                for ev in evs:
                    if isinstance(ev, SpeechPause) and args.speculative and args.realtime:
                        spec = SpeculativeSTT(pipe.stt, ev.audio, ev.pause_id, prompt=pipe.stt_prompt())
                        n_spec += 1
                    elif isinstance(ev, SpeechEnd):
                        end = ev
                if end is not None:
                    break
            if end is None:
                end = next((e for e in seg.flush() if isinstance(e, SpeechEnd)), None)
            if end is None:
                print(f"{p['id']}: VAD found no speech", file=sys.stderr)
                continue
            if spec is not None and spec.pause_id != end.pause_id:
                spec = None  # user resumed after the pause; speculative transcript is stale
            t_turn = time.perf_counter()
            metrics = None
            for ev in pipe.run_turn(audio=end.audio, speculative=spec):
                if isinstance(ev, TurnDone):
                    metrics = ev.metrics
            pacer.record(max(1, metrics.llm_requests))
            m = metrics.as_dict()
            if args.realtime:
                # wall clock: the last speech frame reached the VAD -> first reply audio was ready
                last_speech_frame = max(0, int(round(end.t_end * 16_000 / FRAME)) - 1)
                t_speech_end = frame_wall[min(last_speech_frame, len(frame_wall) - 1)]
                endpoint = t_turn - t_speech_end
                e2e = (t_turn + m["time_to_first_audio_s"] - t_speech_end) if m["time_to_first_audio_s"] else None
            else:
                endpoint = end.t_detected - end.t_end
                e2e = endpoint + m["time_to_first_audio_s"] if m["time_to_first_audio_s"] else None
            row = {
                "round": rnd,
                "id": p["id"],
                "lang": p["lang"],
                "category": p["category"],
                "source": p["source"],
                "reference": p["text"],
                "stt_language": spec.language if spec is not None else pipe.stt.last_language,
                "utterance_s": round(end.duration, 3),
                "vad_cpu_ms_per_frame": round(1000 * vad_cpu / max(1, len(frame_wall)), 3),
                "endpoint_delay_s": round(endpoint, 3),
                "speculative_decodes_wasted": n_spec - (1 if spec is not None else 0),
                **m,
                "e2e_ttfa_s": round(e2e, 4) if e2e is not None else None,
                # other processes share this machine; record how busy it was during the turn
                "loadavg_1m": round(os.getloadavg()[0], 2),
                "mem_available_mb": _mem_available_mb(),
            }
            rows.append(row)
            print(
                f"[{rnd}] {p['id']:<28} stt={m['stt_s']:.2f} llm1={m['llm_first_token_s'] or -1:.2f} "
                f"tts1={m['tts_first_audio_s'] or -1:.2f} e2e={row['e2e_ttfa_s'] or -1:.2f} tools={m['tools']} "
                f"retries={m['llm_retries']} err={m['error']}\n      heard: {m['transcript']}\n      said : {m['reply'][:110]}",
                flush=True,
            )

    ok = [r for r in rows if not r["error"] and r["e2e_ttfa_s"] is not None]

    def stats(key, subset=ok):
        xs = [r[key] for r in subset if r.get(key) is not None]
        return {"n": len(xs), "p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "mean": statistics.fmean(xs) if xs else None}

    summary = {
        "tag": args.tag,
        "cpu": cpu_model(),
        "threads_stt": s.stt_threads,
        "llm": {"provider": pipe.agent.client.name, "model": pipe.agent.client.model,
                "thinking": getattr(pipe.agent.client, "thinking", None)},
        "llm_tokens_per_turn": {"prompt": stats("llm_prompt_tokens", rows), "completion": stats("llm_completion_tokens", rows)},
        "llm_requests_per_turn": stats("llm_requests", rows),
        "stt_model": str(s.stt_model.name),
        "tts_voices": [Path(s.tts_hindi).stem, Path(s.tts_english).stem],
        "rounds": args.rounds,
        "realtime_feed": args.realtime,
        "vad": {"end_silence_ms": vad_cfg.end_silence_ms, "pause_ms": vad_cfg.pause_ms},
        "stt_window": {"pad_s": s.stt_pad_s, "short_window_max_s": s.stt_short_window_max_s,
                       "fallbacks": sum(bool(r.get("stt_fallback")) for r in rows)},
        "speculative_decodes_wasted": sum(r["speculative_decodes_wasted"] for r in rows),
        "llm_tokens_total": {"prompt": sum(r["llm_prompt_tokens"] for r in rows),
                             "completion": sum(r["llm_completion_tokens"] for r in rows)},
        "speculative_stt": args.speculative and args.realtime,
        "turns": len(rows),
        "turns_ok": len(ok),
        "errors": [r["error"] for r in rows if r["error"]],
        "llm_models_used": {m: sum(r["llm_model"] == m for r in rows) for m in sorted({str(r["llm_model"]) for r in rows})},
        "model_load_s": round(load_s, 2),
        "logical_cpus": os.cpu_count(),
        "loadavg_1m_during_turns": stats("loadavg_1m", rows),
        "mem_available_mb_during_turns": stats("mem_available_mb", rows),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "stages": {k: stats(k) for k in ("vad_cpu_ms_per_frame", "endpoint_delay_s", "stt_s", "stt_compute_s",
                                         "llm_ttfb_s", "llm_first_token_s",
                                         "llm_first_sentence_s", "tts_first_audio_s", "time_to_first_audio_s",
                                         "e2e_ttfa_s")},
        "by_group": {
            name: stats("e2e_ttfa_s", [r for r in ok if pred(r)])
            for name, pred in {
                "no tool call": lambda r: not r["tools"],
                "with tool call": lambda r: bool(r["tools"]),
                "synthetic prompts": lambda r: r["source"].startswith("synthetic"),
                "FLEURS real speech": lambda r: r["source"].startswith("FLEURS"),
                "Hindi": lambda r: r["lang"] == "hi",
                "Hinglish": lambda r: r["lang"] == "hinglish",
                "English": lambda r: r["lang"] == "en",
            }.items()
        },
        "llm_by_group": {
            "no tool call": stats("llm_first_token_s", [r for r in ok if not r["tools"]]),
            "with tool call": stats("llm_first_token_s", [r for r in ok if r["tools"]]),
        },
        "language_id_accuracy": {
            "en": _acc([r for r in rows if r["lang"] == "en"], "en"),
            "hi+hinglish": _acc([r for r in rows if r["lang"] != "en"], "hi"),
        },
        "tool_use": {
            cat: f"{sum(1 for r in rows if r['category'] == cat and cat.split(':')[1] in _TOOL_OF and _TOOL_OF[cat.split(':')[1]] in r['tools'])}"
            f"/{sum(1 for r in rows if r['category'] == cat)}"
            for cat in sorted({r["category"] for r in rows if r["category"].startswith("tool:")})
        },
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"bench-{args.tag}.json").write_text(
        json.dumps({"summary": summary, "turns": rows}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def _mem_available_mb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


_TOOL_OF = {"calculator": "calculator", "datetime": "get_datetime", "knowledge": "search_knowledge"}


def _acc(rows, lang):
    return f"{sum(r['stt_language'] == lang for r in rows)}/{len(rows)}"


if __name__ == "__main__":
    main()
