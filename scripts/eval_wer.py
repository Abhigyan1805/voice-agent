"""Word error rate of the agent's speech recogniser on held-out Hindi speech.

Data: FLEURS hi_in *test* clips 150-249 in file-name order (100 clips, real read speech). The
test split was never used for training or model selection, and clips 0-149 were already used for
the CPU int8 benchmark in the fine-tuning repo, so this slice is new to both.

Decoding settings are scored with the same model and clips, named ``<window>/<language>``:
  window    30s    the encoder always sees a 30 s window (Whisper's training setup)
            short  utterance + pad_s seconds below short_window_max_s, with a fallback to 30 s
                   when the short decode loops or runs too long (the agent's default)
  language  forced-hi  language token fixed to Hindi (what the fine-tuning repo reports)
            auto       one encoder pass, Hindi-vs-English language ID, then decode (the agent)

    python scripts/eval_wer.py                 # writes results/wer-150-250.json
    python scripts/eval_wer.py --start 0 --stop 150   # the fine-tuning repo's CPU subset, for comparison
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_agent.config import Settings  # noqa: E402
from voice_agent.fleurs import load_clips, normalize  # noqa: E402
from voice_agent.stt import TOKEN_LIMIT, WhisperSTT  # noqa: E402


def score(refs, hyps):
    import jiwer

    r = [normalize(x) for x in refs]
    h = [normalize(x) for x in hyps]
    pairs = [(a, b) for a, b in zip(r, h) if a]
    r, h = [a for a, _ in pairs], [b for _, b in pairs]
    return {"wer": round(100 * jiwer.wer(r, h), 2), "cer": round(100 * jiwer.cer(r, h), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=150)
    ap.add_argument("--stop", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--modes", default="30s/forced-hi,30s/auto,short/auto")
    ap.add_argument("--pad-s", type=float, default=None, help="short-window padding (default: Settings)")
    args = ap.parse_args()

    clips = load_clips(args.start, args.stop)
    s = Settings()
    pad_s = s.stt_pad_s if args.pad_s is None else args.pad_s
    stt = WhisperSTT(s.stt_model, threads=args.threads, pad_s=pad_s, short_window_max_s=s.stt_short_window_max_s)
    stt.transcribe(clips[0].audio)  # warm-up
    results = {}
    rows = [{"uid": c.uid, "ref": c.text, "duration_s": round(c.duration, 2)} for c in clips]
    for mode in args.modes.split(","):
        window, language = mode.split("/")
        stt.pad_s = 0.0 if window == "30s" else pad_s
        lat, hyps, langs, fallbacks, limit_hits = [], [], [], 0, 0
        for c, row in zip(clips, rows):
            t = time.perf_counter()
            hyp = stt.transcribe(c.audio, language="hi" if language == "forced-hi" else "auto")
            lat.append(time.perf_counter() - t)
            hyps.append(hyp)
            langs.append(stt.last_language)
            fallbacks += stt.last_fallback
            limit_hits += stt.last_hit_token_limit
            row[f"hyp_{mode}"] = hyp
            row[f"tokens_{mode}"] = stt.last_tokens
            row[f"window_s_{mode}"] = stt.last_window_s
            row[f"fallback_{mode}"] = stt.last_fallback
        audio_s = sum(c.duration for c in clips)
        results[mode] = {
            **score([c.text for c in clips], hyps),
            "rtf": round(sum(lat) / audio_s, 3),
            "latency_p50_s": round(float(np.percentile(lat, 50)), 3),
            "latency_p95_s": round(float(np.percentile(lat, 95)), 3),
            "detected_english": sum(l == "en" for l in langs),
            "fallbacks_to_30s": fallbacks,
            "hit_token_limit": limit_hits,
        }
        print(mode, results[mode], flush=True)

    from bench import cpu_model  # same CPU string as the latency benchmark

    out = {
        "data": f"FLEURS hi_in test, clips {args.start}-{args.stop - 1} in file-name order",
        "clips": len(clips),
        "audio_minutes": round(sum(c.duration for c in clips) / 60, 2),
        "model": s.stt_model.name,
        "cpu": cpu_model(),
        "threads": args.threads,
        "decoding": f"greedy, no timestamps; generation stops at {TOKEN_LIMIT} tokens",
        "short_window": {"pad_s": pad_s, "short_window_max_s": s.stt_short_window_max_s},
        "normaliser": "NFC, punctuation/symbols and danda removed, lowercase (same as the fine-tuning repo)",
        "results": results,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "rows": rows,
    }
    path = Path(args.out) if args.out else ROOT / "results" / f"wer-{args.start}-{args.stop}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
