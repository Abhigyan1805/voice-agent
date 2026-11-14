"""End-of-turn sweep: how short can the VAD's silence hangover be before it cuts speakers off?

VAD only (Silero v6 + the agent's segmenter), no STT, no LLM. Every clip is padded with 0.5 s of
silence before and 1.5 s after, as a turn would arrive from a microphone. For each
``end_silence_ms`` the segmenter is replayed over the same frame probabilities and reports:

  endpoint delay       silence the VAD waits for before it declares the turn over (fixed per setting)
  premature cut        the clip produced more than one turn: the VAD ended the turn at a pause
                       inside the sentence, before the speaker had finished
  no speech            the VAD never started a turn
  wasted speculations  SpeechPause events (speculative STT starts) that the speaker talked through

Clips: the 20 benchmark question prompts (synthetic) and FLEURS hi_in test clips 150-249 (real
read speech, longer sentences with natural pauses).

    python scripts/eval_endpoint.py          # writes results/endpoint-sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_agent.audio import load_audio  # noqa: E402
from voice_agent.vad import FRAME, FRAME_MS, SileroScorer, SpeechEnd, SpeechPause, VadConfig, VadSegmenter  # noqa: E402

SETTINGS_MS = (256, 320, 384, 448, 544, 704)
PAUSE_MS = (96, 160)
FLEURS_BOUND_PP = 2.0  # allowed rise in FLEURS premature-cut rate over the 544 ms baseline, percentage points


class Replay:
    """A frame scorer that replays precomputed Silero probabilities."""

    def __init__(self, probs):
        self.probs, self.i = probs, 0

    def __call__(self, frame):
        p = self.probs[self.i]
        self.i += 1
        return p

    def reset(self):
        self.i = 0


def pad(a: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(8000, np.float32), a, np.zeros(24_000, np.float32)])


def run(probs, n_samples, end_ms, pause_ms):
    seg = VadSegmenter(Replay(probs), VadConfig(end_silence_ms=end_ms, pause_ms=pause_ms))
    ends, pauses = [], []
    dummy = np.zeros(FRAME, np.float32)
    for _ in range(n_samples // FRAME):
        for ev in seg._frame(dummy):
            (ends if isinstance(ev, SpeechEnd) else pauses if isinstance(ev, SpeechPause) else []).append(ev)
    ends += [e for e in seg.flush() if isinstance(e, SpeechEnd)]
    used = {e.pause_id for e in ends if e.pause_id is not None}
    return len(ends), sum(p.pause_id not in used for p in pauses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "endpoint-sweep.json"))
    args = ap.parse_args()
    from voice_agent.fleurs import load_clips

    clips = []
    for p in json.loads((ROOT / "prompts" / "prompts.json").read_text(encoding="utf-8")):
        clips.append(("questions", p["id"], pad(load_audio(ROOT / "prompts" / "audio" / f"{p['id']}.wav"))))
    for c in load_clips(150, 250):
        clips.append(("fleurs", c.uid, pad(c.audio)))

    scorer = SileroScorer()
    probs = []
    for _, _, a in clips:
        scorer.reset()
        probs.append([scorer(a[i : i + FRAME]) for i in range(0, len(a) - FRAME + 1, FRAME)])

    sweep = []
    for end_ms in SETTINGS_MS:
        for pause_ms in PAUSE_MS:
            if pause_ms >= end_ms:
                continue
            row = {"end_silence_ms": end_ms, "pause_ms": pause_ms,
                   "endpoint_delay_ms": round(max(1, round(end_ms / FRAME_MS)) * FRAME_MS, 1)}
            for group in ("questions", "fleurs"):
                idx = [i for i, c in enumerate(clips) if c[0] == group]
                res = [run(probs[i], len(clips[i][2]), end_ms, pause_ms) for i in idx]
                cut = [clips[i][1] for i, (n, _) in zip(idx, res) if n > 1]
                row[group] = {
                    "clips": len(idx),
                    "premature_cut": len(cut),
                    "premature_cut_pct": round(100 * len(cut) / len(idx), 1),
                    "no_speech": sum(n == 0 for n, _ in res),
                    "wasted_speculations_per_clip": round(sum(w for _, w in res) / len(idx), 2),
                    "cut_ids": cut,
                }
            sweep.append(row)
            print(end_ms, pause_ms, row["questions"]["premature_cut"], row["fleurs"]["premature_cut"],
                  row["fleurs"]["wasted_speculations_per_clip"], flush=True)

    base = next(r for r in sweep if r["end_silence_ms"] == 544 and r["pause_ms"] == 160)
    ok = [r for r in sweep if r["pause_ms"] == 160 and r["questions"]["premature_cut"] == 0
          and r["fleurs"]["premature_cut_pct"] <= base["fleurs"]["premature_cut_pct"] + FLEURS_BOUND_PP]
    chosen = min(ok, key=lambda r: r["end_silence_ms"]) if ok else base
    out = {
        "data": "20 benchmark prompts (Piper) + FLEURS hi_in test clips 150-249, each padded 0.5 s before / 1.5 s after",
        "vad": "Silero v6, start 0.5 / end 0.35 thresholds, min speech 192 ms",
        "rule": f"smallest end_silence_ms with 0 premature cuts on the questions and a FLEURS premature-cut rate at most "
                f"{FLEURS_BOUND_PP} percentage points above the 544 ms baseline",
        "chosen_end_silence_ms": chosen["end_silence_ms"],
        "sweep": sweep,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("chosen", chosen["end_silence_ms"])


if __name__ == "__main__":
    main()
