"""Render every user turn of the scripted calls (evals/calls/calls.json) to 16 kHz wav files.

Synthetic callers, exactly as ``scripts/make_prompts.py`` renders the benchmark prompts: Piper
voices that differ from the agent's own (pratham / rohan for Hindi, ryan for English), 0.5 s of
silence before and 1.2 s after, and -60 dBFS noise so silence is not digitally perfect.

    python scripts/make_call_audio.py        # writes evals/calls/audio/<call>-<turn>.wav
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_agent.audio import resample, write_wav  # noqa: E402
from voice_agent.config import MODELS_DIR  # noqa: E402


def main() -> None:
    from piper import PiperVoice

    calls = json.loads((ROOT / "evals" / "calls" / "calls.json").read_text(encoding="utf-8"))["calls"]
    out_dir = ROOT / "evals" / "calls" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    voices = {}
    for call in calls:
        for k, turn in enumerate(call["turns"]):
            if turn["voice"] not in voices:
                voices[turn["voice"]] = PiperVoice.load(str(MODELS_DIR / "piper" / f"{turn['voice']}.onnx"))
            v = voices[turn["voice"]]
            audio = np.concatenate([c.audio_float_array for c in v.synthesize(turn["user"])])
            audio = resample(audio, v.config.sample_rate, 16_000)
            audio = np.concatenate([np.zeros(8_000), audio, np.zeros(19_200)]).astype(np.float32)
            audio += rng.normal(0, 10 ** (-60 / 20), len(audio)).astype(np.float32)
            write_wav(out_dir / f"{call['id']}-{k}.wav", audio, 16_000)
            print(f"{call['id']}-{k}: {len(audio) / 16_000:.2f} s")


if __name__ == "__main__":
    main()
