"""Render the fixed benchmark prompts in prompts/prompts.json to 16 kHz wav files.

These are *synthetic* recordings: each question is spoken by a Piper voice that is different from
the voices the agent answers with (pratham / rohan for Hindi, ryan for English; the agent speaks
with priyamvada / lessac). 0.5 s of silence is added before and 1.2 s after each prompt so the
VAD sees a real start and end of speech. A small amount of noise (-60 dBFS) keeps the silence
from being digitally perfect.

    python scripts/make_prompts.py
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

    prompts = json.loads((ROOT / "prompts" / "prompts.json").read_text(encoding="utf-8"))
    out_dir = ROOT / "prompts" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    voices: dict[str, PiperVoice] = {}
    for p in prompts:
        if p["voice"] not in voices:
            voices[p["voice"]] = PiperVoice.load(str(MODELS_DIR / "piper" / f"{p['voice']}.onnx"))
        v = voices[p["voice"]]
        audio = np.concatenate([c.audio_float_array for c in v.synthesize(p["text"])])
        audio = resample(audio, v.config.sample_rate, 16_000)
        audio = np.concatenate([np.zeros(8_000), audio, np.zeros(19_200)]).astype(np.float32)
        audio += rng.normal(0, 10 ** (-60 / 20), len(audio)).astype(np.float32)
        write_wav(out_dir / f"{p['id']}.wav", audio, 16_000)
        print(f"{p['id']}: {len(audio) / 16_000:.2f} s")


if __name__ == "__main__":
    main()
