"""Speed and a rough intelligibility check for the Piper voices.

Intelligibility is measured by a round trip: synthesise fixed Hindi sentences with each voice,
transcribe them with the agent's own speech recogniser, and score the character error rate
against the input text. It is a proxy (it inherits the recogniser's errors), but it is objective
and it ranks the voices on the same footing. Also reports synthesis real-time factor.

    python scripts/eval_tts.py        # writes results/tts.json
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_agent.audio import resample  # noqa: E402
from voice_agent.config import MODELS_DIR, Settings  # noqa: E402
from voice_agent.fleurs import normalize  # noqa: E402
from voice_agent.stt import WhisperSTT  # noqa: E402

HINDI = [
    "आज मौसम बहुत अच्छा है।",
    "तेईस गुणा सैंतालीस का जवाब एक हज़ार इक्यासी है।",
    "साइबर धोखाधड़ी होने पर तुरंत उन्नीस सौ तीस पर कॉल करें।",
    "भारत की राजधानी नई दिल्ली है।",
    "रात को सोने से एक घंटा पहले फ़ोन बंद कर देना चाहिए।",
    "दिवाली रोशनी का त्योहार है और इसे पूरे देश में मनाया जाता है।",
    "गर्मियों में खूब पानी पिएँ और धूप से बचें।",
    "मैं आपकी हिंदी और अंग्रेज़ी दोनों में मदद कर सकती हूँ।",
    "ताजमहल को शाहजहाँ ने बनवाया था।",
    "अभी सुबह के नौ बजकर बीस मिनट हुए हैं।",
]
ENGLISH = [
    "The answer is one thousand and eighty one.",
    "Today is Tuesday, the tenth of November.",
    "I can help you in Hindi and in English.",
    "Drink plenty of water and stay out of the sun.",
    "Call one nine three zero to report online fraud.",
]


def main() -> None:
    import jiwer
    from piper import PiperVoice

    stt = WhisperSTT(Settings().stt_model)
    voices = {
        "hi_IN-priyamvada-medium": ("hi", HINDI),
        "hi_IN-pratham-medium": ("hi", HINDI),
        "hi_IN-rohan-medium": ("hi", HINDI),
        "en_US-lessac-medium": ("en", ENGLISH),
    }
    out = {}
    for name, (lang, sentences) in voices.items():
        path = MODELS_DIR / "piper" / f"{name}.onnx"
        if not path.exists():
            continue
        v = PiperVoice.load(str(path))
        v.synthesize(sentences[0])  # warm-up
        synth_s, audio_s, hyps = 0.0, 0.0, []
        first = []
        for text in sentences:
            t = time.perf_counter()
            chunks = [c.audio_float_array for c in v.synthesize(text)]
            dt = time.perf_counter() - t
            import numpy as np

            audio = np.concatenate(chunks)
            synth_s += dt
            first.append(dt)
            audio_s += len(audio) / v.config.sample_rate
            hyps.append(stt.transcribe(resample(audio, v.config.sample_rate, 16_000), language=lang))
        refs = [normalize(s) for s in sentences]
        out[name] = {
            "sentences": len(sentences),
            "rtf": round(synth_s / audio_s, 3),
            "mean_synth_s_per_sentence": round(synth_s / len(sentences), 3),
            "roundtrip_cer": round(100 * jiwer.cer(refs, [normalize(h) for h in hyps]), 2),
            "roundtrip_wer": round(100 * jiwer.wer(refs, [normalize(h) for h in hyps]), 2),
            "examples": [{"text": s, "heard": h} for s, h in list(zip(sentences, hyps))[:3]],
        }
        print(name, {k: v for k, v in out[name].items() if k != "examples"}, flush=True)
        del v
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    (ROOT / "results" / "tts.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
