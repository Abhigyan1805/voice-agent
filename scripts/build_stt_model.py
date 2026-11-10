"""Build the int8 CTranslate2 speech model: whisper-small + the Hindi LoRA adapter, merged, quantised.

This is the same recipe as ``python -m hindi_asr.quant convert`` in the fine-tuning repo
(github.com/Abhigyan1805/hindi-voice-asr), which produced the model used for every number in this
repo. It needs the heavier conversion stack once, in a separate environment if you like:

    pip install torch transformers peft ctranslate2
    python scripts/build_stt_model.py            # -> models/whisper-small-hindi-ct2-int8/

Peak RAM is about 2 GB (the fp32 model is loaded once to merge the adapter).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "openai/whisper-small"
ADAPTER = "Dratinia/whisper-small-hindi-lora"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--out", default=str(ROOT / "models" / "whisper-small-hindi-ct2-int8"))
    args = ap.parse_args()

    import torch
    from ctranslate2.converters import TransformersConverter
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(BASE, language="hindi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        processor.save_pretrained(tmp)
        del model
        out = Path(args.out)
        shutil.rmtree(out, ignore_errors=True)
        TransformersConverter(tmp, copy_files=["tokenizer.json"]).convert(str(out), quantization="int8")
        # faster-whisper reads the feature-extractor settings from preprocessor_config.json
        processor.feature_extractor.save_pretrained(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
