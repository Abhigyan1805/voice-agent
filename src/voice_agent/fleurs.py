"""Load a slice of Google FLEURS Hindi (hi_in, test split; CC-BY-4.0) for evaluation, with low memory.

Clips are ordered by audio file name, the same stable order the Whisper fine-tuning repo uses, so
"clips 150-249" means the same audio there and here. Only the selected rows' audio is decoded;
the parquet is streamed in small batches.
"""

from __future__ import annotations

import io
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GLOB = "datasets--google--fleurs/snapshots/*/parquet-data/hi_in/test-*.parquet"


@dataclass
class Clip:
    uid: str
    text: str
    audio: np.ndarray  # 16 kHz mono float32

    @property
    def duration(self) -> float:
        return len(self.audio) / 16_000


def parquet_path() -> Path:
    if os.environ.get("FLEURS_HI_TEST"):
        return Path(os.environ["FLEURS_HI_TEST"])
    from huggingface_hub.constants import HF_HUB_CACHE

    files = sorted(Path(HF_HUB_CACHE).glob(GLOB))
    if not files:
        raise FileNotFoundError(
            "FLEURS hi_in test parquet not found; download it with "
            "`huggingface-cli download google/fleurs --repo-type dataset --include 'parquet-data/hi_in/test-*'` "
            "or set FLEURS_HI_TEST=/path/to/test.parquet"
        )
    return files[0]


def extract(start: int, stop: int, out_dir: Path, path: Path | None = None) -> None:
    """Decode clips [start, stop) to 16 kHz wav files plus an index. Reading the 300 MB single
    row-group parquet costs ~1.5 GB of RAM, so callers run this in a child process."""
    import json

    import pyarrow.parquet as pq
    import soundfile as sf

    from .audio import resample

    path = path or parquet_path()
    table = pq.read_table(path, columns=["audio", "raw_transcription"])
    names = table.column("audio").combine_chunks().field("path").to_pylist()
    order = sorted(range(len(names)), key=lambda i: names[i])[start:stop]
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for i in order:
        row = table.slice(i, 1).to_pylist()[0]
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        uid = names[i].removesuffix(".wav")
        sf.write(str(out_dir / f"{uid}.wav"), resample(audio, sr, 16_000), 16_000, subtype="PCM_16")
        index.append({"uid": uid, "text": row["raw_transcription"].strip()})
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=0), encoding="utf-8")


def load_clips(start: int, stop: int) -> list[Clip]:
    """Clips [start, stop) in file-name order, extracted once (in a subprocess) to ``cache/fleurs/``."""
    import json
    import subprocess
    import sys

    from .audio import load_audio
    from .config import CACHE_DIR

    out_dir = CACHE_DIR / "fleurs" / f"test-{start}-{stop}"
    if not (out_dir / "index.json").exists():
        parquet_path()  # raise a helpful error here rather than in the child
        code = f"from pathlib import Path; from voice_agent.fleurs import extract; extract({start}, {stop}, Path({str(out_dir)!r}))"
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        subprocess.run([sys.executable, "-c", code], check=True, env=env)
    index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    return [Clip(e["uid"], e["text"], load_audio(out_dir / f"{e['uid']}.wav")) for e in index]


_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)


def normalize(text: str) -> str:
    """Scoring normaliser, identical in behaviour to the fine-tuning repo's ``hindi_asr.text.normalize``:
    NFC, drop zero-width characters, punctuation/symbols (incl. danda) -> space, lowercase, collapse spaces.
    (Whisper's BasicTextNormalizer would delete Devanagari matras and under-count errors.)"""
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" or ch in "।॥" else ch for ch in text)
    return " ".join(text.lower().split())
