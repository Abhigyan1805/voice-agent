"""Paths and settings. Everything can be overridden with environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("VOICE_AGENT_MODELS", ROOT / "models"))
CACHE_DIR = Path(os.environ.get("VOICE_AGENT_CACHE", ROOT / "cache"))

STT_SAMPLE_RATE = 16_000


def load_env_file(path: str | os.PathLike | None = None) -> None:
    """Read KEY=VALUE lines into os.environ without overriding variables already set.

    The file is named by ``VOICE_AGENT_ENV_FILE`` (or ``.env`` in the repo root). Values are
    never printed or logged.
    """
    path = Path(path or os.environ.get("VOICE_AGENT_ENV_FILE", ROOT / ".env"))
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass
class Settings:
    stt_model: Path = field(
        default_factory=lambda: Path(os.environ.get("VOICE_AGENT_STT", MODELS_DIR / "whisper-small-hindi-ct2-int8"))
    )
    stt_threads: int = int(os.environ.get("VOICE_AGENT_STT_THREADS", "4"))
    tts_hindi: Path = field(
        default_factory=lambda: Path(
            os.environ.get("VOICE_AGENT_TTS_HI", MODELS_DIR / "piper" / "hi_IN-priyamvada-medium.onnx")
        )
    )
    tts_english: Path = field(
        default_factory=lambda: Path(
            os.environ.get("VOICE_AGENT_TTS_EN", MODELS_DIR / "piper" / "en_US-lessac-medium.onnx")
        )
    )
    # "gemini", "openai" (any OpenAI-compatible endpoint, e.g. DeepSeek) or "offline".
    llm_provider: str = os.environ.get("VOICE_AGENT_LLM", "auto")
    llm_model: str | None = os.environ.get("VOICE_AGENT_LLM_MODEL") or None
    llm_cache: bool = os.environ.get("VOICE_AGENT_LLM_CACHE", "1") != "0"
