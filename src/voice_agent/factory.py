"""Build the pipeline from settings. Models load once; total resident memory stays well under 1 GB."""

from __future__ import annotations

from .agent import Agent
from .config import Settings, load_env_file
from .llm import make_client
from .pipeline import VoicePipeline


def build_pipeline(settings: Settings | None = None, need_stt: bool = True, filler: bool = False, cache=None):
    from .stt import WhisperSTT
    from .tts import PiperTTS

    load_env_file()
    s = settings or Settings()
    stt = WhisperSTT(s.stt_model, threads=s.stt_threads) if need_stt else None
    tts = PiperTTS(s.tts_hindi, s.tts_english)
    client = make_client(s.llm_provider, s.llm_model, cache=s.llm_cache if cache is None else cache)
    return VoicePipeline(stt, Agent(client), tts, filler=filler)
