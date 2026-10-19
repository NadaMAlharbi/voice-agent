"""TTS interface, voices and engine selection.

Voices come from config.yaml (tts.voices); each one is a built-in voice of a model:
  English -> Kokoro-82M  (kokoro_engine.py)
  Arabic  -> XTTS-v2     (xtts_engine.py)
The pipeline calls `synthesize(text, voice)` once per chunk.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Voice:
    name: str               # config key, e.g. "en-american-female"
    language: str           # "en" | "ar"
    gender: str             # "female" | "male"
    engine: str             # "kokoro" | "xtts" | "tone"
    voice_id: str           # the model's own voice name, e.g. "af_heart" or "Ana Florence"


class TTSEngine(Protocol):
    name: str
    sample_rate: int

    def load(self, voices: dict[str, Voice]) -> None: ...

    def synthesize(self, text: str, voice: Voice) -> Iterator[np.ndarray]:
        """Yield float32 mono audio pieces at `sample_rate`. Stopping early must be safe (barge-in)."""
        ...

    def device_info(self) -> str: ...

    # Optional, for barge-in:
    #   cancel()        stop the current synthesis as soon as possible
    #   reset_cancel()  called before each synthesis


def synthesize(engine, text: str, voice: Voice):
    # This function speaks one chunk with the engine (XTTS yields several pieces while it generates).
    return engine.synthesize(text, voice)


def cancel(engine) -> None:
    # This function stops the current speech when the user interrupts.
    fn = getattr(engine, "cancel", None)
    if fn:
        fn()


def reset_cancel(engine) -> None:
    fn = getattr(engine, "reset_cancel", None)
    if fn:
        fn()


def parse_voice_key(key: str) -> tuple[str, str]:
    # This function reads the language and gender from a voice name:
    # "en-american-female" -> ("en", "female"), "ar-male" -> ("ar", "male").
    parts = key.split("-")
    if len(parts) in (2, 3) and parts[0] in ("en", "ar") and parts[-1] in ("female", "male"):
        return parts[0], parts[-1]
    raise ValueError(f"tts.voices: bad voice name {key!r} (use en-american-female or ar-male)")


def all_voices(cfg) -> dict[str, Voice]:
    # This function builds every voice listed in config.yaml.
    voices = {}
    for key, v in cfg.tts.voices.items():
        language, gender = parse_voice_key(key)
        voices[key] = Voice(key, language, gender, v["engine"], str(v["voice"]))
    ids = [(v.engine, v.voice_id) for v in voices.values()]
    if len(set(ids)) != len(ids):
        raise ValueError("tts.voices: two entries use the same model voice; every voice must be different")
    return voices


def load_voices(cfg, quiet: bool = False) -> dict[str, Voice]:
    # This function returns the voices the assistant uses and checks that English and Arabic
    # each have a female and a male voice.
    voices = all_voices(cfg)
    for lang in ("en", "ar"):
        for gender in ("female", "male"):
            if not any(v.language == lang and v.gender == gender for v in voices.values()):
                raise ValueError(f"tts.voices: no {lang} {gender} voice")
    if not quiet:
        log.info("voices: %s", ", ".join(f"{v.name}={v.engine}:{v.voice_id}" for v in voices.values()))
    return voices


def pick_voice(voices: dict[str, Voice], language: str, gender: str) -> Voice:
    # This function picks the voice for a reply language and gender.
    for v in voices.values():
        if v.language == language and v.gender == gender:
            return v
    for v in voices.values():
        if v.language == language:
            return v
    raise KeyError(f"no voice for language {language!r}")


class MultiTTS:
    """Sends each voice to its model (Kokoro for English, XTTS for Arabic)."""
    name = "kokoro+xtts"
    sample_rate = 24000

    def __init__(self, cfg):
        self.cfg = cfg
        self.engines: dict = {}

    def load(self, voices: dict[str, Voice]) -> None:
        # This function loads only the models that the voices need.
        needed = {v.engine for v in voices.values()}
        if "kokoro" in needed:
            from voice_agent.tts.kokoro_engine import KokoroEngine
            self.engines["kokoro"] = KokoroEngine(self.cfg)
        if "xtts" in needed:
            from voice_agent.tts.xtts_engine import XttsEngine
            self.engines["xtts"] = XttsEngine(self.cfg)
        for name, engine in self.engines.items():
            engine.load({k: v for k, v in voices.items() if v.engine == name})
            assert engine.sample_rate == self.sample_rate

    def synthesize(self, text: str, voice: Voice) -> Iterator[np.ndarray]:
        engine = self.engines[voice.engine]
        yield from engine.synthesize(text, voice)
        self.last_stats = getattr(engine, "last_stats", {})

    def is_streaming(self, voice: Voice) -> bool:
        # True when the voice's engine sends audio while it is still generating (XTTS).
        return bool(getattr(self.engines.get(voice.engine), "streaming", False))

    def cancel(self) -> None:
        for e in self.engines.values():
            cancel(e)

    def reset_cancel(self) -> None:
        for e in self.engines.values():
            reset_cancel(e)

    def device_info(self) -> str:
        return " | ".join(e.device_info() for e in self.engines.values())


def make_tts(cfg, name: str | None = None) -> TTSEngine:
    # This function creates the TTS engine named in config.yaml (tts.engine).
    name = name or cfg.tts.engine
    if name == "models":
        return MultiTTS(cfg)
    if name == "tone":
        from voice_agent.tts.tone_engine import ToneEngine

        return ToneEngine()
    raise ValueError(f"unknown TTS engine {name!r} (models | tone)")
