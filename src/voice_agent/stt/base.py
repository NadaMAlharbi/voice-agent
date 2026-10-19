"""Common STT interface. Every engine turns 16 kHz float32 audio into a Transcript."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

SR = 16000
LANGS = ("ar", "en")


@dataclass
class Transcript:
    text: str
    language: str          # "ar" | "en"
    ms: float              # inference time
    audio_s: float         # audio duration
    info: dict = field(default_factory=dict)  # diagnostics: speech_s, lang probs, skip reason...

    @property
    def rtf(self) -> float:
        """Real-time factor: < 1 means faster than real time."""
        return (self.ms / 1000) / max(self.audio_s, 1e-6)


class STTEngine(Protocol):
    name: str

    def load(self) -> None: ...
    def transcribe(self, audio: np.ndarray, language: str = "auto") -> Transcript: ...
    def device_info(self) -> str: ...


def warmup(engine: STTEngine) -> float:
    """Run the model once so the first real call is fast. Returns ms.

    Engines that skip silence provide their own warmup() that runs the model directly.
    """
    t0 = time.perf_counter()
    fn = getattr(engine, "warmup", None)
    if fn:
        fn()
    else:
        engine.transcribe(np.zeros(SR, np.float32), language="en")
    return (time.perf_counter() - t0) * 1000


def speech_only(audio: np.ndarray, threshold: float = 0.5, min_speech_ms: int = 250,
                pad_ms: int = 200) -> tuple[np.ndarray, float]:
    """Keep only the voiced parts of a segment (Silero VAD from faster-whisper).

    Whisper invents text on silent or noisy edges (e.g. "Thank you.", "اشتركوا في القناة"),
    so those are cut off. Voiced parts shorter than min_speech_ms (clicks, coughs) are
    dropped. Returns (speech audio, seconds of speech); empty if there is no speech.
    """
    from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

    ts = get_speech_timestamps(audio, VadOptions(threshold=threshold, min_speech_duration_ms=min_speech_ms,
                                                 min_silence_duration_ms=300, speech_pad_ms=pad_ms))
    if not ts:
        return np.zeros(0, np.float32), 0.0
    chunks, _ = collect_chunks(audio, ts)
    speech = np.concatenate(chunks).astype(np.float32)
    return speech, len(speech) / SR


def make_engine(name: str, cfg) -> STTEngine:
    """Factory. cfg is the loaded Config."""
    if name == "whisper":
        from voice_agent.stt.whisper_engine import WhisperEngine

        return WhisperEngine(cfg)
    raise ValueError(f"unknown STT engine {name!r} (whisper)")
