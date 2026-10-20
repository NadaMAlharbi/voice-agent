"""Tone TTS: a short beep per word. No model, no GPU. For dry runs (tts.engine: tone).

`delay_s` adds a delay per word, to test latency and cancellation.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import numpy as np

from voice_agent.tts.base import Voice


class ToneEngine:
    name = "tone"
    sample_rate = 24000

    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s
        self._cancel = threading.Event()
        self.cancelled = 0              # times a synthesis stopped early

    def load(self, voices: dict[str, Voice]) -> None:
        pass

    def device_info(self) -> str:
        return "tone generator (cpu)"

    def cancel(self) -> None:
        self._cancel.set()

    def reset_cancel(self) -> None:
        self._cancel.clear()

    def synthesize(self, text: str, voice: Voice) -> Iterator[np.ndarray]:
        freq = 440.0 if voice.gender == "female" else 220.0
        t = np.arange(int(0.12 * self.sample_rate)) / self.sample_rate
        beep = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        gap = np.zeros(int(0.05 * self.sample_rate), np.float32)
        for _ in text.split():
            if self.delay_s:
                time.sleep(self.delay_s)
            if self._cancel.is_set():
                self.cancelled += 1
                return
            yield np.concatenate([beep, gap])  # one piece per word, like streaming TTS
