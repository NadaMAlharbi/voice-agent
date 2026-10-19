"""Gap-free TTS: the next chunk must be ready before the queued audio ends.

While one chunk plays, the next is synthesized. If that takes longer than the audio
still queued, the listener hears silence. So before each synthesis, `plan_chunk`
checks whether the chunk fits. If not, it speaks only the part that fits (at a phrase
boundary) and keeps the rest for the next round.

The cost model learns this machine's TTS speed from every synthesis and adds a safety
margin based on its own past under-estimates (`over`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from voice_agent.chunker import phrase_break

MIN_TAIL_WORDS = 2       # the rest of a split chunk: shorter is a poor TTS utterance
MARGIN_S = 0.15          # keep this much queued audio unused (thread hops, jitter)


@dataclass
class TTSCostModel:
    """synth_seconds(words) ~= (fixed_s + words * s_per_word[lang] * rtf) * (1 + over)  (EMAs)."""
    s_per_word: dict = field(default_factory=lambda: {"en": 0.35, "ar": 0.5})
    rtf: float = 0.7          # synthesis seconds per audio second (autoregressive part)
    fixed_s: float = 0.4      # per-call overhead (model start-up per chunk)
    over: float = 0.1         # observed relative under-estimate (pessimism); starts at 10 %
    alpha: float = 0.3

    def _ema(self, old: float, new: float) -> float:
        return (1 - self.alpha) * old + self.alpha * new

    def observe(self, lang: str, words: int, audio_s: float, synth_s: float) -> None:
        if words <= 0 or audio_s <= 0:
            return
        predicted = self.estimate(lang, words, safe=False)
        if predicted > 0 and synth_s > 0:
            miss = synth_s / predicted - 1.0
            # being late is remembered fast, being early is forgotten slowly (5 % to 60 %)
            self.over = min(0.6, max(0.05, 0.5 * self.over + 0.5 * miss if miss > self.over
                                     else self._ema(self.over, max(miss, 0.0))))
        self.s_per_word[lang] = self._ema(self.s_per_word.get(lang, 0.4), audio_s / words)
        self.fixed_s = self._ema(self.fixed_s, 0.0)
        self.rtf = self._ema(self.rtf, synth_s / audio_s)

    def estimate(self, lang: str, words: int, safe: bool = True) -> float:
        est = self.fixed_s + words * self.s_per_word.get(lang, 0.4) * self.rtf
        return est * (1 + self.over) if safe else est

    def min_words(self, lang: str, min_piece_s: float = 1.4) -> int:
        """Smallest piece worth a TTS call (~1.4 s of audio): 4 English words, 3 Arabic words."""
        return max(3, min(5, math.ceil(min_piece_s / max(self.s_per_word.get(lang, 0.4), 0.1))))

    def max_words(self, lang: str, budget_s: float) -> int:
        per = self.s_per_word.get(lang, 0.4) * self.rtf * (1 + self.over)
        return int((budget_s - self.fixed_s * (1 + self.over)) / per) if per > 0 else 10**6


def split_to_budget(text: str, lang: str, model: TTSCostModel, budget_s: float,
                    min_words: int | None = None) -> tuple[str, str | None]:
    """(part to speak now, rest or None). Splits whenever the chunk does not fit the budget and
    both parts stay usable (head >= min_words, rest >= MIN_TAIL_WORDS), at a phrase boundary."""
    min_words = min_words or model.min_words(lang)
    words = text.split()
    if len(words) < min_words + MIN_TAIL_WORDS or model.estimate(lang, len(words)) <= budget_s:
        return text, None
    hi = max(min_words, min(model.max_words(lang, budget_s), len(words) - MIN_TAIL_WORDS))
    k = phrase_break(words, min_words, hi)
    return " ".join(words[:k]), " ".join(words[k:])


@dataclass
class ChunkPlan:
    text: str
    carry: str | None
    ahead_s: float                # audio still queued when synthesis starts


def plan_chunk(chunk: str, lang: str, cost: TTSCostModel, ahead_s: float) -> ChunkPlan:
    # This function keeps only the part of a chunk that can be ready before the queued audio ends.
    # Same decision for the server (ResponseProcessor) and latency-bench.
    carry = None
    if ahead_s > 0.05:
        chunk, carry = split_to_budget(chunk, lang, cost, max(ahead_s - MARGIN_S, 0.0))
    else:
        ahead_s = 0.0
    return ChunkPlan(chunk, carry, ahead_s)
