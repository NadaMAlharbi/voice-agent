"""Word error rate with Arabic-aware normalization."""

from __future__ import annotations

import re
import unicodedata

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # harakat + tatweel
_ALEF = re.compile(r"[آأإٱ]")  # آ أ إ ٱ -> ا


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _DIACRITICS.sub("", text)
    text = _ALEF.sub("ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")  # ى->ي, ة->ه
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return text.split()


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)
