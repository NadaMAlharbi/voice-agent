"""Arabic/English only: language detection from text, LLM directive, output guard.

Used by the response pipeline (server/response.py) and `llm-test --lang-check`.
"""

from __future__ import annotations

import re
import unicodedata

LANGS = ("ar", "en")
_THINK = re.compile(r"<think>.*?</think>", re.S)


def arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("؀" <= c <= "ۿ" or "ݐ" <= c <= "ݿ" for c in letters) / len(letters) if letters else 0.0


def _is_arabic(c: str) -> bool:
    return "\u0600" <= c <= "\u06ff" or "\u0750" <= c <= "\u077f"


def text_language(text: str, default: str = "en", hint: str | None = None) -> str:
    """Language of a user utterance: "ar" or "en".

    Rules, in order:
      - no Arabic letters: English (or `default` when there are no letters at all)
      - at least 30% Arabic letters: Arabic
      - the first word is Arabic ("افتح Google Chrome"): Arabic
      - otherwise the STT language (`hint`) decides, then English
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return hint if hint in LANGS else default
    if not any(_is_arabic(c) for c in letters):
        return "en"
    if arabic_ratio(text) >= 0.3:
        return "ar"
    first = next((w for w in text.split() if any(c.isalpha() for c in w)), "")
    if any(_is_arabic(c) for c in first):
        return "ar"
    return "ar" if hint == "ar" else "en"


def directive(lang: str) -> str:
    """The reply-language rule, added to every turn."""
    if lang == "ar":
        return ("The user speaks Arabic. Reply ONLY in Arabic (simple Modern Standard Arabic, Arabic script), "
                "even if names, tool results or earlier messages are in English. Keep app and site names "
                "as they are. Never use Chinese or English sentences. أجب بالعربية.")
    return ("The user speaks English. Reply ONLY in English. "
            "Never use Chinese, Arabic or any other language.")


def reminder(lang: str) -> str:
    """A very short language note, added after tool results."""
    return "Reply in Arabic. أجب بالعربية." if lang == "ar" else "Reply in English."


def _allowed(c: str) -> bool:
    """Characters that may reach TTS/UI: Latin, Arabic, digits, punctuation, whitespace, symbols."""
    o = ord(c)
    if c.isspace() or o < 0x250:                       # ASCII + Latin-1 + Latin Extended-A/B
        return True
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
        return True                                    # Arabic blocks + presentation forms
    if 0x2000 <= o <= 0x206F or 0x20A0 <= o <= 0x20CF:  # general punctuation, currency
        return True
    if 0x2190 <= o <= 0x2BFF or 0x1F000 <= o <= 0x1FAFF or 0xFE00 <= o <= 0xFE0F or o == 0x200D:
        return False                                   # emoji, arrows, dingbats: TTS would read them oddly
    cat = unicodedata.category(c)
    return cat.startswith(("P", "N", "S")) and not (0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF)


class LanguageGuard:
    """Streaming filter for LLM output.

    Qwen sometimes drifts into Chinese mid-reply. This guard removes any other script
    before it reaches TTS or the UI and counts what it removed (shown in the logs).
    It also drops <think>...</think> blocks.
    """

    def __init__(self):
        self.removed = 0
        self.kept = 0
        self._pending = ""   # holds text while inside a possible <think> block

    def feed(self, delta: str) -> str:
        text = self._pending + delta
        if "<think>" in text and "</think>" not in text[text.index("<think>"):]:
            self._pending = text            # inside a think block: wait for its end
            return ""
        text = _THINK.sub("", text)
        self._pending = ""
        for k in range(min(len(text), 6), 0, -1):   # hold back a possible "<think" prefix
            if "<think>".startswith(text[-k:]):
                self._pending, text = text[-k:], text[:-k]
                break
        out = []
        for c in text:
            if _allowed(c):
                out.append(c)
            else:
                self.removed += 1
        self.kept += len(out)
        return "".join(out)

    def flush(self) -> str:
        rest, self._pending = self._pending, ""
        if "<think>" in rest:
            return ""
        out = "".join(c for c in rest if _allowed(c))
        self.kept += len(out)
        return out

    @property
    def drift_ratio(self) -> float:
        total = self.removed + self.kept
        return self.removed / total if total else 0.0
