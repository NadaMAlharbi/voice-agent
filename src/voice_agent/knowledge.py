"""Offline knowledge vs real-time questions.

The agent runs offline: the LLM knows general facts but nothing live. Each utterance
is classified before the LLM runs:

  general   -> the LLM answers from its own knowledge
  time/date -> answered from the computer's clock (works offline)
  weather, news, prices, sports, traffic -> needs live data:
      - with a live-data provider that returns facts, the facts go into the prompt
      - otherwise (the default) the LLM is told it has no live data: it says so
        briefly, never invents numbers, and gives general help instead

An optional online source can be added later behind `LiveDataProvider`.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

LIVE_KINDS = ("weather", "news", "prices", "sports", "traffic")

# (kind, English pattern, Arabic pattern). Word boundaries are used for English only.
_PATTERNS: list[tuple[str, str, str]] = [
    ("time", r"\bwhat(?:'s| is)? the time\b|\bwhat time is it\b|\bcurrent time\b|\btime now\b",
     r"كم الساعة|الساعة (?:الآن|كم)|ما الوقت|كم الوقت"),
    ("date", r"\bwhat(?:'s| is)? (?:the )?(?:date|day)\b|\btoday'?s date\b|\bwhat day is (?:it|today)\b",
     r"ما التاريخ|ما هو التاريخ|تاريخ اليوم|أي يوم|اي يوم|ما اليوم|كم التاريخ"),
    ("weather", r"\bweather\b|\bforecast\b|\btemperature\b|\b(?:is it|will it) (?:rain|snow|hot|cold)\b|\bhumidity\b",
     r"الطقس|طقس|حالة الجو|الجو اليوم|درجة الحرارة|درجات الحرارة|أمطار|امطار|هل ستمطر|الرطوبة"),
    ("news", r"\bnews\b|\bheadlines?\b|\blatest (?:on|about|updates?)\b|\bwhat happened (?:today|yesterday)\b",
     r"الأخبار|الاخبار|أخبار|اخبار|آخر المستجدات|ماذا حدث اليوم"),
    ("prices", r"\bstock\b|\bshare price\b|\bprice of\b|\bexchange rate\b|\bbitcoin\b|\bcrypto\b|"
               r"\bgold price\b|\boil price\b|\bhow much is (?:the )?(?:dollar|euro|riyal)\b",
     r"سعر|أسعار|اسعار|البورصة|الأسهم|الاسهم|سعر الصرف|البيتكوين|الذهب اليوم"),
    ("sports", r"\bscore\b|\bwho won\b|\bmatch (?:today|tonight|result)\b|\bstandings\b|\bfixtures?\b",
     r"نتيجة المباراة|نتيجة مباراة|من فاز|ترتيب الدوري|مباراة اليوم"),
    ("traffic", r"\btraffic\b|\bflight status\b|\bis my flight\b",
     r"زحمة|الازدحام|حالة المرور|المرور الآن|حالة الرحلة"),
]
_COMPILED = [(k, re.compile(en, re.I), re.compile(ar)) for k, en, ar in _PATTERNS]


@dataclass(frozen=True)
class Intent:
    kind: str            # general | time | date | weather | news | prices | sports | traffic

    @property
    def live(self) -> bool:
        return self.kind in LIVE_KINDS

    @property
    def local(self) -> bool:
        return self.kind in ("time", "date")


def classify(text: str) -> Intent:
    for kind, en, ar in _COMPILED:
        if en.search(text) or ar.search(text):
            return Intent(kind)
    return Intent("general")


# ------------------------------------------------------------------ live data (optional, future)

class LiveDataProvider(Protocol):
    """Optional online source. Returns short factual text, or None if unavailable."""
    kinds: tuple[str, ...]

    async def fetch(self, kind: str, query: str, language: str) -> str | None: ...


class OfflineProvider:
    """Default: no internet access. Never returns data."""
    kinds: tuple[str, ...] = ()

    async def fetch(self, kind: str, query: str, language: str) -> str | None:
        return None


async def fetch_live(provider: LiveDataProvider | None, intent: Intent, query: str, language: str,
                     timeout_s: float = 2.0) -> str | None:
    """Ask the provider, never blocking the voice loop longer than timeout_s."""
    if provider is None or intent.kind not in getattr(provider, "kinds", ()):
        return None
    try:
        return await asyncio.wait_for(provider.fetch(intent.kind, query, language), timeout_s)
    except Exception:  # noqa: BLE001  (timeout, network down): behave as offline
        return None


# ------------------------------------------------------------------ prompt notes

_KIND_NAMES = {
    "weather": ("weather", "الطقس"), "news": ("news", "الأخبار"), "prices": ("live prices", "الأسعار الحالية"),
    "sports": ("live sports results", "نتائج المباريات"), "traffic": ("live traffic or flight status", "حالة المرور"),
}


def clock_note(now: datetime) -> str:
    """Always included: the device clock is local, so date/time questions work offline."""
    return f"Device date and time: {now.strftime('%A %d %B %Y, %H:%M')}."


def turn_note(intent: Intent, live_facts: str | None) -> str:
    """Extra system instruction for this turn, or '' for general questions."""
    if intent.local:
        return "The user asks for the time or date: answer briefly from the device date and time."
    if not intent.live:
        return ""
    name = _KIND_NAMES[intent.kind][0]
    if live_facts:
        return (f"Live {name} data retrieved just now: {live_facts}\n"
                "Answer using only these facts; do not add numbers that are not in them.")
    return (f"This may need live {name}, and you are offline. If the user wants current {name}, say in one "
            f"sentence that you can't check it offline, never invent values, then give typical information and "
            "say it is typical (for weather: the usual climate there in this season). "
            "If the question is general or historical, just answer it.")


def system_prompt(base: str, now: datetime, note: str, language_directive: str) -> str:
    """The one system prompt used by the voice pipeline and by `llm-test`."""
    return "\n\n".join(p for p in (base, clock_note(now), note, language_directive) if p).strip()


def turn_context(now: datetime, note: str, language_directive: str) -> str:
    """The per-turn part of the instructions (changes every turn / minute)."""
    return " ".join(p for p in (clock_note(now), note, language_directive) if p).strip()


def build_messages(base: str, history: list[dict], user_text: str | None, now: datetime, note: str,
                   language_directive: str, cache_friendly: bool) -> list[dict]:
    """System message + history (+ the current user text, if given).

    cache_friendly=True keeps the system message the same every turn, so Ollama can reuse
    its cache of the system prompt and tool list (faster replies). The per-turn context
    (clock, knowledge note, reply language) is added to the current user message instead.
    False puts everything in the system message.
    """
    msgs = list(history)
    if user_text is not None:
        msgs.append({"role": "user", "content": user_text})
    if not cache_friendly:
        return [{"role": "system", "content": system_prompt(base, now, note, language_directive)}] + msgs
    ctx = turn_context(now, note, language_directive)
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            msgs[i] = {**msgs[i], "content": f"[Context for this reply, not said by the user: {ctx}]\n"
                                              f"{msgs[i].get('content', '')}"}
            break
    return [{"role": "system", "content": base.strip()}] + msgs
