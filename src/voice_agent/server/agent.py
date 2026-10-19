"""Per-connection agent state: tools, router, pending confirmation, sleep mode.

ResponseProcessor asks the Agent what to do with each finished utterance:

  asleep          -> ignore unless the wake word is heard
  "go to sleep"   -> sleep
  pending confirm -> yes runs the action, no cancels, anything else drops it
  router match    -> run the tool now (no LLM) and speak a template
  otherwise       -> LLM with tools (agent loop in response.py)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from voice_agent.tools.apps import normalize
from voice_agent.tools.base import ToolRegistry
from voice_agent.tools.toolbox import Toolbox
from voice_agent.tools.website_open import site_url
from voice_agent.tools.router import Routed, route

TOOLS_PROMPT = """
You control this Windows PC with the tools provided.
- To DO something (open, close, type, search, timers, files), call the tool. Never say it is done \
unless the tool result has "ok": true.
- If "ok" is false, say what failed in one short sentence.
- Risky actions (closing apps, shutdown, writing or deleting files): call the tool anyway; the system asks \
the user to confirm.
- Questions you can answer from knowledge: answer, no tools.
- Copy app names, search words and text to type as the user said them. Keep Arabic text in Arabic.
- If the request is unclear (which app, what text), ask one short question.
- After a tool, reply with one short sentence. Summarize results; do not read raw data or paths aloud.
Hints: time or date -> get_date_time; "remind me in..." -> set_timer; "remember that..." -> remember; \
"what's on my screen" -> describe_screen; any calculation ("how much is 25 times 4", "احسب") -> calculate, \
then say the result it returns (never work it out yourself).""".strip()

COMMAND_START = re.compile(   # clear action verbs only, so questions are never delayed
    r"^(?:(?:please|can you|could you|HAZEM)\s+)*(?:open|close|quit|launch|search|google|look up|turn|increase|"
    r"decrease|raise|lower|mute|unmute|remind|remember|save|note|type|press|click|take|switch|minimi[sz]e|"
    r"maximi[sz]e|pause|lock|shut|restart|scroll|move|copy|paste)\b|"
    r"^(?:لو سمحت\s+|من فضلك\s+|ممكن\s+)?(?:افتح|اغلق|اقفل|سكر|ابحث|دور|شغل|ارفع|اخفض|وطي|نزل|اكتم|اضبط|ذكرني|"
    r"تذكر|احفظ|سجل|اطبع|اضغط|انقر|خذ|صور|انتقل|حول|صغر|كبر|اوقف)", re.I)

RETRY_NOTE = ("[System: the user asked for an action but you did not call a tool. Call the matching tool now. "
              "If no tool fits, say in one short sentence that you cannot do it.]")


def retry_note(lang: str) -> str:
    # The retry is a user message, so it must repeat the reply language.
    from voice_agent.language import reminder

    return f"{RETRY_NOTE[:-1]} {reminder(lang)}]"
_CLAIM = re.compile(r"\b(?:i'll|i will|i've|i have|i'm|let me|opening|closing|turning|setting|saving|searching|"
                    r"launching|starting|lowering|raising|muting|taking|done)\b|سأ|سوف|تم |تم$|جار|حسنا|حاضر", re.I)


# Calculations go to the Windows Calculator (tools/calculator.py), never answered from memory.
CALCULATION = re.compile(
    r"\b(?:calculate|compute|calculator)\b|\bhow much is\b.*\d|"
    r"\d\s*(?:[+*x\u00d7\u00f7]|plus|minus|times|divided by|multiplied by)\s*\d|\d\s+[-/]\s+\d|"
    r"احسب|حاسبة|كم يساوي|كم تساوي|كم حاصل|\d\s*(?:زائد|ناقص|ضرب|قسمة)\s*\d", re.I)


def is_calculation(text: str) -> bool:
    return bool(CALCULATION.search(text))


def is_command(text: str) -> bool:
    # This function checks if the user asked to do something (not a question to answer).
    return bool(COMMAND_START.match(normalize(text))) or is_calculation(text)


def needs_tool_retry(user_text: str, calls: list, reply: str) -> bool:
    # This function catches "I'll do it" replies to a command when no tool was called,
    # and calculations answered without the calculator.
    if calls or not is_command(user_text):
        return False
    return is_calculation(user_text) or bool(_CLAIM.search(reply or ""))


@dataclass
class Pending:
    tool: str
    args: dict
    language: str
    created: float = field(default_factory=time.monotonic)

    def expired(self, ttl_s: float = 45.0) -> bool:
        return time.monotonic() - self.created > ttl_s


class Agent:
    def __init__(self, toolbox: Toolbox | None, use_router: bool = True, max_steps: int = 4):
        self.toolbox = toolbox
        self.use_router = use_router
        self.max_steps = max_steps
        self.pending: Pending | None = None
        self.asleep = False

    @property
    def registry(self) -> ToolRegistry | None:
        return self.toolbox.registry if self.toolbox else None

    def schemas(self) -> list[dict] | None:
        return self.registry.schemas() if self.registry else None

    def system_note(self) -> str:
        if not self.toolbox:
            return ""
        note = TOOLS_PROMPT
        if self.registry.get("list_files") is not None:
            note += ("\nFiles: only in the user's file folder. If you are not sure of a file name, "
                     "call list_files first.")
        facts = self.toolbox.memory.facts()
        if facts:
            note += "\nThings the user asked you to remember:\n" + "\n".join(f"- {f}" for f in facts[-20:])
        return note

    def take_pending(self) -> Pending | None:
        p, self.pending = self.pending, None
        return None if p is None or p.expired() else p

    def route(self, text: str, lang: str) -> Routed | None:
        """Router match whose target exists; otherwise None so the LLM handles it."""
        if not (self.toolbox and self.use_router):
            return None
        r = route(text, lang)
        if r is None:
            return None
        tb = self.toolbox
        name = r.args.get("name", "")
        if r.tool == "open_app" and tb.resolver().resolve(name) is None and site_url(name) is None:
            return None
        if r.tool in ("close_app", "window_action") and not tb.find_windows(name):
            # a known app that is not open: run the tool anyway, it says honestly "I don't see X open"
            return r if tb.resolver().resolve(name) is not None else None
        if r.tool == "focus_window" and not tb.find_windows(name):
            return Routed("open_url", {"url": site_url(name)}) if site_url(name) else None
        return r
