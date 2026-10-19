"""Remember facts about the user (data/memory.json); they go into the system prompt."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from functools import partial
from pathlib import Path

from voice_agent.tools.apps import _ratio, normalize
from voice_agent.tools.base import Tool, ToolResult


class MemoryStore:
    """Facts the user asked to remember ("remember that my car is a white Camry"). JSON, local."""

    def __init__(self, path: Path, limit: int = 50):
        self.path, self.limit = Path(path), limit
        self._lock = threading.Lock()

    def facts(self) -> list[str]:
        try:
            return [f["fact"] for f in json.loads(self.path.read_text(encoding="utf-8"))]
        except (OSError, ValueError, KeyError, TypeError):
            return []

    def _save(self, facts: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [{"fact": f, "saved": datetime.now().isoformat(timespec="seconds")} for f in facts[-self.limit:]]
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def add(self, fact: str) -> bool:
        with self._lock:
            facts = self.facts()
            if any(_ratio(normalize(fact), normalize(f)) > 0.9 for f in facts):
                return False
            self._save(facts + [fact.strip()])
            return True

    def remove(self, about: str) -> str | None:
        with self._lock:
            facts = self.facts()
            n = normalize(about)
            scored = [(max(_ratio(n, normalize(f)), 0.8 if n and n in normalize(f) else 0), f) for f in facts]
            if not scored:
                return None
            score, best = max(scored)
            if score < 0.6:
                return None
            facts.remove(best)
            self._save(facts)
            return best


def remember(tb, fact: str) -> ToolResult:
    # This function saves a fact about the user.
    new = tb.memory.add(fact)
    return ToolResult(True, {"remembered": fact, "already_known": not new})


def forget(tb, about: str) -> ToolResult:
    # This function deletes a remembered fact.
    gone = tb.memory.remove(about)
    return ToolResult(gone is not None, {"forgot": gone} if gone else {},
                      error="" if gone else f"nothing remembered about {about!r}")


def tools(tb) -> list[Tool]:
    S = {"type": "string"}
    return [
        Tool("remember", "Remember a fact about the user for future conversations.", {"fact": S},
             partial(remember, tb), ["fact"]),
        Tool("forget", "Forget a remembered fact.", {"about": S}, partial(forget, tb), ["about"]),
    ]
