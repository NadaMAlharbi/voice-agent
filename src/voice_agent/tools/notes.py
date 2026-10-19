"""Notes saved in data/notes.md."""

from __future__ import annotations

from datetime import datetime
from functools import partial

from voice_agent.tools.base import Tool, ToolResult


def add_note(tb, text: str) -> ToolResult:
    # This function adds a note with the date and time.
    tb.notes_path.parent.mkdir(parents=True, exist_ok=True)
    with tb.notes_path.open("a", encoding="utf-8") as f:
        f.write(f"- {datetime.now():%Y-%m-%d %H:%M}  {text.strip()}\n")
    return ToolResult(True, {"saved_note": text.strip(), "file": str(tb.notes_path)})


def read_notes(tb, count: int = 5) -> ToolResult:
    # This function returns the latest notes.
    try:
        lines = [ln[2:] for ln in tb.notes_path.read_text(encoding="utf-8").splitlines() if ln.startswith("- ")]
    except OSError:
        lines = []
    return ToolResult(True, {"notes": lines[-max(1, count):], "total": len(lines)})


def tools(tb) -> list[Tool]:
    return [
        Tool("add_note", "Save a note for the user.", {"text": {"type": "string"}}, partial(add_note, tb), ["text"]),
        Tool("read_notes", "Read the user's latest notes.", {"count": {"type": "integer"}}, partial(read_notes, tb)),
    ]
