"""Countdown timers; the assistant announces them when they end."""

from __future__ import annotations

from functools import partial

from voice_agent.tools.base import Tool, ToolResult


def set_timer(tb, seconds: int = 0, minutes: int = 0, label: str = "") -> ToolResult:
    # This function starts a timer through the voice session.
    total = int(seconds) + 60 * int(minutes)
    if total <= 0:
        return ToolResult(False, error="timer length must be positive")
    if tb.scheduler is None:
        return ToolResult(False, error="timers need the voice session")
    tid = tb.scheduler(total, label)
    return ToolResult(True, {"timer_id": tid, "seconds": total, "label": label})


def tools(tb) -> list[Tool]:
    I = {"type": "integer"}
    return [Tool("set_timer", "Start a countdown timer; you will announce it when it ends.",
                 {"seconds": I, "minutes": I, "label": {"type": "string"}}, partial(set_timer, tb))]
