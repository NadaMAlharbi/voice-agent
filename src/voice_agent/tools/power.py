"""Lock the computer, shut down or restart (always asks first)."""

from __future__ import annotations

from functools import partial

from voice_agent.tools.base import CONFIRM, SAFE, Tool, ToolResult


def lock_screen(tb) -> ToolResult:
    tb.desktop.lock()
    return ToolResult(True, {"locked": True})


def power(tb, action: str) -> ToolResult:
    # This function shuts down or restarts after 30 s, or cancels a pending shutdown.
    if not tb.allow_power and action != "cancel":
        return ToolResult(False, error="power actions are disabled in config")
    tb.desktop.power(action)
    return ToolResult(True, {"power": action, "delay_s": 0 if action == "cancel" else 30})


def power_risk(args: dict) -> str:
    return SAFE if args.get("action") == "cancel" else CONFIRM


def tools(tb) -> list[Tool]:
    return [
        Tool("lock_screen", "Lock the computer.", {}, partial(lock_screen, tb)),
        Tool("power", "Shut down or restart the computer (30 s delay), or cancel a pending shutdown.",
             {"action": {"type": "string", "enum": ["shutdown", "restart", "cancel"]}}, partial(power, tb),
             ["action"], risk=power_risk),
    ]
