"""Take a screenshot."""

from __future__ import annotations

from functools import partial

from voice_agent.tools.base import Tool, ToolResult


def take_screenshot(tb) -> ToolResult:
    # This function saves a screenshot to the Pictures/HAZEM folder.
    p = tb.desktop.screenshot()
    return ToolResult(True, {"saved": str(p)}, image=str(p))


def tools(tb) -> list[Tool]:
    return [Tool("take_screenshot", "Save a screenshot image to the Pictures/HAZEM folder. Only when the user asks to take or save a "
                 "screenshot; to answer what is on the screen use describe_screen.", {},
                 partial(take_screenshot, tb))]
