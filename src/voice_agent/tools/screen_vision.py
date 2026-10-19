"""Look at the screen with the vision model: describe it, or click an element by name."""

from __future__ import annotations

import time
from functools import partial

from voice_agent.tools.base import Tool, ToolResult


def describe_screen(tb, question: str = "", language: str = "en") -> ToolResult:
    # This function answers a question about what is on the screen.
    if tb.vision is None:
        return ToolResult(False, error="screen vision is not configured")
    p = tb.desktop.screenshot(tb.desktop.screenshots_dir / "_last_seen.png")
    text = tb.vision.describe(p, question, "ar" if language == "ar" else "en")
    return ToolResult(True, {"description": text}, image=str(p))


def click_on(tb, target: str, button: str = "left", double: bool = False) -> ToolResult:
    # This function clicks a button or link by its text: accessibility first, then vision.
    win = tb.desktop._last_target or tb.desktop.keyboard_target()
    if win:                       # the element's window must be on top, not the assistant page
        tb.desktop.focus(win)
        time.sleep(0.25)
    pt = tb.desktop.find_element(target, win)
    how = "accessibility"
    if pt is None and tb.vision is not None:
        if win:
            tb.desktop.focus(win)
            time.sleep(0.3)
        p = tb.desktop.screenshot(tb.desktop.screenshots_dir / "_last_seen.png")
        xy = tb.vision.locate(p, target)
        pt, how = ((xy[0], xy[1], target), "vision") if xy else (None, how)
    if pt is None:
        return ToolResult(False, {"target": target}, error=f"could not find {target!r} on the screen")
    tb.desktop.click(pt[0], pt[1], button=button, clicks=2 if double else 1)
    return ToolResult(True, {"clicked": pt[2], "x": pt[0], "y": pt[1], "found_by": how})


def tools(tb) -> list[Tool]:
    S = {"type": "string"}
    return [
        Tool("describe_screen", "Look at the user's screen and answer a question about what is visible "
             "(what's on my screen, read this, which app is open).",
             {"question": S, "language": {**S, "enum": ["en", "ar"]}}, partial(describe_screen, tb), timeout_s=60),
        Tool("click_on", "Click a button, link or other element on the screen by its visible text or a "
             "short description.", {"target": S, "button": {**S, "enum": ["left", "right"]},
                                    "double": {"type": "boolean"}},
             partial(click_on, tb), ["target"], timeout_s=60),
    ]
