"""Keyboard, mouse and clipboard control of the app the user is working in."""

from __future__ import annotations

from functools import partial

from voice_agent.tools.base import CONFIRM, SAFE, Tool, ToolResult
from voice_agent.tools.desktop import parse_keys

DANGEROUS_KEYS = {("alt", "f4"), ("shift", "delete"), ("ctrl", "shift", "delete"), ("ctrl", "shift", "q"),
                  ("ctrl", "shift", "w")}


def type_text(tb, text: str, press_enter: bool = False) -> ToolResult:
    # This function types text (any language) into the current app.
    into = tb.desktop.type_text(text, press_enter)
    return ToolResult(True, {"typed": text, "into": into})


def press_keys(tb, keys: str) -> ToolResult:
    # This function presses a key or shortcut like ctrl+c.
    ks = parse_keys(keys)
    if not ks:
        return ToolResult(False, error="no keys given")
    into = tb.desktop.press_keys(ks)
    return ToolResult(True, {"pressed": "+".join(ks), "into": into})


def move_mouse(tb, x: int | None = None, y: int | None = None, direction: str = "", distance: int = 200) -> ToolResult:
    # This function moves the mouse to x,y, by a direction, or to the screen center.
    w, h = tb.desktop.screen_size()
    if direction == "center":
        x, y = w // 2, h // 2
    elif direction:
        cx, cy = tb.desktop.mouse_position()
        dx, dy = {"left": (-distance, 0), "right": (distance, 0), "up": (0, -distance),
                  "down": (0, distance)}.get(direction, (0, 0))
        x, y = cx + dx, cy + dy
    if x is None or y is None:
        return ToolResult(False, error="give x and y, or a direction")
    x, y = max(1, min(w - 2, int(x))), max(1, min(h - 2, int(y)))
    tb.desktop.move_mouse(x, y)
    return ToolResult(True, {"x": x, "y": y})


def mouse_click(tb, button: str = "left", double: bool = False, x: int | None = None,
                y: int | None = None) -> ToolResult:
    # This function clicks the mouse.
    tb.desktop.click(x, y, button=button, clicks=2 if double else 1)
    return ToolResult(True, {"button": button, "double": double, "at": list(tb.desktop.mouse_position())})


def scroll(tb, direction: str = "down", amount: int = 5) -> ToolResult:
    tb.desktop.scroll(amount if direction == "up" else -amount)
    return ToolResult(True, {"direction": direction, "amount": amount})


def read_clipboard(tb) -> ToolResult:
    return ToolResult(True, {"clipboard": (tb.desktop.clipboard_get() or "")[:800]})


def keys_risk(args: dict) -> str:
    # This function asks for confirmation before shortcuts that close or delete things.
    return CONFIRM if tuple(parse_keys(args.get("keys", ""))) in DANGEROUS_KEYS else SAFE


def tools(tb) -> list[Tool]:
    S, I, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
    return [
        Tool("type_text", "Type text into the app the user is working in (any language).",
             {"text": S, "press_enter": B}, partial(type_text, tb), ["text"]),
        Tool("press_keys", "Press a key or shortcut, e.g. 'enter', 'ctrl+c', 'alt+tab', 'ctrl+shift+t'.",
             {"keys": S}, partial(press_keys, tb), ["keys"], risk=keys_risk),
        Tool("move_mouse", "Move the mouse pointer to x,y pixels, or by a direction (left/right/up/down by "
             "distance pixels), or to the center.",
             {"x": I, "y": I, "direction": {**S, "enum": ["left", "right", "up", "down", "center"]},
              "distance": I}, partial(move_mouse, tb)),
        Tool("mouse_click", "Click the mouse at the current pointer position (or at x,y).",
             {"button": {**S, "enum": ["left", "right", "middle"]}, "double": B, "x": I, "y": I},
             partial(mouse_click, tb)),
        Tool("scroll", "Scroll the current window.", {"direction": {**S, "enum": ["up", "down"]}, "amount": I},
             partial(scroll, tb)),
        Tool("read_clipboard", "Read the text currently in the clipboard.", {}, partial(read_clipboard, tb)),
    ]
