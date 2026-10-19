"""Find, list, focus, minimize and maximize open windows."""

from __future__ import annotations

import re
from functools import partial

from voice_agent.tools.apps import INSTALLED_HINTS, _ratio, normalize
from voice_agent.tools.base import Tool, ToolResult
from voice_agent.tools.desktop import Window, is_ui_window


def user_windows(tb) -> list[Window]:
    # This function lists open windows, without the assistant's own browser tab.
    return [w for w in tb.desktop.windows() if not is_ui_window(w)]


def find_windows(tb, name: str) -> list[Window]:
    # This function finds windows matching a spoken name (title, process or app alias), best first.
    n = normalize(name)
    key, kscore = tb.resolver().canonical(name)
    hints = INSTALLED_HINTS.get(key, [key]) if key and kscore >= 0.78 else []
    scored = []
    for w in user_windows(tb):
        title, proc = normalize(w.title), normalize(re.sub(r"\.exe$", "", w.process, flags=re.I))
        s = max(_ratio(n, title), _ratio(n, proc), 0.9 if n and f" {n} " in f" {title} " else 0)
        for h in hints:
            if h and (f" {h} " in f" {title} " or h == proc or (key and key == proc)):
                s = max(s, 0.95)
        if s >= 0.75:
            scored.append((s, w))
    return [w for _, w in sorted(scored, key=lambda t: -t[0])]


def list_windows(tb) -> ToolResult:
    return ToolResult(True, {"windows": [w.label() for w in user_windows(tb)][:20]})


def focus_window(tb, name: str) -> ToolResult:
    # This function brings the named window to the front.
    wins = find_windows(tb, name)
    if not wins:
        return ToolResult(False, {"name": name}, error=f"no open window matches {name!r}")
    if not tb.desktop.focus(wins[0]):
        return ToolResult(False, {"name": name}, error=f"Windows did not let {wins[0].title!r} come to the front")
    tb.desktop._last_target = wins[0]
    return ToolResult(True, {"focused": wins[0].title})


def window_action(tb, name: str, action: str) -> ToolResult:
    # This function minimizes, maximizes or restores the named window.
    wins = find_windows(tb, name)
    if not wins:
        return ToolResult(False, {"name": name}, error=f"no open window matches {name!r}")
    tb.desktop.set_state(wins[0], action)
    return ToolResult(True, {"window": wins[0].title, "action": action})


def tools(tb) -> list[Tool]:
    S = {"type": "string"}
    return [
        Tool("list_windows", "List the open application windows.", {}, partial(list_windows, tb)),
        Tool("focus_window", "Switch to / bring to front an open window by app name or title.",
             {"name": S}, partial(focus_window, tb), ["name"]),
        Tool("window_action", "Minimize, maximize or restore an open window.",
             {"name": S, "action": {**S, "enum": ["minimize", "maximize", "restore"]}},
             partial(window_action, tb), ["name", "action"]),
    ]
