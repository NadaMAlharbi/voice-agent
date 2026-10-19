"""Launch and close applications."""

from __future__ import annotations

import time
from functools import partial

from voice_agent.tools.base import CONFIRM, Tool, ToolResult
from voice_agent.tools.desktop import Window, is_ui_window
from voice_agent.tools.website_open import site_url


def open_app(tb, name: str) -> ToolResult:
    # This function opens an installed app, or a known website if no app matches.
    m = tb.resolver().resolve(name)
    if m is None:
        url = site_url(name)
        if url:
            used = tb.desktop.open_url(url)
            return ToolResult(True, {"opened": url, "browser": used, "kind": "website"})
        return ToolResult(False, {"name": name}, error=f"no installed app or website matches {name!r}")
    before = tb.desktop.foreground()
    tb.desktop.launch(m.entry)
    win = wait_app_window(tb, name, before)
    if win:
        tb.desktop._last_target = win
    return ToolResult(True, {"opened": m.entry.name, "window": win.title if win else None, "kind": "app"})


def wait_app_window(tb, name: str, before: Window | None, timeout: float = 6.0) -> Window | None:
    # This function waits until the app's window is in front: a new window, or an app that was
    # already open and is now in front again (like Windows 11 Notepad).
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        time.sleep(0.2)
        try:
            fg = tb.desktop.foreground()
            mine = {w.id for w in tb.find_windows(name)}
        except Exception:  # noqa: BLE001
            return None
        if fg is None or is_ui_window(fg):
            continue
        if fg.id in mine:
            return next(w for w in tb.find_windows(name) if w.id == fg.id)
        is_new = before is None or fg.id != before.id
        if is_new and time.monotonic() - t0 > 1.5:      # app whose title does not contain its name
            return fg
    return None


def close_app(tb, name: str) -> ToolResult:
    # This function closes all windows of the named app. The user already confirmed, so a
    # "save changes?" prompt is answered "Don't save"; if that fails, the prompt is cancelled.
    wins = tb.find_windows(name)
    if not wins:
        return ToolResult(False, {"name": name}, error=f"no open window matches {name!r}")
    target = wins[0]
    same = [w for w in wins if w.process and w.process == target.process] or [target]
    for w in same:
        tb.desktop.close(w)
    still = wait_closed(tb, same, 2.0)
    for _ in range(4):                             # one prompt per unsaved document (tabs)
        if not still:
            break
        tb.desktop.answer_save_prompt(still[0], discard=True)
        still = wait_closed(tb, still, 1.5)
        for w in still:                            # the app may need WM_CLOSE again after a prompt
            tb.desktop.close(w)
        still = wait_closed(tb, still, 0.8)
    if still:
        for w in still:
            tb.desktop.answer_save_prompt(w, discard=False)
        return ToolResult(False, {"name": name, "asking_to_save": [w.title for w in still]},
                          error=f"{name} did not close (it is asking to save); the prompt was cancelled")
    return ToolResult(True, {"closed": [w.title for w in same]})


def wait_closed(tb, wins: list[Window], timeout: float) -> list[Window]:
    # This function waits until the windows are gone and returns those still open.
    t0 = time.monotonic()
    still = [w for w in wins if tb.desktop.is_open(w)]
    while still and time.monotonic() - t0 < timeout:
        time.sleep(0.15)
        still = [w for w in still if tb.desktop.is_open(w)]
    return still


def tools(tb) -> list[Tool]:
    S = {"type": "string"}
    return [
        Tool("open_app", "Open/launch an installed application or a well-known website by name "
             "(e.g. notepad, calculator, chrome, word, spotify, youtube).",
             {"name": {**S, "description": "app or website name as the user said it"}}, partial(open_app, tb), ["name"]),
        Tool("close_app", "Close an open application window by name (unsaved changes are discarded after the user "
             "confirms).", {"name": S}, partial(close_app, tb), ["name"], risk=CONFIRM, timeout_s=30),
    ]
