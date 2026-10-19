"""Real-world checks for computer control.

  uv run tools-test                 # drives YOUR desktop: opens Notepad, types EN + AR, screenshot,
                                    # windows, mouse, browser search, closes Notepad. PASS/FAIL per step.
  uv run tools-test --vision        # + asks Qwen3.5 to describe the screen and locate Notepad's menu
  uv run agent-test                 # real Ollama: does Qwen pick the right tool for EN/AR commands?
                                    # (dry-run desktop: nothing is executed)

Don't touch mouse/keyboard while tools-test runs (~20 s). Move the mouse into a
screen corner to abort (pyautogui fail-safe).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from voice_agent.config import load_config
from voice_agent.tools.toolbox import Toolbox
from voice_agent.tools.desktop import DryRunDesktop, is_ui_window, make_desktop

EN_TEXT = "Hello from HAZEM 123"
AR_TEXT = "مرحبا من حزم"


class Report:
    def __init__(self):
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:28} {detail}", flush=True)
        return ok

    def done(self) -> int:
        bad = sum(not ok for _, ok, _ in self.rows)
        print(f"\nRESULT: {len(self.rows) - bad}/{len(self.rows)} passed")
        return 1 if bad else 0


def _editor_text(box: Toolbox) -> str | None:
    """Read the editor's text back with select all + copy.

    A marker is put on the clipboard first, so a slow copy is never mistaken for wrong text.
    Returns None if the copy never arrived (or the editor is empty)."""
    d = box.desktop
    marker = f"HAZEM-readback-{time.time_ns()}"
    for _ in range(2):
        d.clipboard_set(marker)
        d.press_keys(["ctrl", "a"])
        time.sleep(0.15)
        d.press_keys(["ctrl", "c"])
        got, deadline = marker, time.monotonic() + 2.0
        while got == marker and time.monotonic() < deadline:
            time.sleep(0.1)
            try:
                got = d.clipboard_get()
            except Exception:  # noqa: BLE001  (clipboard busy: try again)
                pass
        d.press_keys(["ctrl", "end"])     # deselect, caret at the end (next typing appends)
        if got != marker:
            return got.replace("\r\n", "\n")
    return None


def _shown(got: str | None) -> str:
    # This function describes what was read back from the editor, for the report line.
    return "editor could not be read back (Ctrl+A / Ctrl+C gave nothing)" if got is None else f"editor has {got[-80:]!r}"


def tools_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the desktop tools for real and verify each one")
    ap.add_argument("--desktop", default="auto", help="auto | windows | x11")
    ap.add_argument("--vision", action="store_true", help="also test screen description + locate (needs Ollama)")
    ap.add_argument("--no-browser", action="store_true", help="skip the browser search step")
    args = ap.parse_args(argv)
    cfg = load_config()
    desk = make_desktop(args.desktop, screenshots_dir=cfg.tools.screenshots_dir, browser=cfg.tools.browser)
    if isinstance(desk, DryRunDesktop):
        print("No desktop found (not Windows, no DISPLAY). Nothing to test.")
        return 1
    vision = None
    if args.vision:
        from voice_agent.tools.vision import OllamaVision
        vision = OllamaVision(cfg.llm.host, cfg.tools.vision_model or cfg.llm.model, num_ctx=cfg.llm.num_ctx)
    box = Toolbox(desk, cfg.tools.data_dir, vision=vision)
    reg, rep = box.registry, Report()
    print(f"desktop backend: {desk.name}; screen {desk.screen_size()}; do not touch mouse/keyboard...\n")

    t0 = time.perf_counter()
    apps = desk.list_apps()
    rep.check("index installed apps", len(apps) > 0, f"{len(apps)} apps in {time.perf_counter() - t0:.1f}s")
    for spoken in ("notepad", "المفكرة", "calculator", "الحاسبة", "chrome", "كروم"):
        m = box.resolver().resolve(spoken)
        rep.check(f"resolve '{spoken}'", m is not None, m.entry.name if m else "no match")

    # start clean: editors left open by an earlier (interrupted) run are closed first
    stale = box.find_windows("notepad")
    if stale:
        r = reg.run_sync("close_app", {"name": "notepad"})
        print(f"[prep] closed {len(stale)} leftover editor window(s): {r.ok} {r.error}")

    r = reg.run_sync("open_app", {"name": "notepad"})
    rep.check("open_app notepad", r.ok and bool(r.data.get("window")), f"{r.data} {r.error}")
    wins = box.find_windows("notepad")
    win = next((w for w in wins if w.title == r.data.get("window")), wins[0] if wins else None)
    rep.check("window appears", win is not None, win.label() if win else str([w.title for w in desk.windows()]))
    if win is None:
        return rep.done()
    r = reg.run_sync("focus_window", {"name": "notepad"})
    rep.check("focus editor", r.ok, r.error)
    desk.press_keys(["ctrl", "a"])                  # empty document: restored tabs/old text never interfere
    desk.press_keys(["delete"])

    # each step reads the editor back, so a failure names the step that really went wrong
    r = reg.run_sync("type_text", {"text": EN_TEXT})
    time.sleep(0.3)
    got = _editor_text(box)
    rep.check("type English", r.ok and EN_TEXT in (got or ""), f"into {r.data.get('into')!r}; {_shown(got)}")
    reg.run_sync("press_keys", {"keys": "enter"})
    r = reg.run_sync("type_text", {"text": AR_TEXT})
    time.sleep(0.3)
    got = _editor_text(box)
    rep.check("type Arabic (clipboard)", r.ok and AR_TEXT in (got or ""), f"into {r.data.get('into')!r}; {_shown(got)}")
    both = got or ""
    rep.check("typed text is in the editor", EN_TEXT in both and AR_TEXT in both
              and both.index(EN_TEXT) < both.index(AR_TEXT), _shown(got))

    # focus tracking: with the HAZEM UI in front, keystrokes must still go to the user's app
    ui = [w for w in desk.windows() if is_ui_window(w)]
    if ui:
        desk.remember_target()
        desk.focus(ui[0])
        time.sleep(0.5)
        r = reg.run_sync("type_text", {"text": " +UI"})
        time.sleep(0.3)
        got = _editor_text(box)
        # only the new text is checked here: this step is about focus, not about the earlier typing
        rep.check("typing with HAZEM UI in front", r.ok and (got or "").rstrip().endswith("+UI"),
                  f"into {r.data.get('into')!r}, UI window {ui[0].title!r}; {_shown(got)}")
    else:
        print("[SKIP] typing with HAZEM UI in front  (open http://127.0.0.1:8000 in Chrome to include this step)")

    r = reg.run_sync("take_screenshot", {})
    p = Path(r.data.get("saved", ""))
    rep.check("take_screenshot", r.ok and p.is_file() and p.stat().st_size > 10_000,
              f"{p} ({p.stat().st_size // 1024} KB)" if p.is_file() else r.error)

    r = reg.run_sync("list_windows", {})
    rep.check("list_windows", r.ok and any("notepad" in w.lower() or "mousepad" in w.lower()
                                         for w in r.data["windows"]), f"{len(r.data['windows'])} windows")

    w, h = desk.screen_size()
    r = reg.run_sync("move_mouse", {"x": w // 3, "y": h // 3})
    pos = desk.mouse_position()
    rep.check("move_mouse absolute", abs(pos[0] - w // 3) <= 2 and abs(pos[1] - h // 3) <= 2, f"{pos}")
    reg.run_sync("move_mouse", {"direction": "right", "distance": 100})
    pos2 = desk.mouse_position()
    rep.check("move_mouse relative", abs(pos2[0] - pos[0] - 100) <= 2, f"{pos} -> {pos2}")

    r = reg.run_sync("window_action", {"name": "notepad", "action": "minimize"})
    time.sleep(0.5)
    rep.check("minimize", r.ok, r.error)
    r = reg.run_sync("focus_window", {"name": "notepad"})
    time.sleep(0.5)
    fg = desk.foreground()
    rep.check("focus_window restores it", r.ok and fg is not None and fg.id == win.id,
              f"fg={fg.title if fg else None!r} {r.error}")

    if vision is not None:
        r = reg.run_sync("describe_screen", {"question": "Which app is in front and what text is in it?"})
        rep.check("describe_screen (EN)", r.ok and len(r.data.get("description", "")) > 10,
                  (r.data.get("description") or r.error)[:160])
        r = reg.run_sync("describe_screen", {"question": "ما البرنامج الظاهر؟", "language": "ar"})
        rep.check("describe_screen (AR)", r.ok and any("؀" <= c <= "ۿ" for c in r.data.get("description", "")),
                  (r.data.get("description") or r.error)[:160])
        shot = desk.screenshot(desk.screenshots_dir / "_locate.png")
        t0 = time.perf_counter()
        pt = vision.locate(shot, "the File menu of the text editor")
        rep.check("vision locate 'File' menu", pt is not None, f"{pt} in {time.perf_counter() - t0:.1f}s")

    if not args.no_browser:
        r = reg.run_sync("web_search", {"query": "Riyadh weather"})
        time.sleep(4)
        titles = [x.title for x in desk.windows()]
        rep.check("web_search opens browser", r.ok and any("riyadh" in t.lower() or "google" in t.lower()
                                                           for t in titles), f"{r.data.get('browser')}: {titles[-3:]}")

    # leave no trace: close it with unsaved text; close_app must answer the "save?" prompt itself
    r = reg.run_sync("close_app", {"name": "notepad"})
    left = [w for w in box.find_windows("notepad") if desk.is_open(w)]
    rep.check("close_app notepad (unsaved text, no prompt left)", r.ok and not left,
              f"{r.data} {r.error}; still open: {[w.title for w in left]}")
    return rep.done()


# LLM tool choice

AGENT_CASES = [
    ("Open Notepad and type hello world", ["open_app"]),
    ("Can you pull up the calculator for me?", ["open_app"]),
    ("Search YouTube for relaxing rain sounds", ["youtube_search"]),
    ("What time is it?", ["get_date_time"]),
    ("Open my Google Calendar", ["open_calendar"]),
    ("Look up the best restaurants in Jeddah", ["web_search"]),
    ("What's on my screen right now?", ["describe_screen"]),
    ("Take a screenshot please", ["take_screenshot"]),
    ("Turn the volume down a bit", ["volume"]),
    ("Close Chrome", ["close_app"]),
    ("Switch to the Word window", ["focus_window"]),
    ("Remind me in ten minutes to check the oven", ["set_timer"]),
    ("Remember that my favourite team is Al Hilal", ["remember"]),
    ("How much battery do I have left?", ["system_status"]),
    ("Press control shift T", ["press_keys"]),
    ("Click the Sign in button", ["click_on"]),
    ("What is the capital of Japan?", []),
    ("افتح المفكرة واكتب مرحبا", ["open_app"]),
    ("ابحث في جوجل عن أسعار الذهب", ["web_search"]),
    ("خذ لقطة شاشة", ["take_screenshot"]),
    ("ارفع الصوت شوي", ["volume"]),
    ("اقفل كروم", ["close_app"]),
    ("ماذا يظهر على شاشتي؟", ["describe_screen"]),
    ("ذكرني بعد خمس دقائق أن أشرب الماء", ["set_timer"]),
    ("احفظ ملاحظة: اجتماع الساعة ثلاثة", ["add_note"]),
    ("من هو مؤلف كتاب الأيام؟", []),
]


async def _agent_run(cases, llm_only: bool = False) -> int:
    """Same path as the live assistant: router first, then the LLM with tools (and the same
    retry). llm_only=True skips the router to test the LLM alone."""
    from datetime import datetime

    from voice_agent.knowledge import build_messages
    from voice_agent.language import directive, text_language
    from voice_agent.llm import OllamaClient, ToolCall
    from voice_agent.server.agent import RETRY_NOTE, Agent, needs_tool_retry

    cfg = load_config()
    box = Toolbox(DryRunDesktop(), cfg.tools.data_dir / "_agent_test")
    agent = Agent(box)
    llm = OllamaClient(cfg.llm)

    async def ask(msgs):
        # This function streams one LLM step and returns its tool calls and text.
        calls, said, t0, first = [], "", time.perf_counter(), None
        async for ev in llm.stream_events(msgs, agent.schemas()):
            first = first or (time.perf_counter() - t0) * 1000
            if isinstance(ev, ToolCall):
                calls.append(ev)
            else:
                said += ev
        return calls, said, first or 0

    ok_n, lat, by_source = 0, [], {"router": 0, "llm": 0, "llm-retry": 0}
    try:
        await llm.warmup()
        for text, want in cases:
            lang = text_language(text)
            routed = None if llm_only else agent.route(text, lang)
            if routed is not None:
                names, shown, first, src = [routed.tool], f"{routed.tool}({routed.args})", 0.0, "router"
            else:
                msgs = build_messages(cfg.llm.system_prompt + "\n\n" + agent.system_note(), [], text, datetime.now(),
                                      "", directive(lang), cfg.llm.cache_friendly)
                calls, said, first = await ask(msgs)
                src = "llm"
                if needs_tool_retry(text, calls, said):
                    more, said, _ = await ask(msgs + [{"role": "assistant", "content": said},
                                                      {"role": "user", "content": RETRY_NOTE}])
                    calls, src = calls + more, "llm-retry"
                names = [c.name for c in calls]
                shown = ", ".join(f"{c.name}({c.arguments})" for c in calls) or f"says: {said.strip()[:70]!r}"
                lat.append(first)
            good = (names[:len(want)] == want) if want else not names
            ok_n += good
            by_source[src] += 1
            print(f"[{'PASS' if good else 'FAIL'}] {src:<9} {first:5.0f} ms  {text}\n        -> {shown}", flush=True)
    finally:
        await llm.aclose()
    lat.sort()
    med = f"{lat[len(lat) // 2]:.0f} ms" if lat else "-"
    print(f"\nRESULT: {ok_n}/{len(cases)} correct tool choices ({by_source}); LLM first event median {med} "
          f"(model {cfg.llm.model})")
    return 0 if ok_n == len(cases) else 1


def agent_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check tool choices for EN/AR commands (nothing is executed)")
    ap.add_argument("prompts", nargs="*")
    ap.add_argument("--llm-only", action="store_true", help="skip the fast router: test the LLM's own choices")
    args = ap.parse_args(argv)
    cases = [(p, []) for p in args.prompts] if args.prompts else AGENT_CASES
    return asyncio.run(_agent_run(cases, args.llm_only))


if __name__ == "__main__":
    sys.exit(tools_main())
