"""Desktop backends: how the assistant controls the computer.

  WindowsDesktop  native Windows (the main target): launch apps, manage windows,
                  find buttons (UI Automation), media keys
  X11Desktop      Linux X11, used to test the same tools on a virtual desktop
  DryRunDesktop   records every action and touches nothing (`agent-test`, tools.desktop: dry-run)

Shared code: screenshots (mss), mouse and keyboard (pyautogui, loaded only when used),
pasting Arabic text through the clipboard (pyautogui types ASCII only), opening the browser.

Focus: you talk to HAZEM in a browser tab, so that tab is in front when a command
arrives. Typing must go to the app you were using, not to the HAZEM page, so the
last window that is not the HAZEM page is remembered and used for keyboard actions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from datetime import datetime
from pathlib import Path

from voice_agent.tools.apps import AppEntry

BROWSERS = ("chrome", "msedge", "firefox", "brave", "opera", "vivaldi", "chromium", "iexplore")


def _ui_title() -> str:
    # This function reads the page title from index.html, so renaming the page never breaks the tools.
    try:
        html = (Path(__file__).parents[1] / "server" / "static" / "index.html").read_text(encoding="utf-8")
        m = re.search(r"<title>(.*?)</title>", html, re.S | re.I)
        if m and m.group(1).strip():
            return " ".join(m.group(1).split())
    except OSError:
        pass
    return "HAZEM Voice Agent"


UI_TITLE_MARK = _ui_title()   # the assistant's browser tab: never a keyboard target


def is_ui_window(w) -> bool:
    # This function checks if a window is the assistant's own browser tab: a browser window whose
    # title starts with the page title. A Notepad tab that only mentions the name is a normal window.
    if not w or not w.title.strip().lower().startswith(UI_TITLE_MARK.lower()):
        return False
    proc = (getattr(w, "process", "") or "").lower()
    return not proc or any(b in proc for b in BROWSERS)


# spoken / LLM key names -> pyautogui names
KEY_ALIASES = {
    "control": "ctrl", "ctl": "ctrl", "cmd": "win", "windows": "win", "super": "win", "meta": "win",
    "return": "enter", "esc": "escape", "del": "delete", "ins": "insert", "pgup": "pageup",
    "pgdn": "pagedown", "page up": "pageup", "page down": "pagedown", "spacebar": "space",
    "arrow up": "up", "arrow down": "down", "arrow left": "left", "arrow right": "right",
    "option": "alt", "backspace": "backspace", "back space": "backspace", "print screen": "printscreen",
}


def parse_keys(spec: str) -> list[str]:
    """'Ctrl + Shift + T' -> ['ctrl', 'shift', 't']."""
    parts = [p.strip().lower() for p in spec.replace("-", "+").split("+") if p.strip()]
    return [KEY_ALIASES.get(p, p) for p in parts]


@dataclass
class Window:
    id: int | str
    title: str
    process: str = ""

    def label(self) -> str:
        return f"{self.title} ({self.process})" if self.process else self.title


@dataclass
class Desktop(ABC):
    screenshots_dir: Path = field(default_factory=lambda: Path.home() / "Pictures" / "HAZEM")
    browser: str = "chrome"
    actions: list = field(default_factory=list)       # log of what was done (shown in the UI)
    _last_target: Window | None = None

    name = "abstract"

    # Pasting borrows the clipboard: the text goes in, Ctrl+V is pressed, and the old content
    # comes back clipboard_restore_s later (apps read the clipboard a moment after the keys).
    clipboard_restore_s = 1.5
    _clip_lock = threading.Lock()
    _clip_pending = None                  # (pasted text, content to put back) while a restore waits
    _clip_gen = 0

    def __post_init__(self) -> None:
        """Subclasses (not dataclasses themselves) set up their OS handles here."""

    # ---- apps & windows (backend specific)
    @abstractmethod
    def list_apps(self) -> list[AppEntry]: ...
    @abstractmethod
    def builtin_apps(self) -> dict[str, AppEntry]: ...
    @abstractmethod
    def launch(self, entry: AppEntry) -> None: ...
    @abstractmethod
    def windows(self) -> list[Window]: ...
    @abstractmethod
    def foreground(self) -> Window | None: ...
    @abstractmethod
    def focus(self, win: Window) -> bool: ...
    @abstractmethod
    def close(self, win: Window) -> None: ...
    @abstractmethod
    def set_state(self, win: Window, state: str) -> None: ...  # minimize | maximize | restore

    # ---- system (optional per backend)
    def lock(self) -> None:
        raise NotImplementedError("locking the screen is not supported on this desktop")

    def power(self, action: str) -> None:
        raise NotImplementedError("power actions are not supported on this desktop")

    def volume(self, action: str, steps: int) -> None:
        key = {"up": "volumeup", "down": "volumedown", "mute": "volumemute"}[action]
        self._gui().press(key, presses=1 if action == "mute" else steps, interval=0.02)

    def media(self, action: str) -> None:
        key = {"play_pause": "playpause", "next": "nexttrack", "previous": "prevtrack", "stop": "stop"}[action]
        self._gui().press(key)

    def find_element(self, label: str, win: Window | None) -> tuple[int, int, str] | None:
        """Screen point of a UI element by its visible name (accessibility tree)."""
        return None

    def clipboard_get(self) -> str:
        import pyperclip
        return pyperclip.paste()

    def clipboard_set(self, text: str) -> None:
        import pyperclip
        pyperclip.copy(text)

    # ---- shared implementations
    @staticmethod
    def _gui():
        import pyautogui  # needs a display: import only when used
        pyautogui.PAUSE = 0.02
        pyautogui.FAILSAFE = True   # slam the mouse into a screen corner to abort
        return pyautogui

    def log(self, what: str) -> None:
        self.actions.append((time.time(), what))

    def screen_size(self) -> tuple[int, int]:
        import mss
        with mss.mss() as s:
            m = s.monitors[1]
            return m["width"], m["height"]

    def screenshot(self, path: Path | None = None, monitor: int = 1) -> Path:
        import mss
        import mss.tools
        path = Path(path) if path else self.screenshots_dir / f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        with mss.mss() as s:
            mons = s.monitors
            img = s.grab(mons[monitor if monitor < len(mons) else 1])
            mss.tools.to_png(img.rgb, img.size, output=str(path))
        self.log(f"screenshot {path}")
        return path

    def open_url(self, url: str) -> str:
        exe = self._browser_exe()
        if exe:
            subprocess.Popen([exe, url], **_detached())
            used = Path(exe).stem
        else:
            webbrowser.open(url)
            used = "default browser"
        self.log(f"open_url {url} ({used})")
        return used

    def _browser_exe(self) -> str | None:
        return None

    def is_open(self, win: Window) -> bool:
        return any(w.id == win.id for w in self.windows())

    def answer_save_prompt(self, win: Window, discard: bool) -> bool:
        """Answer an app's "save changes?" prompt: discard=True -> "Don't save", False -> "Cancel".
        Generic version: keyboard on the app window. Returns True if a key was sent."""
        if not self.focus(win):
            return False
        self.press_keys(["alt", "n"] if discard else ["escape"])
        return True

    # ---- Windows Calculator (tools/calculator.py)
    def calculator_input(self, win: Window, tokens: list[str], delay_s: float = 0.08) -> None:
        """Press calculator keys one by one, so the user sees them. Tokens: "k:<key name>" or "c:<char>"."""
        raise NotImplementedError("the calculator tool works on Windows only")

    def calculator_display(self, win: Window) -> str | None:
        """What the calculator shows (accessibility name of its result field), or None."""
        return None

    def remember_target(self) -> None:
        """Called periodically: remember the last real app the user was in."""
        w = self.foreground()
        if w and not is_ui_window(w) and w.title.strip():
            self._last_target = w

    def keyboard_target(self) -> Window | None:
        """Make sure keystrokes go to the user's app, not the HAZEM page.
        Raises if that app cannot be brought to the front (nothing is typed into the wrong window)."""
        self.remember_target()
        fg = self.foreground()
        if is_ui_window(fg):
            if not self._last_target:
                raise RuntimeError("no app to type into: click the app you want first")
            if not self.focus(self._last_target):
                raise RuntimeError(f"could not bring {self._last_target.title!r} to the front")
            time.sleep(0.15)                  # let the app restore keyboard focus to its text field
            return self._last_target
        return fg

    def type_text(self, text: str, press_enter: bool = False) -> str | None:
        target = self.keyboard_target()
        gui = self._gui()
        if text.isascii():
            gui.write(text, interval=0.01)
        else:  # Arabic (and any non-ASCII): pyautogui can only type ASCII, so paste it
            self._paste(text, lambda: gui.hotkey("ctrl", "v"))
        if press_enter:
            gui.press("enter")
        self.log(f"type {text!r} -> {target.title if target else '?'}")
        return target.title if target else None

    def _paste(self, text: str, press_paste) -> None:
        # This function pastes text through the clipboard. The old content is put back a bit later,
        # and only if our text is still there: some apps read the clipboard late, and anything the
        # user copied in the meantime is kept.
        with self._clip_lock:
            old = None
            try:
                old = self.clipboard_get()
            except Exception:  # noqa: BLE001
                pass
            if self._clip_pending and old == self._clip_pending[0]:
                old = self._clip_pending[1]       # an earlier paste is still waiting: keep the original
            self._clip_gen += 1
            gen = self._clip_gen
            self._clip_pending = (text, old)
            self.clipboard_set(text)
        time.sleep(0.1)                           # clipboard watchers (history, sync) finish first
        press_paste()
        time.sleep(0.3)                           # the paste lands before any key that follows
        timer = threading.Timer(self.clipboard_restore_s, self._restore_clipboard, args=(gen,))
        timer.daemon = True
        timer.start()

    def _restore_clipboard(self, gen: int) -> None:
        # This function puts the old clipboard content back after a paste.
        with self._clip_lock:
            if gen != self._clip_gen or not self._clip_pending:
                return                            # a newer paste took over
            text, old = self._clip_pending
            self._clip_pending = None
            try:
                if old is not None and self.clipboard_get() == text:
                    self.clipboard_set(old)
            except Exception:  # noqa: BLE001
                pass

    def press_keys(self, keys: list[str]) -> str | None:
        target = self.keyboard_target()
        gui = self._gui()
        if len(keys) == 1:
            gui.press(keys[0])
        else:
            gui.hotkey(*keys)
        self.log(f"keys {'+'.join(keys)} -> {target.title if target else '?'}")
        return target.title if target else None

    def mouse_position(self) -> tuple[int, int]:
        p = self._gui().position()
        return int(p[0]), int(p[1])

    def move_mouse(self, x: int, y: int, duration: float = 0.2) -> None:
        self._gui().moveTo(x, y, duration=duration)
        self.log(f"mouse -> {x},{y}")

    def click(self, x: int | None = None, y: int | None = None, button: str = "left", clicks: int = 1) -> None:
        gui = self._gui()
        if x is not None and y is not None:
            gui.click(x, y, clicks=clicks, interval=0.08, button=button)
        else:
            gui.click(clicks=clicks, interval=0.08, button=button)
        self.log(f"click {button} x{clicks} at {self.mouse_position()}")

    def scroll(self, amount: int) -> None:
        self._gui().scroll(amount)
        self.log(f"scroll {amount}")


def _detached() -> dict:
    if sys.platform == "win32":
        return {"creationflags": 0x00000008 | 0x00000200, "close_fds": True}  # DETACHED | NEW_PROCESS_GROUP
    return {"start_new_session": True, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}


# --------------------------------------------------------------------------- Windows

class WindowsDesktop(Desktop):
    """Native Windows. ctypes for Win32 (no pywin32 needed), uiautomation for element search."""

    name = "windows"

    BUILTINS = {
        "notepad": "notepad.exe", "calculator": "calc.exe", "paint": "mspaint.exe", "cmd": "cmd.exe",
        "powershell": "powershell.exe", "explorer": "explorer.exe", "taskmgr": "taskmgr.exe",
        "control": "control.exe", "snippingtool": "snippingtool.exe", "settings": "ms-settings:",
        "chrome": "chrome", "edge": "msedge", "firefox": "firefox", "word": "winword", "excel": "excel",
        "powerpoint": "powerpnt", "outlook": "outlook", "vscode": "code", "camera": "microsoft.windows.camera:",
        "photos": "ms-photos:",
    }

    def __post_init__(self):
        import ctypes

        import ctypes.wintypes as wt

        self._ct = ctypes
        self._u32 = u32 = ctypes.windll.user32
        H = wt.HWND
        for name, args, res in [            # explicit signatures: 64-bit handles are never truncated
            ("GetForegroundWindow", (), H), ("GetWindow", (H, wt.UINT), H), ("IsWindowVisible", (H,), wt.BOOL),
            ("IsIconic", (H,), wt.BOOL), ("GetWindowLongW", (H, ctypes.c_int), ctypes.c_long),
            ("GetWindowTextLengthW", (H,), ctypes.c_int), ("GetWindowTextW", (H, wt.LPWSTR, ctypes.c_int), ctypes.c_int),
            ("GetWindowThreadProcessId", (H, ctypes.POINTER(wt.DWORD)), wt.DWORD),
            ("ShowWindow", (H, ctypes.c_int), wt.BOOL), ("SetForegroundWindow", (H,), wt.BOOL),
            ("PostMessageW", (H, wt.UINT, wt.WPARAM, wt.LPARAM), wt.BOOL),
            ("keybd_event", (wt.BYTE, wt.BYTE, wt.DWORD, ctypes.c_size_t), None), ("LockWorkStation", (), wt.BOOL),
            ("IsWindow", (H,), wt.BOOL), ("BringWindowToTop", (H,), wt.BOOL),
            ("AttachThreadInput", (wt.DWORD, wt.DWORD, wt.BOOL), wt.BOOL),
            ("GetCursorPos", (ctypes.POINTER(wt.POINT),), wt.BOOL), ("SetCursorPos", (ctypes.c_int, ctypes.c_int), wt.BOOL),
        ]:
            f = getattr(u32, name)
            f.argtypes, f.restype = args, res
        self._k32 = ctypes.windll.kernel32
        self._k32.GetCurrentThreadId.restype = wt.DWORD
        try:   # physical pixels everywhere: mss, the mouse and UI Automation then agree on coordinates
            if not u32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):   # per-monitor v2
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:  # noqa: BLE001
            try:
                self._u32.SetProcessDPIAware()
            except Exception:  # noqa: BLE001
                pass
        self._apps: list[AppEntry] | None = None
        self._apps_lock = threading.Lock()

    # apps
    def builtin_apps(self) -> dict[str, AppEntry]:
        return {k: AppEntry(k, v, "uri" if v.endswith(":") else "cmd") for k, v in self.BUILTINS.items()}

    def list_apps(self) -> list[AppEntry]:
        with self._apps_lock:
            if self._apps is None:
                self._apps = self._start_apps() + self._start_menu_links()
            return self._apps

    def _start_apps(self) -> list[AppEntry]:
        """Get-StartApps: every app in the Start menu incl. Store apps (Spotify, WhatsApp, ...)."""
        cmd = ("[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
               "Get-StartApps | Select-Object Name,AppID | ConvertTo-Json -Compress")
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                                 capture_output=True, timeout=20, creationflags=0x08000000)  # CREATE_NO_WINDOW
            import json
            data = json.loads(out.stdout.decode("utf-8-sig", errors="ignore") or "[]")
            data = [data] if isinstance(data, dict) else data
            return [AppEntry(d["Name"], d["AppID"], "appid") for d in data if d.get("Name") and d.get("AppID")]
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _start_menu_links() -> list[AppEntry]:
        roots = [Path(os.environ.get("ProgramData", r"C:\ProgramData")),
                 Path(os.environ.get("APPDATA", ""))]
        out = []
        for r in roots:
            base = r / "Microsoft" / "Windows" / "Start Menu" / "Programs"
            if base.is_dir():
                out += [AppEntry(p.stem, str(p), "lnk") for p in base.rglob("*.lnk")]
        return out

    def launch(self, entry: AppEntry) -> None:
        if entry.kind == "appid":
            subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{entry.target}"], **_detached())
        elif entry.kind in ("lnk", "uri"):
            os.startfile(entry.target)  # noqa: S606  (Windows shell launch, like double-click)
        else:  # "start" resolves App Paths (chrome, winword, msedge, ...) like Win+R does
            subprocess.Popen(["cmd", "/c", "start", "", entry.target], creationflags=0x08000000, close_fds=True)
        self.log(f"launch {entry.name}")

    def _browser_exe(self) -> str | None:
        names = {"chrome": [r"Google\Chrome\Application\chrome.exe"],
                 "edge": [r"Microsoft\Edge\Application\msedge.exe"]}.get(self.browser, [])
        for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                     os.environ.get("LOCALAPPDATA")):
            for n in names:
                if base and (p := Path(base) / n).is_file():
                    return str(p)
        return shutil.which("chrome") or shutil.which("msedge")

    # windows
    def windows(self) -> list[Window]:
        import ctypes.wintypes as wt

        u32, found = self._u32, []
        dwm = self._ct.windll.dwmapi
        GW_OWNER, GWL_EXSTYLE, WS_EX_TOOLWINDOW = 4, -20, 0x00000080

        @self._ct.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        def cb(hwnd, _):
            if not u32.IsWindowVisible(hwnd) or u32.GetWindow(hwnd, GW_OWNER):
                return True
            if u32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW:
                return True
            cloaked = wt.DWORD()       # suspended UWP apps / other virtual desktops: "visible" but not shown
            if dwm.DwmGetWindowAttribute(wt.HWND(hwnd), 14, self._ct.byref(cloaked), 4) == 0 and cloaked.value:
                return True
            n = u32.GetWindowTextLengthW(hwnd)
            if n == 0:
                return True
            buf = self._ct.create_unicode_buffer(n + 1)
            u32.GetWindowTextW(hwnd, buf, n + 1)
            proc = self._process_name(hwnd)
            if buf.value not in ("Program Manager", "Windows Input Experience"):
                found.append(Window(int(hwnd), buf.value, proc))
            return True

        u32.EnumWindows(cb, 0)
        return found

    def foreground(self) -> Window | None:
        hwnd = self._u32.GetForegroundWindow()
        if not hwnd:
            return None
        n = self._u32.GetWindowTextLengthW(hwnd)
        buf = self._ct.create_unicode_buffer(n + 1)
        self._u32.GetWindowTextW(hwnd, buf, n + 1)
        return Window(int(hwnd), buf.value, self._process_name(hwnd))

    def _process_name(self, hwnd: int) -> str:
        # This function returns the exe name of a window's process, or "" if it is unknown.
        # pid 0 means the lookup failed (it is not a real app: psutil would say "System Idle Process").
        import psutil

        pid = self._pid(hwnd)
        if pid <= 0:
            return ""
        try:
            return psutil.Process(pid).name()
        except Exception:  # noqa: BLE001  (process ended or access denied)
            return ""

    def focus(self, win: Window) -> bool:
        # This function brings a window to the front and checks that it worked. Windows blocks
        # this for background processes, so we attach to the foreground thread's input and tap ALT.
        u32, hwnd = self._u32, win.id
        if not u32.IsWindow(hwnd):
            return False
        if u32.IsIconic(hwnd):
            u32.ShowWindow(hwnd, 9)                    # SW_RESTORE
        me = self._k32.GetCurrentThreadId()
        changed = u32.GetForegroundWindow() != hwnd
        for _ in range(3):
            if u32.GetForegroundWindow() == hwnd:
                break
            fg_thread = u32.GetWindowThreadProcessId(u32.GetForegroundWindow(), None)
            attached = bool(fg_thread and fg_thread != me and u32.AttachThreadInput(me, fg_thread, True))
            try:
                u32.keybd_event(0x12, 0, 0, 0)         # tap ALT: no app is left in menu mode
                u32.keybd_event(0x12, 0, 2, 0)
                u32.BringWindowToTop(hwnd)
                u32.SetForegroundWindow(hwnd)
            finally:
                if attached:
                    u32.AttachThreadInput(me, fg_thread, False)
            time.sleep(0.08)
        ok = u32.GetForegroundWindow() == hwnd
        if ok and changed:
            time.sleep(0.25)      # the app restores keyboard focus to its text field after activation
        self.log(f"focus {win.title} {'ok' if ok else 'FAILED'}")
        return ok

    def close(self, win: Window) -> None:
        self._u32.PostMessageW(win.id, 0x0010, 0, 0)   # WM_CLOSE: the app may still ask to save
        self.log(f"close {win.title}")

    def is_open(self, win: Window) -> bool:
        # This function checks if the window still exists (also when it is hidden or cloaked).
        return bool(self._u32.IsWindow(win.id)) and bool(self._u32.IsWindowVisible(win.id))

    def _pid(self, hwnd: int) -> int:
        import ctypes.wintypes as wt

        pid = wt.DWORD()
        self._u32.GetWindowThreadProcessId(hwnd, self._ct.byref(pid))
        return int(pid.value)

    def process_windows(self, win: Window) -> list[int]:
        # This function lists every visible top-level window of the app's process, including
        # its dialogs (a "save changes?" dialog is a separate owned window in classic apps).
        import ctypes.wintypes as wt

        pid, found = self._pid(win.id), []

        @self._ct.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        def cb(hwnd, _):
            if self._u32.IsWindowVisible(hwnd) and self._pid(hwnd) == pid:
                found.append(int(hwnd))
            return True

        self._u32.EnumWindows(cb, 0)
        dialogs = [h for h in found if h != win.id]
        return dialogs + ([win.id] if win.id in found else [])    # dialogs first

    DISCARD = ("don't save", "don’t save", "do not save", "don't save changes", "discard",
               "عدم الحفظ", "لا تحفظ")
    CANCEL = ("cancel", "إلغاء")

    def _press_button(self, hwnd: int, labels: tuple[str, ...]) -> bool:
        # This function finds a button by its name in one window (UI Automation) and presses it.
        try:
            import uiautomation as auto
        except ImportError:
            return False
        from voice_agent.tools.apps import normalize

        want = {normalize(x) for x in labels}
        with auto.UIAutomationInitializerInThread():
            root = ctrl = None
            try:
                root = auto.ControlFromHandle(hwnd)
                deadline = time.monotonic() + 2.0
                for ctrl, _depth in auto.WalkControl(root, maxDepth=25):
                    if time.monotonic() > deadline:
                        break
                    if ctrl.ControlTypeName != "ButtonControl" or normalize(ctrl.Name or "") not in want:
                        continue
                    try:
                        ctrl.GetInvokePattern().Invoke()
                    except Exception:  # noqa: BLE001  (no invoke pattern: click its center)
                        r = ctrl.BoundingRectangle
                        self.click(r.xcenter(), r.ycenter())
                    self.log(f"pressed {ctrl.Name!r}")
                    return True
            except Exception:  # noqa: BLE001  (COMError: the window went away)
                return False
            finally:
                root = ctrl = None
        return False

    def answer_save_prompt(self, win: Window, discard: bool) -> bool:
        # This function answers "save changes?": the button by name in the app's dialogs or its own
        # window (Windows 11 Notepad asks inside its window); else the keyboard on the dialog.
        labels = self.DISCARD if discard else self.CANCEL
        hwnds = self.process_windows(win)
        for hwnd in hwnds:
            if self._press_button(hwnd, labels):
                return True
        if not hwnds:
            return False
        target = Window(hwnds[0], "")
        if not self.focus(target):
            return False
        from voice_agent.tools import win_input

        win_input.send(win_input.combo_events(["n"] if discard else ["escape"]))   # "Do&n't save" / Cancel
        self.log(f"save prompt: key {'n' if discard else 'escape'}")
        return True

    # mouse: Win32 directly, in physical pixels (same as screenshots and screen_size)
    def mouse_position(self) -> tuple[int, int]:
        import ctypes.wintypes as wt

        pt = wt.POINT()
        self._u32.GetCursorPos(self._ct.byref(pt))
        return int(pt.x), int(pt.y)

    def move_mouse(self, x: int, y: int, duration: float = 0.2) -> None:
        # This function glides the pointer to x,y and makes sure it arrived.
        sx, sy = self.mouse_position()
        steps = max(1, int(duration / 0.01))
        for i in range(1, steps + 1):
            self._u32.SetCursorPos(round(sx + (x - sx) * i / steps), round(sy + (y - sy) * i / steps))
            time.sleep(duration / steps)
        for _ in range(3):
            if self.mouse_position() == (x, y):
                break
            self._u32.SetCursorPos(x, y)
            time.sleep(0.02)
        self.log(f"mouse -> {x},{y}")

    def set_state(self, win: Window, state: str) -> None:
        self._u32.ShowWindow(win.id, {"minimize": 6, "maximize": 3, "restore": 9}[state])
        if state != "minimize":
            self.focus(win)
        self.log(f"{state} {win.title}")

    def scroll(self, amount: int) -> None:
        self._gui().scroll(amount * 120)   # pyautogui on Windows passes raw wheel units (120 = one notch)
        self.log(f"scroll {amount}")

    # keyboard: layout independent (an Arabic layout must not turn "hello" into Arabic letters)
    def type_text(self, text: str, press_enter: bool = False) -> str | None:
        from voice_agent.tools import win_input

        target = self.keyboard_target()
        # All text is pasted, in one go, as exact Unicode. Typed key events are not reliable:
        # Windows 11 Notepad drops or repeats fast typed characters (tools-test saw
        # "مرحبا من حزم" become "مرحبا جججحزم"), and its autocorrect changes typed English
        # words. A paste is not autocorrected. Ctrl+V is sent as virtual keys (any layout).
        # "\r\n" line ends work in classic and new Notepad. Typing is only the fallback.
        try:
            self._paste(text.replace("\r\n", "\n").replace("\n", "\r\n"),
                        lambda: win_input.send(win_input.combo_events(["ctrl", "v"])))
        except Exception:  # noqa: BLE001  (clipboard busy or blocked)
            win_input.send_text(text)
        if press_enter:
            win_input.send(win_input.combo_events(["enter"]))
        self.log(f"type {text!r} -> {target.title if target else '?'}")
        return target.title if target else None

    def press_keys(self, keys: list[str]) -> str | None:
        from voice_agent.tools import win_input

        target = self.keyboard_target()
        win_input.send(win_input.combo_events(keys))
        self.log(f"keys {'+'.join(keys)} -> {target.title if target else '?'}")
        return target.title if target else None

    def calculator_input(self, win: Window, tokens: list[str], delay_s: float = 0.08) -> None:
        from voice_agent.tools import win_input

        if not self.focus(win):
            raise RuntimeError(f"could not bring {win.title!r} to the front")
        time.sleep(0.25)                                  # the calculator takes keyboard focus
        for t in tokens:
            kind, value = t.split(":", 1)
            if kind == "k":                               # virtual keys: same on every layout
                win_input.send(win_input.combo_events(value.split("+")))
            else:                                         # characters like "(" as Unicode
                win_input.send(win_input.text_events(value))
            time.sleep(delay_s)
        self.log(f"calculator keys {' '.join(t[2:] for t in tokens)} -> {win.title}")

    def calculator_display(self, win: Window) -> str | None:
        # Windows Calculator names its result field "Display is 42" (in the UI language).
        try:
            import uiautomation as auto
        except ImportError:
            return None
        with auto.UIAutomationInitializerInThread():
            root = ctrl = None
            try:
                root = auto.ControlFromHandle(win.id)
                ctrl = root.Control(searchDepth=30, AutomationId="CalculatorResults")
                return ctrl.Name if ctrl.Exists(2, 0.1) else None
            except Exception:  # noqa: BLE001  (window closed, COM error)
                return None
            finally:
                root = ctrl = None                        # release COM objects before CoUninitialize

    def volume(self, action: str, steps: int) -> None:
        from voice_agent.tools import win_input

        key = {"up": "volumeup", "down": "volumedown", "mute": "volumemute"}[action]
        n = 1 if action == "mute" else steps          # one step = 2 %
        win_input.send(win_input.combo_events([key]) * n)
        self.log(f"volume {action} x{n}")

    def media(self, action: str) -> None:
        from voice_agent.tools import win_input

        key = {"play_pause": "playpause", "next": "nexttrack", "previous": "prevtrack", "stop": "stop"}[action]
        win_input.send(win_input.combo_events([key]))
        self.log(f"media {action}")

    # system
    def lock(self) -> None:
        self._u32.LockWorkStation()
        self.log("lock")

    def power(self, action: str) -> None:
        args = {"shutdown": ["/s", "/t", "30"], "restart": ["/r", "/t", "30"], "cancel": ["/a"]}[action]
        subprocess.run(["shutdown", *args], creationflags=0x08000000, check=False)
        self.log(f"power {action}")

    def find_element(self, label: str, win: Window | None) -> tuple[int, int, str] | None:
        """UI Automation: buttons, links, menu items, tabs ... by visible name (fast, exact)."""
        try:
            import uiautomation as auto
        except ImportError:
            return None
        from voice_agent.tools.apps import _ratio, normalize

        want, best, score = normalize(label), None, 0.0
        if win is not None and not self._u32.IsWindow(win.id):
            return None                                  # that window was closed
        with auto.UIAutomationInitializerInThread():     # COM must be initialised in this worker thread
            root = ctrl = None
            try:
                root = auto.ControlFromHandle(win.id) if win else auto.GetForegroundControl()
                deadline, seen = time.monotonic() + 2.5, 0
                for ctrl, _depth in auto.WalkControl(root, maxDepth=25):
                    seen += 1
                    if seen > 4000 or time.monotonic() > deadline:
                        break
                    name = (ctrl.Name or "").strip()
                    if not name or len(name) > 80:
                        continue
                    r = _ratio(want, normalize(name))
                    if r > score:
                        rect = ctrl.BoundingRectangle
                        if rect.width() > 0 and rect.height() > 0:
                            best, score = (rect.xcenter(), rect.ycenter(), name), r
            except Exception:  # noqa: BLE001  (COMError from a dying window: fall back to vision)
                best = None
            finally:
                root = ctrl = None                       # release COM objects before CoUninitialize
        return best if best and score >= 0.75 else None


# --------------------------------------------------------------------------- Linux X11

class X11Desktop(Desktop):
    """Linux/X11 (wmctrl, xdotool, xclip). Used to run the real tools on a virtual desktop."""

    name = "x11"

    def builtin_apps(self) -> dict[str, AppEntry]:
        cands = {"notepad": "mousepad", "cmd": "xterm", "calculator": "gnome-calculator", "chrome": "chromium",
                 "firefox": "firefox", "xeyes": "xeyes", "explorer": "thunar"}
        return {k: AppEntry(v, v, "cmd") for k, v in cands.items() if shutil.which(v)}

    def list_apps(self) -> list[AppEntry]:
        out = []
        for d in (Path("/usr/share/applications"), Path.home() / ".local/share/applications"):
            for f in d.glob("*.desktop") if d.is_dir() else []:
                name = exe = None
                nodisplay = False
                for line in f.read_text(errors="ignore").splitlines():
                    if line.startswith("[") and line.strip() != "[Desktop Entry]" and name:
                        break
                    if line.startswith("Name=") and name is None:
                        name = line[5:].strip()
                    elif line.startswith("Exec=") and exe is None:
                        exe = " ".join(p for p in line[5:].split() if not p.startswith("%"))
                    elif line.strip() == "NoDisplay=true":
                        nodisplay = True
                if name and exe and not nodisplay:
                    out.append(AppEntry(name, exe, "desktop"))
        return out

    def launch(self, entry: AppEntry) -> None:
        import shlex
        subprocess.Popen(shlex.split(entry.target), **_detached())
        self.log(f"launch {entry.name}")

    def _browser_exe(self) -> str | None:
        return shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")

    def windows(self) -> list[Window]:
        out = subprocess.run(["wmctrl", "-lp"], capture_output=True, text=True, timeout=5).stdout
        wins = []
        for line in out.splitlines():
            parts = line.split(None, 4)
            if len(parts) >= 4:                     # id desktop pid host title
                title = parts[4] if len(parts) == 5 else ""
                proc = ""
                try:
                    import psutil
                    proc = psutil.Process(int(parts[2])).name()
                except Exception:  # noqa: BLE001
                    pass
                wins.append(Window(hex(int(parts[0], 16)), title, proc))   # same id format as foreground()
        return wins

    def foreground(self) -> Window | None:
        r = subprocess.run(["xdotool", "getactivewindow", "getwindowname"], capture_output=True, text=True,
                           timeout=5)
        r2 = subprocess.run(["xdotool", "getactivewindow"], capture_output=True, text=True, timeout=5)
        if r2.returncode != 0:
            return None
        return Window(hex(int(r2.stdout.strip())), r.stdout.strip())

    def focus(self, win: Window) -> bool:
        ok = subprocess.run(["wmctrl", "-ia", str(win.id)], timeout=5).returncode == 0
        self.log(f"focus {win.title}")
        return ok

    def close(self, win: Window) -> None:
        subprocess.run(["wmctrl", "-ic", str(win.id)], timeout=5)
        self.log(f"close {win.title}")

    def set_state(self, win: Window, state: str) -> None:
        if state == "minimize":
            subprocess.run(["xdotool", "windowminimize", str(int(str(win.id), 16))], timeout=5)
        else:
            op = "add" if state == "maximize" else "remove"
            subprocess.run(["wmctrl", "-ir", str(win.id), "-b", f"{op},maximized_vert,maximized_horz"], timeout=5)
            self.focus(win)
        self.log(f"{state} {win.title}")

    def answer_save_prompt(self, win: Window, discard: bool) -> bool:
        # GTK dialogs (Mousepad, gedit): "_Don't Save" = alt+d; Escape cancels. The dialog is in front.
        self.press_keys(["alt", "d"] if discard else ["escape"])
        return True

    def clipboard_get(self) -> str:
        return subprocess.run(["xclip", "-selection", "clipboard", "-o"], capture_output=True, text=True,
                              timeout=5).stdout

    def clipboard_set(self, text: str) -> None:
        p = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE, **_detached_io())
        p.communicate(text.encode("utf-8"), timeout=5)


def _detached_io() -> dict:
    return {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}


# --------------------------------------------------------------------------- dry run

class DryRunDesktop(Desktop):
    """Does nothing, records everything. For dry runs and LLM tool-choice checks."""

    name = "dry-run"

    def __init__(self, apps: list[AppEntry] | None = None, wins: list[Window] | None = None, **kw):
        super().__init__(**kw)
        self._apps = apps if apps is not None else [
            AppEntry("Notepad", "notepad.exe"), AppEntry("Google Chrome", "chrome"),
            AppEntry("Calculator", "calc.exe"), AppEntry("Microsoft Word", "winword"),
            AppEntry("Spotify", "Spotify.exe", "appid"), AppEntry("File Explorer", "explorer.exe"),
        ]
        self._wins = wins if wins is not None else [
            Window(1, f"{UI_TITLE_MARK} - Google Chrome", "chrome.exe"),
            Window(2, "Untitled - Notepad", "notepad.exe"),
        ]
        self._fg = self._wins[0] if self._wins else None
        self._mouse = (500, 400)
        self._clip = ""

    def builtin_apps(self):
        return {"settings": AppEntry("Settings", "ms-settings:", "uri")}

    def list_apps(self):
        return self._apps

    def launch(self, entry):
        self.log(f"launch {entry.name}")
        w = Window(len(self._wins) + 1, entry.name, entry.target)
        self._wins.append(w)
        self._fg = w

    def windows(self):
        return list(self._wins)

    def foreground(self):
        return self._fg

    def focus(self, win):
        self._fg = win
        self.log(f"focus {win.title}")
        return True

    def close(self, win):
        self._wins = [w for w in self._wins if w.id != win.id]
        self.log(f"close {win.title}")

    def set_state(self, win, state):
        self.log(f"{state} {win.title}")

    def lock(self):
        self.log("lock")

    def power(self, action):
        self.log(f"power {action}")

    def volume(self, action, steps):
        self.log(f"volume {action} {steps}")

    def media(self, action):
        self.log(f"media {action}")

    def screen_size(self):
        return 1920, 1080

    def screenshot(self, path=None, monitor=1):
        path = Path(path) if path else self.screenshots_dir / f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png"
        self.log(f"screenshot {path}")
        return path

    def open_url(self, url):
        self.log(f"open_url {url}")
        return self.browser

    def clipboard_get(self):
        return self._clip

    def clipboard_set(self, text):
        self._clip = text

    def calculator_input(self, win, tokens, delay_s=0.0):
        self.focus(win)
        self.log(f"calculator keys {' '.join(t[2:] for t in tokens)} -> {win.title}")

    def type_text(self, text, press_enter=False):
        t = self.keyboard_target()
        self.log(f"type {text!r} -> {t.title if t else '?'}" + (" +enter" if press_enter else ""))
        return t.title if t else None

    def press_keys(self, keys):
        t = self.keyboard_target()
        self.log(f"keys {'+'.join(keys)} -> {t.title if t else '?'}")
        return t.title if t else None

    def mouse_position(self):
        return self._mouse

    def move_mouse(self, x, y, duration=0.2):
        self._mouse = (x, y)
        self.log(f"mouse -> {x},{y}")

    def click(self, x=None, y=None, button="left", clicks=1):
        if x is not None and y is not None:
            self._mouse = (x, y)
        self.log(f"click {button} x{clicks} at {self._mouse}")

    def scroll(self, amount):
        self.log(f"scroll {amount}")


def make_desktop(kind: str = "auto", **kw) -> Desktop:
    if kind == "dry-run":
        return DryRunDesktop(**kw)
    if kind == "windows" or (kind == "auto" and sys.platform == "win32"):
        return WindowsDesktop(**kw)
    if kind == "x11" or (kind == "auto" and os.environ.get("DISPLAY")):
        return X11Desktop(**kw)
    return DryRunDesktop(**kw)
