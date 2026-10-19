"""Windows keyboard input that works with any keyboard layout.

pyautogui types through the current layout, so with the Arabic layout active "hello"
would come out as Arabic letters and ctrl+c may not work. Here:
  text -> Unicode key events (any character, no clipboard needed)
  keys -> virtual-key codes (the same on every layout)

The event builders are plain functions (tested on any OS); only send() needs Windows.
"""

from __future__ import annotations

import ctypes

# fixed-size types matching Win32 (c_long is 8 bytes on Linux but 4 on Windows)
WORD, DWORD, LONG, UINT = ctypes.c_uint16, ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32

KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE = 0x1, 0x2, 0x4
INPUT_KEYBOARD = 1


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", WORD), ("wScan", WORD), ("dwFlags", DWORD), ("time", DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", LONG), ("dy", LONG), ("mouseData", DWORD), ("dwFlags", DWORD),
                ("time", DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", DWORD), ("wParamL", WORD), ("wParamH", WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", DWORD), ("u", _INPUTUNION)]


VK = {
    "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "shift": 0x10, "ctrl": 0x11, "alt": 0x12, "pause": 0x13,
    "capslock": 0x14, "escape": 0x1B, "space": 0x20, "pageup": 0x21, "pagedown": 0x22, "end": 0x23,
    "home": 0x24, "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28, "printscreen": 0x2C, "insert": 0x2D,
    "delete": 0x2E, "win": 0x5B, "apps": 0x5D, "volumemute": 0xAD, "volumedown": 0xAE, "volumeup": 0xAF,
    "nexttrack": 0xB0, "prevtrack": 0xB1, "stop": 0xB2, "playpause": 0xB3,
    "plus": 0xBB, "comma": 0xBC, "minus": 0xBD, "period": 0xBE,
    # number pad keys: the same on every keyboard layout (the calculator tool uses them)
    "multiply": 0x6A, "add": 0x6B, "subtract": 0x6D, "decimal": 0x6E, "divide": 0x6F,
    **{f"f{i}": 0x6F + i for i in range(1, 25)},
}
EXTENDED = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0x5B, 0x5D,
            0xAD, 0xAE, 0xAF, 0xB0, 0xB1, 0xB2, 0xB3}


def vk_of(key: str) -> int:
    k = key.lower()
    if k in VK:
        return VK[k]
    if len(k) == 1 and ("a" <= k <= "z" or "0" <= k <= "9"):
        return ord(k.upper())                 # VK_A..VK_Z, VK_0..VK_9: same on every layout
    raise ValueError(f"unknown key {key!r}")


def _scan(vk: int) -> int:
    """Hardware scan code for a key (Chrome and VS Code read shortcuts from it). 0 off Windows."""
    try:
        return ctypes.windll.user32.MapVirtualKeyW(vk, 0) & 0xFFFF
    except AttributeError:
        return 0


def _key(vk: int, up: bool) -> INPUT:
    flags = (KEYEVENTF_KEYUP if up else 0) | (KEYEVENTF_EXTENDEDKEY if vk in EXTENDED else 0)
    return INPUT(INPUT_KEYBOARD, _INPUTUNION(ki=KEYBDINPUT(vk, _scan(vk), flags, 0, 0)))


def combo_events(keys: list[str]) -> list[INPUT]:
    """['ctrl', 'shift', 't'] -> ctrl down, shift down, t down, t up, shift up, ctrl up."""
    vks = [vk_of(k) for k in keys]
    return [_key(v, False) for v in vks] + [_key(v, True) for v in reversed(vks)]


def text_events(text: str) -> list[INPUT]:
    """Unicode text -> UTF-16 code unit events (newline = Enter)."""
    out: list[INPUT] = []
    for ch in text.replace("\r\n", "\n"):
        if ch == "\n":
            out += [_key(VK["enter"], False), _key(VK["enter"], True)]
            continue
        data = ch.encode("utf-16-le")
        for i in range(0, len(data), 2):
            unit = int.from_bytes(data[i:i + 2], "little")
            for up in (False, True):
                flags = KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0)
                out.append(INPUT(INPUT_KEYBOARD, _INPUTUNION(ki=KEYBDINPUT(0, unit, flags, 0, 0))))
    return out


def send_text(text: str, delay_s: float = 0.004) -> int:
    """Type text one character at a time. Some apps (Windows 11 Notepad, other WinUI apps) drop
    or reorder fast bursts of typed characters, so each character is sent on its own with a
    short pause (~4 ms, so 100 characters take ~0.5 s)."""
    import time

    sent = 0
    for ch in text.replace("\r\n", "\n"):
        sent += send(text_events(ch))
        time.sleep(delay_s)
    return sent


def send(events: list[INPUT], batch: int = 64) -> int:
    """Inject events (Windows only). Returns how many were accepted."""
    user32 = ctypes.windll.user32
    user32.SendInput.argtypes = (UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    user32.SendInput.restype = UINT
    sent = 0
    for i in range(0, len(events), batch):
        part = events[i:i + batch]
        arr = (INPUT * len(part))(*part)
        sent += user32.SendInput(len(part), arr, ctypes.sizeof(INPUT))
    if sent < len(events):
        raise OSError(f"Windows accepted only {sent} of {len(events)} key events "
                      "(the target app may run as administrator; run HAZEM elevated to control it)")
    return sent
