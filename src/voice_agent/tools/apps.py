"""Spoken app names (English or Arabic) -> an installed app.

"open notepad", "افتح المفكرة", "open google chrome", "شغل الحاسبة" must all
reach the right program."""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class AppEntry:
    name: str          # display name ("Google Chrome")
    target: str        # what the backend launches (path, AppID, command)
    kind: str = "cmd"  # backend specific: lnk | appid | cmd | uri | desktop


# canonical key -> spoken aliases (EN + AR, incl. common STT spellings)
ALIASES: dict[str, list[str]] = {
    "notepad": ["notepad", "note pad", "notes app", "text editor", "المفكرة", "مفكرة", "النوت باد", "نوت باد",
                "المحرر", "محرر النصوص", "mousepad"],
    "calculator": ["calculator", "calc", "الحاسبة", "حاسبة", "الآلة الحاسبة", "الاله الحاسبة", "كالكوليتر"],
    "chrome": ["chrome", "google chrome", "the browser", "browser", "كروم", "جوجل كروم", "قوقل كروم", "المتصفح",
               "متصفح", "chromium"],
    "edge": ["edge", "microsoft edge", "ايدج", "إيدج", "مايكروسوفت ايدج"],
    "firefox": ["firefox", "فايرفوكس"],
    "explorer": ["file explorer", "explorer", "files", "my files", "مستكشف الملفات", "الملفات", "مدير الملفات"],
    "settings": ["settings", "windows settings", "الإعدادات", "الاعدادات", "إعدادات"],
    "paint": ["paint", "ms paint", "الرسام", "برنامج الرسام", "الرسم"],
    "cmd": ["command prompt", "cmd", "terminal", "موجه الأوامر", "سطر الأوامر", "الطرفية", "xterm"],
    "powershell": ["powershell", "power shell", "باورشيل", "بورشيل"],
    "taskmgr": ["task manager", "مدير المهام"],
    "control": ["control panel", "لوحة التحكم"],
    "snippingtool": ["snipping tool", "أداة القص", "اداة القص"],
    "word": ["word", "microsoft word", "وورد", "الوورد"],
    "excel": ["excel", "microsoft excel", "اكسل", "إكسل", "الاكسل"],
    "powerpoint": ["powerpoint", "power point", "باوربوينت", "بوربوينت"],
    "outlook": ["outlook", "اوتلوك", "أوتلوك"],
    "vscode": ["vs code", "visual studio code", "vscode", "code", "في اس كود", "فيجوال ستوديو كود"],
    "spotify": ["spotify", "سبوتيفاي"],
    "whatsapp": ["whatsapp", "واتساب", "الواتساب", "واتس اب"],
    "telegram": ["telegram", "تيليجرام", "تلغرام", "تليجرام"],
    "teams": ["teams", "microsoft teams", "تيمز"],
    "discord": ["discord", "ديسكورد"],
    "obs": ["obs", "obs studio"],
    "steam": ["steam", "ستيم"],
    "vlc": ["vlc", "vlc player", "في ال سي"],
    "camera": ["camera", "الكاميرا", "كاميرا"],
    "photos": ["photos", "الصور"],
    "xeyes": ["xeyes", "eyes"],         
}

# canonical key -> name fragments of the installed entry it should match
INSTALLED_HINTS: dict[str, list[str]] = {
    "notepad": ["notepad", "mousepad"], "calculator": ["calculator", "calc"], "chrome": ["google chrome", "chrome"],
    "edge": ["microsoft edge"], "explorer": ["file explorer"], "paint": ["paint"], "cmd": ["command prompt", "xterm"],
    "powershell": ["windows powershell", "powershell"], "taskmgr": ["task manager"], "control": ["control panel"],
    "snippingtool": ["snipping tool"], "word": ["word"], "excel": ["excel"], "powerpoint": ["powerpoint"],
    "outlook": ["outlook"], "vscode": ["visual studio code"], "settings": ["settings"],
}

_FILLER = re.compile(
    r"^(?:the|a|an|my|app|application|program|programme|برنامج|تطبيق|ال?برنامج|ال?تطبيق)\s+|"
    r"\s+(?:app|application|program|please|for me|now|برنامج|تطبيق|لو سمحت|من فضلك|الآن|لي)$",
    re.IGNORECASE,
)


def normalize(s: str) -> str:
    """Lowercase, drop diacritics/tatweel, unify alef/ya/ta-marbuta, strip punctuation."""
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = re.sub(r"[ً-ْٰـ]", "", s)             
    s = s.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"}))
    s = re.sub(r"[^\w\s+.#-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    prev = None
    while prev != s:
        prev, s = s, _FILLER.sub("", s).strip()
    return s


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    short, long_ = sorted((a, b), key=len)
    extra = len(long_.split()) - len(short.split())
    if f" {short} " in f" {long_} " and extra <= 2:              # whole-word containment, few extra words
        r = max(r, 0.86 if len(short) >= 4 else 0.75)
    return r


@dataclass
class Match:
    entry: AppEntry
    score: float
    canonical: str | None


class AppResolver:
    """Matches spoken names against the installed apps. Thread-safe for reads."""

    def __init__(self, entries: list[AppEntry], builtins: dict[str, AppEntry] | None = None,
                 min_score: float = 0.78):
        self.entries = entries
        self.builtins = builtins or {}
        self.min_score = min_score
        self._norm = [(normalize(e.name), e) for e in entries]

    def canonical(self, spoken: str) -> tuple[str | None, float]:
        n = normalize(spoken)
        best, score = None, 0.0
        for key, names in ALIASES.items():
            for alias in names:
                r = _ratio(n, normalize(alias))
                if r > score:
                    best, score = key, r
        return best, score

    def resolve(self, spoken: str) -> Match | None:
        n = normalize(spoken)
        if not n:
            return None
        key, kscore = self.canonical(spoken)
        if key and kscore >= self.min_score:
            # prefer an installed entry for that app, then the built-in command
            for hint in INSTALLED_HINTS.get(key, [key]):
                exact = [e for ne, e in self._norm if ne == hint]
                if exact:
                    return Match(exact[0], kscore, key)
            for hint in INSTALLED_HINTS.get(key, [key]):
                part = [e for ne, e in self._norm if f" {hint} " in f" {ne} " and "uninstall" not in ne]
                if part:
                    return Match(min(part, key=lambda e: len(e.name)), kscore, key)
            if key in self.builtins:
                return Match(self.builtins[key], kscore, key)
        best, score = None, 0.0
        for ne, e in self._norm:
            if "uninstall" in ne:
                continue
            r = _ratio(n, ne)
            if r > score or (r == score and best and len(e.name) < len(best.name)):
                best, score = e, r
        if best and score >= self.min_score:
            return Match(best, score, None)
        return None
