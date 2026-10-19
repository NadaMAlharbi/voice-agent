"""Fast command router: clear commands skip the LLM entirely."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from voice_agent.tools.apps import normalize


@dataclass
class Routed:
    tool: str
    args: dict = field(default_factory=dict)


# text helpers

_POLITE_EN = re.compile(r"^(?:hey |ok |okay )?(?:HAZEM[,.]?\s+)?(?:please\s+|can you\s+|could you\s+|would you\s+|"
                        r"i want you to\s+|go ahead and\s+)*", re.I)
_POLITE_EN_TAIL = re.compile(r"(?:,?\s*(?:please|for me|right now|now|HAZEM))+$", re.I)
_POLITE_AR = re.compile(r"^(?:يا\s+)?(?:حزم|حزم)?\s*[،,]?\s*(?:(?:لو سمحت|من فضلك|ممكن|رجاء|ابغاك|ابيك|اريدك ان|"
                        r"اريد ان|ابي|ابغي)\s+)*")
_POLITE_AR_TAIL = re.compile(r"(?:\s+(?:لو سمحت|من فضلك|يا حزم|الحين|الان|لو تكرمت))+$")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty five": 45, "sixty": 60,
    "half": 0.5, "واحد": 1, "واحده": 1, "اثنين": 2, "اثنتين": 2, "ثلاث": 3, "ثلاثه": 3, "اربع": 4, "اربعه": 4,
    "خمس": 5, "خمسه": 5, "سته": 6, "ست": 6, "سبع": 7, "سبعه": 7, "ثمان": 8, "ثمانيه": 8, "تسع": 9, "تسعه": 9,
    "عشر": 10, "عشره": 10, "ربع": 0.25, "نص": 0.5, "نصف": 0.5, "خمسطعش": 15, "خمسه عشر": 15, "عشرين": 20,
    "ثلاثين": 30,
}


def _strip_polite(text: str, tail: bool = True) -> str:
    """tail=False keeps trailing words (text to type must stay exactly as dictated)."""
    t = text.strip().strip(".!?؟،,").strip()
    t = _POLITE_EN.sub("", t)
    t = _POLITE_AR.sub("", t)
    if tail:
        t = _POLITE_EN_TAIL.sub("", t)
        t = _POLITE_AR_TAIL.sub("", t)
    return t.strip().strip(".!?؟،,").strip()


def _num(s: str) -> float | None:
    s = s.strip().translate(_AR_DIGITS)
    try:
        return float(s)
    except ValueError:
        return NUM_WORDS.get(normalize(s))


# yes / no / sleep / wake

_YES = {"yes", "yeah", "yep", "yup", "sure", "of course", "do it", "go ahead", "confirm", "confirmed", "ok", "okay",
        "please do", "yes please", "correct", "affirmative", "proceed",
        "نعم", "ايوه", "ايوا", "اي", "ايه", "اجل", "تمام", "اكيد", "موافق", "نفذ", "طيب", "بلى", "يب", "اوكي", "ماشي",
        "نعم من فضلك", "اي نعم", "اكيد نفذ", "سوها", "سويها", "افعل", "افعلها"}
_NO = {"no", "nope", "nah", "cancel", "don't", "do not", "stop", "never mind", "nevermind", "abort", "no thanks",
       "لا", "لاء", "لأ", "الغ", "الغي", "الغاء", "لا تفعل", "لا تسوي", "خلاص لا", "لا شكرا", "توقف", "بلاش", "لا لا"}


def yes_no(text: str) -> bool | None:
    n = normalize(_strip_polite(text))
    n = re.sub(r"^(?:HAZEM|حزم|حزم)\s+", "", n)
    if n in {normalize(w) for w in _YES}:
        return True
    if n in {normalize(w) for w in _NO}:
        return False
    first = n.split(" ")[0] if n else ""
    if first in ("yes", "yeah", "sure", "نعم", "اكيد", "ايوه", "اجل") and len(n.split()) <= 4:
        return True
    if first in ("no", "nope", "لا", "الغ") and len(n.split()) <= 4:
        return False
    return None


_SLEEP = re.compile(r"^(?:go to sleep|sleep|sleep mode|stop listening|go quiet|نام|نم|وضع السكون|توقف عن الاستماع|"
                    r"روح نام|اذهب للنوم)$")
_WAKE = re.compile(r"(?:\bHAZEM\b|حزم|حزم|حزم|\bwake up\b|استيقظ|اصحي|اصحى)", re.I)


def is_sleep(text: str) -> bool:
    return bool(_SLEEP.match(normalize(_strip_polite(text))))


def is_wake(text: str) -> bool:
    return bool(_WAKE.search(text.lower()) or _WAKE.search(normalize(text)))


# commands

EN_RULES: list[tuple[re.Pattern, callable]] = []
AR_RULES: list[tuple[re.Pattern, callable]] = []


def rule(rules, pattern):
    def deco(fn):
        rules.append((re.compile(pattern, re.I), fn))
        return fn
    return deco


def first_rule(rules, pattern):
    # Same as rule(), but checked before the generic open/close/search patterns.
    def deco(fn):
        rules.insert(len(_FIRST.setdefault(id(rules), [])), (re.compile(pattern, re.I), fn))
        _FIRST[id(rules)].append(fn)
        return fn
    return deco


_FIRST: dict[int, list] = {}
_UNITS_EN = r"(?P<u>seconds?|secs?|minutes?|mins?|hours?)"
_UNITS_AR = r"(?P<u>ثانيه|ثواني|ثوان|دقيقه|دقايق|دقائق|دقيقتين|ساعه|ساعات|ساعتين)"


def _seconds(n: str | None, unit: str) -> int | None:
    # This function turns "ten" + "minutes" (or "خمس" + "دقائق") into seconds.
    if unit in ("دقيقتين", "ساعتين"):
        k = 2
    else:
        k = _num(n) if n else 1
    if k is None:
        return None
    return int(k * (3600 if unit.startswith(("hour", "ساع")) else 60 if unit.startswith(("min", "دق")) else 1))


def _site_from(word: str | None) -> str:
    w = normalize(word or "")
    if w in ("youtube", "يوتيوب", "اليوتيوب"):
        return "youtube"
    if w in ("maps", "google maps", "الخرائط", "خرائط جوجل", "خرائط"):
        return "maps"
    if w in ("wikipedia", "ويكيبيديا"):
        return "wikipedia"
    if w in ("images", "صور", "الصور"):
        return "images"
    return "google"


def _search(query: str, site: str, lang: str) -> Routed:
    # This function picks YouTube search or Google search.
    if site == "youtube":
        return Routed("youtube_search", {"query": query})
    return Routed("web_search", {"query": query, "site": site, "language": lang})


# English

@first_rule(EN_RULES, r"^remind me (?:in |after )?(?P<n>[\w.]+(?: [\w]+)?) " + _UNITS_EN + r" (?:to |that |about )?(?P<w>.+)$")
def _en_remind_in(m, t):
    secs = _seconds(m["n"], m["u"].lower())
    what = _orig_after(r"(?:minutes?|mins?|seconds?|secs?|hours?)\s+(?:to |that |about )?(.+)$") or m["w"]
    return Routed("set_timer", {"seconds": secs, "label": what}) if secs else None


@first_rule(EN_RULES, r"^remind me (?:to |that |about )?(?P<w>.+?) in (?P<n>[\w.]+(?: [\w]+)?) " + _UNITS_EN + r"$")
def _en_remind_to(m, t):
    secs = _seconds(m["n"], m["u"].lower())
    what = _orig_after(r"remind me (?:to |that |about )?(.+?) in \S+(?: \S+)? (?:minutes?|mins?|seconds?|secs?|hours?)$")
    return Routed("set_timer", {"seconds": secs, "label": what or m["w"]}) if secs else None


@first_rule(EN_RULES, r"^set (?:a |an )?(?:reminder|alarm) (?:for |in )?(?P<n>[\w.]+(?: [\w]+)?) " + _UNITS_EN
            + r"(?: (?:to |that |about )?(?P<w>.+))?$")
def _en_reminder(m, t):
    secs = _seconds(m["n"], m["u"].lower())
    return Routed("set_timer", {"seconds": secs, "label": m["w"] or "reminder"}) if secs else None


@first_rule(EN_RULES, r"^(?:remember|memorize) (?:that )?(?P<f>.+)$")
def _en_remember(m, t):
    return Routed("remember", {"fact": _orig_after(r"(?:remember|memorize)\s+(?:that\s+)?(.+)$") or m["f"]})


@first_rule(EN_RULES, r"^(?:save|take|make|add|write|create)(?: a| this)? note(?: that| saying)?:?\s*(?P<x>.+)$|"
                      r"^note(?: that| down)\s+(?P<x2>.+)$")
def _en_note(m, t):
    x = _orig_after(r"note(?:\s+that|\s+saying|\s+down)?\s*:?\s*(.+)$") or m["x"] or m["x2"]
    return Routed("add_note", {"text": x})


@first_rule(EN_RULES, r"^(?:read|show|what are) (?:me )?(?:my )?(?:latest |last )?notes$")
def _en_read_notes(m, t):
    return Routed("read_notes", {})


@first_rule(EN_RULES, r"^(?:turn|crank|bring|put) (?:the )?(?:volume|sound) (?P<d>up|down)(?: a (?:bit|little))?$|"
                      r"^(?:turn|bring) (?P<d2>up|down) the (?:volume|sound)(?: a (?:bit|little))?$|"
                      r"^(?P<i>increase|raise|lower|decrease|reduce) (?:the )?(?:volume|sound)(?: a (?:bit|little))?$|"
                      r"^(?:volume|sound) (?P<d3>up|down)(?: a (?:bit|little))?$")
def _en_volume_more(m, t):
    d = m["d"] or m["d2"] or m["d3"] or ("up" if m["i"].lower() in ("increase", "raise") else "down")
    return Routed("volume", {"action": d.lower()})


@first_rule(EN_RULES, r"^(?:what(?:'s| is| do you see| can you see) on (?:my|the) screen|describe (?:my|the) screen)"
                      r"(?: right now| now)?$")
def _en_describe_now(m, t):
    return Routed("describe_screen", {"question": t, "language": "en"})


@first_rule(EN_RULES, r"^(?:open|show|show me|go to) (?:my |the )?(?:google )?calendar$")
def _en_calendar(m, t):
    return Routed("open_calendar", {})


@rule(EN_RULES, r"^(?:what(?:'s| is) the time|what time is it|tell me the time|(?:the )?time please|"
                r"what(?:'s| is) the time now)(?: now| right now)?$")
def _en_time(m, t):
    return Routed("get_date_time", {"ask": "time"})


@rule(EN_RULES, r"^(?:what(?:'s| is) (?:the |today'?s )?date(?: today)?|what(?:'s| is) today'?s date|"
                r"what(?:'s| is) the date today|what date is it(?: today)?)$")
def _en_date(m, t):
    return Routed("get_date_time", {"ask": "date"})


@rule(EN_RULES, r"^what day is (?:it )?(?:today)?$|^what day is it$")
def _en_day(m, t):
    return Routed("get_date_time", {"ask": "day"})


@rule(EN_RULES, r"^(?:what(?:'s| is) the |how(?:'s| is) the )?weather(?: like)?(?: (?:in|for|at) (?P<city>.+?))?"
                r"(?: today| now| right now)?$")
def _en_weather(m, t):
    return Routed("get_weather", {"city": m["city"] or ""})


@rule(EN_RULES, r"^(?:cancel|abort|stop) (?:the )?(?:shutdown|restart)$")
def _en_power_cancel(m, t):
    return Routed("power", {"action": "cancel"})


@rule(EN_RULES, r"^(?P<a>shut ?down|turn off|restart|reboot) (?:the |my )?(?:computer|pc|laptop|system)$")
def _en_power(m, t):
    return Routed("power", {"action": "restart" if m["a"].lower() in ("restart", "reboot") else "shutdown"})


@rule(EN_RULES, r"^(?:take|capture|grab|save)\s+(?:a\s+|the\s+)?screen\s?shot(?:\s+of (?:my |the )?screen)?$|"
                r"^screen\s?shot$|^take a picture of (?:my|the) screen$")
def _en_shot(m, t):
    return Routed("take_screenshot")


@rule(EN_RULES, r"^(?:what(?:'s| is| do you see| can you see) on (?:my|the) screen|describe (?:my|the) screen|"
                r"look at (?:my|the) screen|read (?:my|the) screen(?: to me)?|what am i looking at|"
                r"what(?:'s| is) this (?:window|page|app))$")
def _en_describe(m, t):
    return Routed("describe_screen", {"question": t, "language": "en"})


@rule(EN_RULES, r"^(?:search|google|look up)\s+(?:on\s+)?(?:(?P<site1>google|youtube|chrome|the web|the internet|"
                r"wikipedia|maps)\s+)?(?:for\s+)?(?P<q>.+?)(?:\s+on\s+(?P<site2>google|youtube|chrome|wikipedia|"
                r"maps))?$")
def _en_search(m, t):
    return _search(m["q"], _site_from(m["site1"] or m["site2"]), "en")


@rule(EN_RULES, r"^play\s+(?P<q>.+?)\s+on\s+youtube$")
def _en_yt(m, t):
    return Routed("youtube_search", {"query": m["q"]})


@rule(EN_RULES, r"^press\s+(?:the\s+)?(?P<k>.+?)(?:\s+key)?$")
def _en_press(m, t):
    return Routed("press_keys", {"keys": re.sub(r"\s+(?:and|plus)\s+|\s+", "+", m["k"].strip())})


@rule(EN_RULES, r"^(?:turn (?:the )?(?:volume|sound) (?P<d1>up|down)|(?:volume|sound) (?P<d2>up|down)|"
                r"(?P<l>louder|quieter|make it louder|make it quieter)|(?P<mu>mute|unmute)(?: the)?"
                r"(?: sound| volume| audio| computer)?)$")
def _en_volume(m, t):
    if m["mu"]:
        return Routed("volume", {"action": "mute"})
    d = m["d1"] or m["d2"] or ("up" if "louder" in (m["l"] or "") else "down")
    return Routed("volume", {"action": d})


@rule(EN_RULES, r"^(?:(?P<p>pause|resume)(?: the)?(?: music| song| video| playback)?|"
                r"stop (?:the )?(?:music|song|video|playback)|play music|"
                r"(?P<n>next|skip)(?: the)?(?: song| track| video)?|(?P<b>previous|last)(?: song| track| video))$")
def _en_media(m, t):
    if m["n"]:
        return Routed("media", {"action": "next"})
    if m["b"]:
        return Routed("media", {"action": "previous"})
    return Routed("media", {"action": "play_pause"})


@rule(EN_RULES, r"^lock (?:the |my )?(?:screen|computer|pc|laptop)$")
def _en_lock(m, t):
    return Routed("lock_screen")


@rule(EN_RULES, r"^(?:system status|how(?:'s| is) (?:my|the) (?:computer|pc|system|laptop)(?: doing)?|"
                r"(?:what(?:'s| is) (?:my |the )?)?(?:cpu|memory|ram) usage|(?:what(?:'s| is) (?:my |the )?)?"
                r"battery(?: level| status)?|how much battery(?: do i have)?(?: left)?)$")
def _en_status(m, t):
    return Routed("system_status")


@rule(EN_RULES, r"^(?:set |start )?(?:a )?timer (?:for )?(?P<n>[\w.]+(?: [\w]+)?)\s+(?P<u>seconds?|minutes?|hours?)$|"
                r"^(?:set |start )?(?:a )?(?P<n2>[\w.]+) (?P<u2>seconds?|minutes?|hours?) timer$")
def _en_timer(m, t):
    n, u = _num(m["n"] or m["n2"]), (m["u"] or m["u2"])
    if n is None:
        return None
    secs = n * (3600 if u.startswith("hour") else 60 if u.startswith("min") else 1)
    return Routed("set_timer", {"seconds": int(secs)})


@rule(EN_RULES, r"^(?P<a>minimi[sz]e|maximi[sz]e|restore)\s+(?:the\s+)?(?P<name>.+?)(?:\s+window)?$")
def _en_winstate(m, t):
    a = m["a"].lower().replace("s", "z") if m["a"].lower().startswith(("minimis", "maximis")) else m["a"].lower()
    return Routed("window_action", {"name": m["name"], "action": a})


@rule(EN_RULES, r"^(?:switch to|go to|focus(?: on)?|bring up|show me)\s+(?:the\s+)?(?P<name>.+?)(?:\s+window)?$")
def _en_focus(m, t):
    return Routed("focus_window", {"name": m["name"]})


@rule(EN_RULES, r"^(?:close|quit|exit|kill)\s+(?:the\s+)?(?P<name>.+?)(?:\s+(?:app|application|window))?$")
def _en_close(m, t):
    return Routed("close_app", {"name": m["name"]})


@rule(EN_RULES, r"^(?:open|launch|start|run|fire up|load)\s+(?:up\s+)?(?:the\s+)?(?P<name>.+?)$")
def _en_open(m, t):
    return Routed("open_app", {"name": m["name"]})


# Arabic

@first_rule(AR_RULES, r"^ذكرني\s+(?:بعد|خلال)\s+(?:(?P<n>[\d.٠-٩]+|\S+)\s+)?" + _UNITS_AR + r"\s+(?:ان\s+|ب)?(?P<w>.+)$")
def _ar_remind(m, t):
    secs = _seconds(m["n"], m["u"])
    what = _orig_after(r"(?:دقيقة|دقيقه|دقائق|دقايق|دقيقتين|ثانية|ثانيه|ثواني|ثوان|ساعة|ساعه|ساعات|ساعتين)\s+"
                       r"(?:[أا]ن\s+)?(.+)$") or m["w"]
    return Routed("set_timer", {"seconds": secs, "label": what}) if secs else None


@first_rule(AR_RULES, r"^(?:تذكر|احفظ في ذاكرتك)\s+(?:ان\s+)?(?P<f>.+)$")
def _ar_remember(m, t):
    return Routed("remember", {"fact": _orig_after(r"(?:تذكر|ذاكرتك)\s+(?:[أا]ن\s+)?(.+)$") or m["f"]})


@first_rule(AR_RULES, r"^(?:احفظ|سجل|اكتب|دون)\s+(?:لي\s+)?ملاحظه\s*:?\s*(?P<x>.+)$")
def _ar_note(m, t):
    return Routed("add_note", {"text": _orig_after(r"ملاحظ[ةه]\s*:?\s*(.+)$") or m["x"]})


@first_rule(AR_RULES, r"^(?:اقرا|اقرأ|اعرض)\s+(?:لي\s+)?(?:ال)?ملاحظات(?:ي)?$|^(?:ما|وش|ايش)\s+(?:هي\s+)?ملاحظاتي$")
def _ar_read_notes(m, t):
    return Routed("read_notes", {})


@first_rule(AR_RULES, r"^(?P<v>ارفع|علي|زود|عل|اخفض|وطي|نزل|قلل|خفض)\s+(?:ال)?صوت(?:\s+(?:شوي|شويه|قليلا|قليل|اكثر|كمان))*$")
def _ar_volume_more(m, t):
    return Routed("volume", {"action": "up" if m["v"] in ("ارفع", "علي", "زود", "عل") else "down"})


@first_rule(AR_RULES, r"^(?:ماذا|ما الذي|ما|ايش|وش|شو)\s+(?:(?:يوجد|يظهر|تري|تشوف|في|على|علي)\s+)*(?:ال)?شاش(?:ه|تي)"
                      r"(?:\s+(?:الان|الحين))?$|^(?:صف|اوصف)\s+(?:لي\s+)?(?:ال)?شاش(?:ه|تي)$")
def _ar_describe_more(m, t):
    return Routed("describe_screen", {"question": _ORIGINAL["text"], "language": "ar"})


@first_rule(AR_RULES, r"^كم\s+(?:ال)?وقت(?:\s+(?:الان|الحين))?$")
def _ar_time_more(m, t):
    return Routed("get_date_time", {"ask": "time"})


@first_rule(AR_RULES, r"^(?:افتح|اعرض)\s+(?:لي\s+)?(?:ال)?تقويم(?:\s+(?:جوجل|قوقل))?$")
def _ar_calendar(m, t):
    return Routed("open_calendar", {})


@rule(AR_RULES, r"^(?:كم\s+)?(?:ال)?ساعه(?:\s+كم)?(?:\s+(?:الان|الحين|هلا))?$|^(?:وش|ايش|ما)\s+(?:ال)?ساعه(?:\s+(?:الان|الحين))?$")
def _ar_time(m, t):
    return Routed("get_date_time", {"ask": "time"})


@rule(AR_RULES, r"^(?:ما|وش|ايش|كم)?\s*(?:هو\s+)?(?:ال)?تاريخ(?:\s+(?:اليوم|اليوم كم))?$|^(?:ما|وش|ايش)\s+تاريخ\s+اليوم$")
def _ar_date(m, t):
    return Routed("get_date_time", {"ask": "date"})


@rule(AR_RULES, r"^(?:اي|ايش|وش|ما)\s+(?:هو\s+)?(?:ال)?يوم(?:\s+اليوم)?$|^اليوم\s+(?:وش|ايش|ما)$")
def _ar_day(m, t):
    return Routed("get_date_time", {"ask": "day"})


@rule(AR_RULES, r"^(?:كيف|ما|وش|ايش)?\s*(?:هو\s+)?(?:ال)?(?:طقس|جو)(?:\s+(?:في|ب)\s*(?P<city>.+?))?(?:\s+(?:اليوم|الان))?$")
def _ar_weather(m, t):
    return Routed("get_weather", {"city": m["city"] or ""})


@rule(AR_RULES, r"^(?:الغ|الغي|الغاء|اوقف|وقف)\s+(?:ال)?(?:ايقاف|اطفاء|اعاده التشغيل)(?:\s+(?:ال)?تشغيل)?$")
def _ar_power_cancel(m, t):
    return Routed("power", {"action": "cancel"})


@rule(AR_RULES, r"^(?P<a>اطفئ|اطفي|طفي|اغلق|اعد تشغيل|ريستارت)\s+(?:ال)?(?:جهاز|كمبيوتر|حاسوب|لابتوب)$")
def _ar_power(m, t):
    return Routed("power", {"action": "restart" if m["a"] in ("اعد تشغيل", "ريستارت") else "shutdown"})


@rule(AR_RULES, r"^(?:خذ|التقط|اخذ|سو|سوي|اعمل|احفظ)\s+(?:لي\s+)?(?:لقطه|صوره|سكرين ?شوت|سكرين)"
                r"(?:\s+(?:لل?شاشه|شاشه))?$|^(?:لقطه شاشه|سكرين ?شوت|صور الشاشه)$")
def _ar_shot(m, t):
    return Routed("take_screenshot")


@rule(AR_RULES, r"^(?:ماذا|ما الذي|ايش|وش|شو)\s+(?:يوجد\s+|يظهر\s+|تري\s+|في\s+|على\s+|علي\s+)*(?:على\s+|علي\s+|في\s+)?"
                r"(?:ال)?شاشه(?:\s+الان)?$|^(?:صف|اوصف|اقرا|اقرأ)\s+(?:لي\s+)?(?:ال)?شاشه$|^انظر (?:الي|الى) الشاشه$")
def _ar_describe(m, t):
    return Routed("describe_screen", {"question": t, "language": "ar"})


@rule(AR_RULES, r"^(?:ابحث|دور|فتش|بحث)\s+(?:لي\s+)?(?:(?:في|على|علي)\s+(?P<site>جوجل|قوقل|كروم|النت|الانترنت|"
                r"يوتيوب|اليوتيوب|ويكيبيديا|الخرائط)\s+)?(?:عن\s+|على\s+|علي\s+)?(?P<q>.+?)(?:\s+(?:في|على|علي)\s+(?P<site2>جوجل|قوقل|"
                r"يوتيوب|اليوتيوب|ويكيبيديا|الخرائط))?$")
def _ar_search(m, t):
    return _search(m["q"], _site_from(m["site"] or m["site2"]), "ar")


@rule(AR_RULES, r"^(?:ارفع|علي|زود|عل)\s+(?:ال)?صوت$|^(?:اخفض|وطي|نزل|قلل|خفض)\s+(?:ال)?صوت$|^(?:اكتم|اسكت|كتم)\s+"
                r"(?:ال)?صوت$|^(?:ال)?صوت\s+(?:اعلي|اقل|اكثر)$")
def _ar_volume(m, t):
    if re.match(r"^(?:اكتم|اسكت|كتم)", t):
        return Routed("volume", {"action": "mute"})
    up = re.match(r"^(?:ارفع|علي|زود|عل)|.*(?:اعلي|اكثر)$", t)
    return Routed("volume", {"action": "up" if up else "down"})


@rule(AR_RULES, r"^(?:(?P<p>اوقف|وقف|شغل|كمل)\s+(?:ال)?(?:موسيقي|اغنيه|اغاني|فيديو|مقطع)|(?:ال)?(?P<n>اغنيه|مقطع)\s+"
                r"(?P<d>التالي|التاليه|السابق|السابقه))$")
def _ar_media(m, t):
    if m["d"]:
        return Routed("media", {"action": "next" if m["d"].startswith("التالي") else "previous"})
    return Routed("media", {"action": "play_pause"})


@rule(AR_RULES, r"^(?:اقفل|قفل|اغلق)\s+(?:ال)?(?:شاشه|جهاز|كمبيوتر|حاسوب|لابتوب)$")
def _ar_lock(m, t):
    return Routed("lock_screen")


@rule(AR_RULES, r"^(?:حاله|وضع)\s+(?:ال)?(?:جهاز|نظام|كمبيوتر|حاسوب)$|^(?:كم\s+)?(?:نسبه\s+)?(?:ال)?بطاريه(?:\s+كم)?$|"
                r"^(?:كم\s+)?استهلاك\s+(?:ال)?(?:معالج|ذاكره|رام)$")
def _ar_status(m, t):
    return Routed("system_status")


@rule(AR_RULES, r"^(?:اضبط\s+|شغل\s+|ابدا\s+|سو\s+)?(?:لي\s+)?(?:مؤقت|موقت|منبه|تايمر)\s+(?:لمده\s+|ل|على\s+|علي\s+)?"
                r"(?P<n>[\d.٠-٩]+|\S+)?\s*(?P<u>ثانيه|ثواني|ثوان|دقيقه|دقايق|دقائق|دقيقتين|ساعه|ساعات|ساعتين)$")
def _ar_timer(m, t):
    u = m["u"]
    n = 2 if u in ("دقيقتين", "ساعتين") else (_num(m["n"]) if m["n"] else 1)
    if n is None:
        return None
    secs = n * (3600 if u.startswith("ساع") else 60 if u.startswith("دق") else 1)
    return Routed("set_timer", {"seconds": int(secs)})


@rule(AR_RULES, r"^(?P<a>صغر|كبر|ارجع)\s+(?:نافذه\s+)?(?P<name>.+)$")
def _ar_winstate(m, t):
    return Routed("window_action", {"name": m["name"], "action": {"صغر": "minimize", "كبر": "maximize"}.get(
        m["a"], "restore")})


@rule(AR_RULES, r"^(?:انتقل|روح|حول|بدل|اذهب)\s+(?:الي|الى|على|علي|ل)\s+(?P<name>.+)$")
def _ar_focus(m, t):
    return Routed("focus_window", {"name": m["name"]})


@rule(AR_RULES, r"^(?:اغلق|اقفل|سكر|قفل)\s+(?:لي\s+)?(?P<name>.+)$")
def _ar_close(m, t):
    return Routed("close_app", {"name": m["name"]})


@rule(AR_RULES, r"^(?:افتح|افتحي|افتحلي|شغل|شغلي|شغللي)\s+(?:لي\s+)?(?P<name>.+)$")
def _ar_open(m, t):
    return Routed("open_app", {"name": m["name"]})


_EN_TYPE = re.compile(r"^(?:type|dictate)\s+(?:out\s+)?(?P<t>.+)$", re.I)
_AR_TYPE = re.compile(r"^(?:اطبع|اكتب\s+(?:في|على|علي|هنا|بالشاشه|في الشاشه)\s*:?)\s+(?P<t>.+)$")


_COMPOUND = re.compile(r"\b(?:and|then|after that|and then|but)\b|,|\s(?:ثم|بعدين|بعدها)\s|"
                       r"\sو(?:اكتب|ابحث|افتح|اطبع|خذ|شغل|اغلق|اقفل|ارفع|انسخ|الصق)", re.I)
_NAMED = {"open_app", "close_app", "focus_window", "window_action"}


def route(text: str, lang: str) -> Routed | None:
    """A Routed command for clear commands, else None (the LLM decides).

    Compound requests ("open notepad and type hello", "افتح المفكرة واكتب") are left
    to the LLM, which can chain several tools.
    """
    r = _route(text, lang)
    if r is not None and r.tool in _NAMED and _COMPOUND.search(r.args.get("name", "")):
        return None
    return r


_ORIGINAL = {"text": ""}


def _orig_after(pattern: str) -> str | None:
    # This function returns the user's own words after a keyword (not the normalized text).
    m = re.search(pattern, _ORIGINAL["text"], re.I)
    return m.group(1).strip().strip(".!?؟،,:").strip() if m else None


def _route(text: str, lang: str) -> Routed | None:
    _ORIGINAL["text"] = _strip_polite(text, tail=False)
    t = _strip_polite(text)
    if not t:
        return None
    head = _strip_polite(text, tail=False)        # typing keeps every dictated word, as spelled
    arabic = lang == "ar" or re.search(r"[\u0600-\u06FF]", t)
    m = (_AR_TYPE.match(re.sub("^[\u0623\u0625]", "\u0627", head)) if arabic else None) or _EN_TYPE.match(head)
    if m:
        return Routed("type_text", {"text": m["t"].strip().strip('"')})
    if arabic:
        n = normalize(t)
        for pat, fn in AR_RULES:
            m = pat.match(n)
            if m:
                r = fn(m, n)
                if r:
                    return r
        return None
    n = re.sub(r"\s+", " ", t)
    for pat, fn in EN_RULES:
        m = pat.match(n)
        if m:
            r = fn(m, n)
            if r:
                return r
    return None
