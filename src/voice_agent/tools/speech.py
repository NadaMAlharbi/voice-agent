"""Short spoken sentences for router actions and confirmations (English / Arabic).

Router commands skip the LLM, so their reply is a fixed sentence filled from the
tool result. App names are spoken as the user said them (Arabic name in Arabic).
"""

from __future__ import annotations

from voice_agent.tools.base import ToolResult

ACTION_WORDS = {
    "volume": {"up": ("Volume up.", "رفعت الصوت."), "down": ("Volume down.", "خفضت الصوت."),
               "mute": ("Sound muted.", "تم كتم الصوت.")},
    "media": {"play_pause": ("Done.", "تم."), "next": ("Next track.", "المقطع التالي."),
              "previous": ("Previous track.", "المقطع السابق."), "stop": ("Stopped.", "تم الإيقاف.")},
    "window_action": {"minimize": ("Minimized {n}.", "صغرت {n}."), "maximize": ("Maximized {n}.", "كبرت {n}."),
                      "restore": ("Restored {n}.", "أعدت {n}.")},
}


def _pick(lang: str, en: str, ar: str) -> str:
    return ar if lang == "ar" else en


def _dur(seconds: int, lang: str) -> str:
    if seconds % 3600 == 0:
        h = seconds // 3600
        return _pick(lang, f"{h} hour{'s' if h > 1 else ''}", "ساعة" if h == 1 else ("ساعتين" if h == 2 else f"{h} ساعات"))
    if seconds % 60 == 0:
        m = seconds // 60
        return _pick(lang, f"{m} minute{'s' if m > 1 else ''}",
                     "دقيقة" if m == 1 else ("دقيقتين" if m == 2 else f"{m} دقائق" if m <= 10 else f"{m} دقيقة"))
    if seconds == 1:
        return _pick(lang, "1 second", "ثانية واحدة")
    return _pick(lang, f"{seconds} seconds", f"{seconds} ثانية")


def say(tool: str, args: dict, res: ToolResult, lang: str) -> str:
    """One sentence describing what was done (or why not)."""
    d, n = res.data, args.get("name", "")
    if not res.ok:
        if tool == "open_app":
            return _pick(lang, f"I couldn't find an app called {n}.", f"لم أجد برنامجاً باسم {n}.")
        if tool == "close_app" and res.data.get("asking_to_save"):
            return _pick(lang, f"{n} would not close because it wants to save changes. Please check it.",
                         f"لم يُغلق {n} لأنه يريد حفظ التغييرات. تحقق منه من فضلك.")
        if tool in ("close_app", "focus_window", "window_action"):
            if res.data.get("name") and "front" in res.error:
                return _pick(lang, f"Windows did not let me switch to {n}.", f"لم يسمح ويندوز بالانتقال إلى {n}.")
            return _pick(lang, f"I don't see {n} open.", f"لا أرى {n} مفتوحاً.")
        if tool == "click_on":
            return _pick(lang, f"I couldn't find {args.get('target', 'that')} on the screen.",
                         f"لم أجد {args.get('target', 'ذلك')} على الشاشة.")
        if tool == "describe_screen":
            return _pick(lang, "I can't see the screen right now.", "لا أستطيع رؤية الشاشة الآن.")
        if tool == "get_weather":
            if "no city" in res.error:
                return _pick(lang, "Which city?", "في أي مدينة؟")
            return _pick(lang, "I can't get live weather right now. I may be offline.",
                         "لا أستطيع جلب حالة الطقس الآن، قد لا يوجد اتصال بالإنترنت.")
        return _pick(lang, "Sorry, that didn't work.", "عذراً، لم أتمكن من ذلك.")
    if tool == "open_app":
        return _pick(lang, f"Opening {n}.", f"جارٍ فتح {n}.")
    if tool == "close_app":
        if d.get("asking_to_save"):
            return _pick(lang, f"{n} is asking whether to save your changes.", f"{n} يسأل إن كنت تريد حفظ التغييرات.")
        return _pick(lang, f"Closed {n}.", f"أغلقت {n}.")
    if tool == "focus_window":
        return _pick(lang, f"Switched to {n}.", f"انتقلت إلى {n}.")
    if tool == "take_screenshot":
        return _pick(lang, "Screenshot saved.", "حفظت لقطة الشاشة.")
    if tool == "describe_screen":
        return d.get("description", "")
    if tool == "web_search":
        q = args.get("query", "")
        if args.get("site") == "youtube":
            return _pick(lang, f"Searching YouTube for {q}.", f"أبحث في يوتيوب عن {q}.")
        return _pick(lang, f"Searching for {q}.", f"أبحث عن {q}.")
    if tool == "youtube_search":
        q = args.get("query", "")
        return _pick(lang, f"Searching YouTube for {q}.", f"أبحث في يوتيوب عن {q}.")
    if tool == "get_date_time":
        return _date_time(d, args.get("ask", "time"), lang)
    if tool == "get_weather":
        return _pick(lang, f"In {d['city']} it is {round(d['temperature_c'])} degrees, {d['condition']}, "
                           f"humidity {round(d['humidity_percent'])} percent.",
                     f"في {d['city']} الحرارة {round(d['temperature_c'])} درجة، والرطوبة {round(d['humidity_percent'])} بالمئة.")
    if tool == "open_calendar":
        return _pick(lang, "Opening your calendar.", "أفتح التقويم.")
    if tool == "add_calendar_event":
        return _pick(lang, f"I opened the event {d['event']} in Google Calendar. Press save to keep it.",
                     f"فتحت الموعد {d['event']} في تقويم جوجل. اضغط حفظ لتثبيته.")
    if tool == "open_url":
        return _pick(lang, "Opening it in the browser.", "أفتحه في المتصفح.")
    if tool == "type_text":
        return _pick(lang, "Typed it.", "تمت الكتابة.")
    if tool == "press_keys":
        return _pick(lang, "Done.", "تم.")
    if tool in ACTION_WORDS:
        key = args.get("action", "")
        en, ar = ACTION_WORDS[tool].get(key, ("Done.", "تم."))
        return _pick(lang, en, ar).format(n=n)
    if tool == "lock_screen":
        return _pick(lang, "Locking the computer.", "أقفل الجهاز الآن.")
    if tool == "system_status":
        parts_en = [f"CPU at {round(d['cpu_percent'])} percent", f"memory at {round(d['memory_percent'])} percent"]
        parts_ar = [f"المعالج {round(d['cpu_percent'])} بالمئة", f"الذاكرة {round(d['memory_percent'])} بالمئة"]
        if "battery_percent" in d:
            parts_en.append(f"battery at {d['battery_percent']} percent" + (", charging" if d.get("charging") else ""))
            parts_ar.append(f"البطارية {d['battery_percent']} بالمئة" + (" وهي تشحن" if d.get("charging") else ""))
        if "gpu_percent" in d:
            parts_en.append(f"GPU at {d['gpu_percent']} percent")
            parts_ar.append(f"كرت الشاشة {d['gpu_percent']} بالمئة")
        return _pick(lang, ", ".join(parts_en).capitalize() + ".", "، ".join(parts_ar) + ".")
    if tool == "set_timer":
        if d.get("label"):
            return _pick(lang, f"I'll remind you in {_dur(d['seconds'], lang)}.",
                         f"سأذكرك بعد {_dur(d['seconds'], lang)}.")
        return _pick(lang, f"Timer set for {_dur(d['seconds'], lang)}.", f"ضبطت مؤقتاً لمدة {_dur(d['seconds'], lang)}.")
    if tool == "remember":
        return _pick(lang, "Got it, I'll remember that.", "حسناً، سأتذكر ذلك.")
    if tool == "forget":
        return _pick(lang, "Forgotten.", "تم النسيان.")
    if tool == "add_note":
        return _pick(lang, "Note saved.", "حفظت الملاحظة.")
    if tool == "read_notes":
        notes = d.get("notes") or []
        if not notes:
            return _pick(lang, "You have no notes yet.", "لا توجد ملاحظات بعد.")
        items = [n.split("  ", 1)[-1] for n in notes[-3:]]
        return _pick(lang, "Your latest notes: " + "; ".join(items) + ".", "آخر ملاحظاتك: " + "، ".join(items) + ".")
    if tool == "power":
        a = d.get("power")
        if a == "cancel":
            return _pick(lang, "Shutdown cancelled.", "ألغيت إيقاف التشغيل.")
        return _pick(lang, f"The computer will {'restart' if a == 'restart' else 'shut down'} in 30 seconds. "
                           "Say cancel shutdown to stop it.",
                     f"سيتم {'إعادة تشغيل' if a == 'restart' else 'إيقاف'} الجهاز خلال ثلاثين ثانية. قل ألغ الإيقاف للتراجع.")
    return _pick(lang, "Done.", "تم.")


# Slow tools: say this at once, then only the outcome (follow_up) when the tool returns.
PRE = {"close_app": ("Closing {n}.", "جارٍ إغلاق {n}."), "describe_screen": ("Let me look.", "دعني ألقي نظرة."),
       "click_on": ("Looking for it.", "أبحث عنه.")}


def pre(tool: str, args: dict, lang: str) -> str:
    en, ar = PRE[tool]
    return _pick(lang, en, ar).format(n=args.get("name", ""))


def follow_up(tool: str, args: dict, res: ToolResult, lang: str) -> str:
    """What to add after the PRE sentence ('' when the PRE sentence already said it)."""
    if not res.ok:
        return say(tool, args, res, lang)
    if tool == "close_app":
        return ""
    if tool == "click_on":
        return _pick(lang, "Clicked.", "تم النقر.")
    return say(tool, args, res, lang)


def confirm_question(tool: str, args: dict, lang: str) -> str:
    n = args.get("name", "")
    if tool == "close_app":
        return _pick(lang, f"Should I close {n}? Unsaved changes will be lost. Say yes or no.",
                     f"هل أغلق {n}؟ ستضيع التغييرات غير المحفوظة. قل نعم أو لا.")
    if tool == "power":
        act = args.get("action")
        return _pick(lang, f"Do you really want to {'restart' if act == 'restart' else 'shut down'} the computer? "
                           "Say yes or no.",
                     f"هل تريد فعلاً {'إعادة تشغيل' if act == 'restart' else 'إيقاف'} الجهاز؟ قل نعم أو لا.")
    if tool == "press_keys":
        return _pick(lang, f"Pressing {args.get('keys')} can close things. Should I do it?",
                     f"الضغط على {args.get('keys')} قد يغلق شيئاً. هل أنفذ ذلك؟")
    f = args.get("path") or args.get("source") or ""
    if tool == "write_file":
        return _pick(lang, f"Should I save the file {f}? Say yes or no.", f"هل أحفظ الملف {f}؟ قل نعم أو لا.")
    if tool == "edit_file":
        return _pick(lang, f"Should I change the file {f}? Say yes or no.", f"هل أعدل الملف {f}؟ قل نعم أو لا.")
    if tool == "move_file":
        return _pick(lang, f"Should I move {f} to {args.get('destination', '')}? Say yes or no.",
                     f"هل أنقل {f} إلى {args.get('destination', '')}؟ قل نعم أو لا.")
    if tool == "delete_file":
        return _pick(lang, f"Should I delete {f}? Say yes or no.", f"هل أحذف {f}؟ قل نعم أو لا.")
    return _pick(lang, "Should I go ahead? Say yes or no.", "هل أنفذ ذلك؟ قل نعم أو لا.")


def cancelled(lang: str) -> str:
    return _pick(lang, "Okay, I won't.", "حسناً، لن أفعل.")


def sleeping(lang: str) -> str:
    return _pick(lang, "Going quiet. Say HAZEM when you need me.", "سأصمت الآن. قل حزم عندما تحتاجني.")


def cannot_do(lang: str) -> str:
    return _pick(lang, "Sorry, I couldn't do that.", "عذراً، لم أتمكن من ذلك.")


def waking(lang: str) -> str:
    return _pick(lang, "I'm here.", "أنا هنا.")


def timer_done(label: str, lang: str) -> str:
    if label:
        return _pick(lang, f"Reminder: {label}.", f"تذكير: {label}.")
    return _pick(lang, "Your timer is done.", "انتهى المؤقت.")


def _date_time(d: dict, ask: str, lang: str) -> str:
    # This function says the time, the date, or the weekday.
    if ask == "date":
        return _pick(lang, f"Today is {d['weekday']}, {d['month']} {d['day']}, {d['year']}.",
                     f"اليوم {d['weekday_ar']}، {d['day']} {d['month_ar']} {d['year']}.")
    if ask == "day":
        return _pick(lang, f"Today is {d['weekday']}.", f"اليوم {d['weekday_ar']}.")
    return _pick(lang, f"It's {d['time']}.", f"الساعة الآن {d['time']}.")
