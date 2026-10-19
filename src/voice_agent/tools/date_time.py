"""Current date and time from this computer's clock (works offline)."""

from __future__ import annotations

from datetime import datetime
from functools import partial

from voice_agent.tools.base import Tool, ToolResult

WEEKDAYS_AR = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]
MONTHS_AR = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر",
             "نوفمبر", "ديسمبر"]


def get_date_time(tb=None, ask: str = "time", now: datetime | None = None) -> ToolResult:
    # This function returns today's date, the weekday and the time. `ask` only picks the spoken answer.
    now = now or datetime.now().astimezone()
    return ToolResult(True, {
        "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M"),
        "weekday": now.strftime("%A"), "weekday_ar": WEEKDAYS_AR[now.weekday()],
        "month": now.strftime("%B"), "month_ar": MONTHS_AR[now.month - 1], "day": now.day, "year": now.year,
        "hour": now.hour, "minute": now.minute, "timezone": now.strftime("%Z") or now.strftime("%z")})


def tools(tb) -> list[Tool]:
    return [Tool("get_date_time", "Current date, weekday and time on this computer.",
                 {"ask": {"type": "string", "enum": ["time", "date", "day"]}}, partial(get_date_time, tb))]
