"""Google Calendar in the browser: open it, or open a pre-filled new event (no account setup needed)."""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import partial
from urllib.parse import urlencode

from voice_agent.tools.base import Tool, ToolResult

CALENDAR = "https://calendar.google.com/calendar/r"
NEW_EVENT = "https://calendar.google.com/calendar/render?"


def open_calendar(tb) -> ToolResult:
    # This function opens Google Calendar.
    used = tb.desktop.open_url(CALENDAR)
    return ToolResult(True, {"opened": CALENDAR, "browser": used})


def event_url(title: str, start: datetime, minutes: int = 60) -> str:
    # This function builds the Google Calendar link for a new event.
    end = start + timedelta(minutes=max(5, minutes))
    dates = f"{start:%Y%m%dT%H%M%S}/{end:%Y%m%dT%H%M%S}"
    return NEW_EVENT + urlencode({"action": "TEMPLATE", "text": title, "dates": dates})


def add_calendar_event(tb, title: str, start: str, minutes: int = 60) -> ToolResult:
    # This function opens a new event in Google Calendar; the user checks it and presses Save.
    try:
        when = datetime.fromisoformat(start)
    except ValueError:
        return ToolResult(False, error="start must be a date and time like 2026-10-05T15:00")
    url = event_url(title, when, int(minutes))
    used = tb.desktop.open_url(url)
    return ToolResult(True, {"event": title, "start": when.isoformat(timespec="minutes"), "minutes": minutes,
                             "browser": used, "needs_save": True})


def tools(tb) -> list[Tool]:
    S, I = {"type": "string"}, {"type": "integer"}
    return [
        Tool("open_calendar", "Open Google Calendar in the browser.", {}, partial(open_calendar, tb)),
        Tool("add_calendar_event", "Create a Google Calendar event (opens it pre-filled; the user presses Save). "
             "start is local time in ISO format, e.g. 2026-10-05T15:00.",
             {"title": S, "start": S, "minutes": I}, partial(add_calendar_event, tb), ["title", "start"]),
    ]
