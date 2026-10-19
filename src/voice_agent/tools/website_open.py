"""Open websites: a web address, or a well-known site by its spoken name."""

from __future__ import annotations

import re
from functools import partial

from voice_agent.tools.apps import normalize
from voice_agent.tools.base import Tool, ToolResult

SITES = {  # spoken site name -> url ("open youtube", "افتح يوتيوب")
    "youtube": "https://www.youtube.com", "يوتيوب": "https://www.youtube.com", "اليوتيوب": "https://www.youtube.com",
    "gmail": "https://mail.google.com", "جيميل": "https://mail.google.com", "الايميل": "https://mail.google.com",
    "google": "https://www.google.com", "جوجل": "https://www.google.com", "قوقل": "https://www.google.com",
    "github": "https://github.com", "جيت هب": "https://github.com",
    "wikipedia": "https://www.wikipedia.org", "ويكيبيديا": "https://ar.wikipedia.org",
    "google maps": "https://maps.google.com", "maps": "https://maps.google.com", "خرائط جوجل": "https://maps.google.com",
    "الخرائط": "https://maps.google.com",
    "twitter": "https://x.com", "x": "https://x.com", "تويتر": "https://x.com", "اكس": "https://x.com",
    "facebook": "https://www.facebook.com", "فيسبوك": "https://www.facebook.com", "فيس بوك": "https://www.facebook.com",
    "instagram": "https://www.instagram.com", "انستقرام": "https://www.instagram.com", "انستغرام": "https://www.instagram.com",
    "linkedin": "https://www.linkedin.com", "لينكد ان": "https://www.linkedin.com",
    "chatgpt": "https://chatgpt.com", "netflix": "https://www.netflix.com", "نتفلكس": "https://www.netflix.com",
}

_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}(?:/\S*)?$", re.I)


def site_url(spoken: str) -> str | None:
    n = normalize(spoken)
    for name, url in SITES.items():
        if n == normalize(name) or n in (f"{normalize(name)} website", f"موقع {normalize(name)}"):
            return url
    s = spoken.strip().lower().replace(" dot ", ".").replace(" نقطة ", ".")
    if _DOMAIN.match(s.replace(" ", "")):
        s = s.replace(" ", "")
        return s if s.startswith("http") else f"https://{s}"
    return None


def open_url(tb, url: str) -> ToolResult:
    # This function opens the requested website (http/https only).
    u = url.strip()
    if not re.match(r"^https?://", u, re.I):
        if re.match(r"^[a-z][\w+.-]*:", u, re.I):
            return ToolResult(False, error="only http and https links can be opened")
        u = "https://" + u.replace(" ", "")
    used = tb.desktop.open_url(u)
    return ToolResult(True, {"opened": u, "browser": used})


def tools(tb) -> list[Tool]:
    return [Tool("open_url", "Open a web address in the browser.", {"url": {"type": "string"}},
                 partial(open_url, tb), ["url"])]
