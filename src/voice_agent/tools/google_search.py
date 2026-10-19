"""Search Google (web, maps, Wikipedia, images) in the browser."""

from __future__ import annotations

from functools import partial
from urllib.parse import quote_plus

from voice_agent.tools.base import Tool, ToolResult

SEARCH_URLS = {
    "google": "https://www.google.com/search?q={q}",
    "maps": "https://www.google.com/maps/search/{q}",
    "wikipedia": "https://{lang}.wikipedia.org/w/index.php?search={q}",
    "images": "https://www.google.com/search?tbm=isch&q={q}",
}


def web_search(tb, query: str, site: str = "google", language: str = "en") -> ToolResult:
    # This function opens a search results page in the browser.
    tpl = SEARCH_URLS.get(site, SEARCH_URLS["google"])
    url = tpl.format(q=quote_plus(query.strip()), lang="ar" if language == "ar" else "en")
    used = tb.desktop.open_url(url)
    return ToolResult(True, {"searched": query, "site": site, "browser": used, "url": url})


def tools(tb) -> list[Tool]:
    S = {"type": "string"}
    return [Tool("web_search", "Search the web in the browser. site: google (default), maps, wikipedia or images.",
                 {"query": S, "site": {**S, "enum": list(SEARCH_URLS)}, "language": {**S, "enum": ["en", "ar"]}},
                 partial(web_search, tb), ["query"])]
