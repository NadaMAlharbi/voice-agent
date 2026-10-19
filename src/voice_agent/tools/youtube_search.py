"""Search YouTube in the browser."""

from __future__ import annotations

from functools import partial
from urllib.parse import quote_plus

from voice_agent.tools.base import Tool, ToolResult

URL = "https://www.youtube.com/results?search_query={q}"


def youtube_search(tb, query: str) -> ToolResult:
    # This function opens YouTube search results for the query.
    url = URL.format(q=quote_plus(query.strip()))
    used = tb.desktop.open_url(url)
    return ToolResult(True, {"searched": query, "site": "youtube", "browser": used, "url": url})


def tools(tb) -> list[Tool]:
    return [Tool("youtube_search", "Search YouTube for videos and open the results in the browser.",
                 {"query": {"type": "string"}}, partial(youtube_search, tb), ["query"])]
