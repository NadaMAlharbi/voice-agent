"""The assistant's tools. Each feature lives in its own file; this file only collects them.

To add a tool: create tools/<feature>.py with handler functions and a `tools(tb)` function
returning Tool(...) entries, then add the module to FEATURES below.

Tools from other projects come through MCP instead: add the server to tools.mcp_servers
in config.yaml (tools/mcp_bridge.py). No code change is needed for those.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from voice_agent.tools import (
    calculator,
    date_time,
    files,
    google_calendar,
    google_search,
    gui,
    launch_app,
    media_volume,
    memory,
    notes,
    open_window,
    power,
    screen_vision,
    screenshot,
    system_stats,
    timers,
    weather,
    website_open,
    youtube_search,
)
from voice_agent.tools.apps import AppResolver
from voice_agent.tools.base import ToolRegistry
from voice_agent.tools.desktop import Desktop, Window

FEATURES = [launch_app, open_window, screenshot, screen_vision, website_open, google_search, youtube_search,
            google_calendar, gui, media_volume, system_stats, power, date_time, weather, timers, notes, memory,
            files, calculator]


class Toolbox:
    """What the tools share (desktop, data folder, vision model) and the tool registry."""

    def __init__(self, desktop: Desktop, data_dir: Path, vision=None,
                 scheduler: Callable[[float, str], str] | None = None, allow_power: bool = True,
                 home_city: str = "", files=None, mcp=None):
        self.desktop = desktop
        self.data_dir = Path(data_dir)
        self.vision = vision
        self.scheduler = scheduler          # set by the voice session: (seconds, label) -> timer id
        self.memory = memory.MemoryStore(self.data_dir / "memory.json")
        self.notes_path = self.data_dir / "notes.md"
        self.allow_power = allow_power
        self.home_city = home_city          # default city for the weather tool
        self.files = files                  # tools/files.py Files (MCP), or None
        self.mcp = mcp                      # tools/mcp_bridge.py McpHub (servers from config), or None
        self._resolver: AppResolver | None = None
        self._resolver_lock = threading.Lock()
        self.registry = ToolRegistry([t for f in FEATURES for t in f.tools(self)])
        for t in (mcp.tools() if mcp is not None else []):   # only servers that are ready
            if self.registry.get(t.name) is None:           # a built-in tool always wins
                self.registry.add(t)

    def resolver(self) -> AppResolver:
        with self._resolver_lock:
            if self._resolver is None:
                self._resolver = AppResolver(self.desktop.list_apps(), self.desktop.builtin_apps())
            return self._resolver

    def find_windows(self, name: str) -> list[Window]:
        return open_window.find_windows(self, name)
