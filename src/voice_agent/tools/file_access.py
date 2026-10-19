"""The one folder the file tools may use, and the Filesystem MCP server for it.

The server gets that folder as its only allowed directory. Paths are checked here
too, so a path outside the folder never reaches the server.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path, PurePosixPath

from voice_agent.tools.mcp_client import McpClient

DEFAULT_COMMAND = ["npx", "-y", "--prefer-offline", "@modelcontextprotocol/server-filesystem@2026.8.31"]

_clients: dict[tuple, McpClient] = {}
_clients_lock = threading.Lock()


class Files:
    """The allowed folder and its MCP server."""

    def __init__(self, root: str | Path, command: list[str] | None = None):
        self.root = Path(root).expanduser().resolve()
        self.command = list(command or DEFAULT_COMMAND)

    @property
    def client(self) -> McpClient:
        key = (*self.command, str(self.root))
        with _clients_lock:                 # one server for all browser tabs
            if key not in _clients:
                _clients[key] = McpClient(self.command + [str(self.root)])
            return _clients[key]

    def available(self) -> bool:
        return shutil.which(self.command[0]) is not None

    def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.client.start()

    def path(self, rel: str) -> Path:
        """User path -> absolute path inside the root. Raises ValueError outside it."""
        rel = (rel or "").strip().strip('"').replace("\\", "/")
        p = Path(rel).expanduser()
        p = (p if p.is_absolute() else self.root / p).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"only files inside {self.root.name} are allowed")
        return p

    def rel(self, p: str | Path) -> str:
        try:
            return PurePosixPath(Path(p).resolve().relative_to(self.root)).as_posix() or "."
        except ValueError:
            return str(p)

    def call(self, tool: str, **args) -> tuple[bool, str]:
        self.start()
        return self.client.call(tool, args)


def make_files(cfg: dict | None) -> Files | None:
    """Files from config (tools.files), or None when turned off or Node.js is missing."""
    cfg = cfg or {}
    if not cfg.get("enabled"):
        return None
    fs = Files(cfg.get("root") or Path.home() / "Documents" / "HAZEM", cfg.get("command") or None)
    return fs if fs.available() else None
