"""MCP servers from config.yaml -> ordinary tools for the agent (the generic MCP bridge).

Any project can give HAZEM new tools by shipping an MCP server. HAZEM starts it as a
child process (stdio), asks for its tools, and wraps each one as a normal Tool, so the
LLM, the "yes or no" confirmation, timeouts and logging all work as for built-in tools.

config.yaml:

  tools:
    mcp_servers:
      calculator:                                   # server name = tool name prefix
        command: [uv, run, --directory, examples/mcp-calculator, mcp-calculator]
        tools: [add_numbers]                        # optional allow-list (default: all)
        confirm: []                                 # always ask "yes or no" first
        safe: []                                    # never ask
        timeout_s: 30

The LLM then sees the tool "calculator__add_numbers".

Rules:
  - Native tools (tools/*.py) are not changed. tools/files.py keeps its own hand-made
    layer over the Filesystem MCP server; this bridge is for everything else.
  - Risk: confirm/safe in config win. Otherwise a tool is "safe" only when the server
    marks it readOnlyHint=true; every other tool asks "yes or no" first.
  - A server that fails to start is reported and skipped; other tools still work.
  - A server that crashes later is restarted on the next call (McpClient.start).
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from voice_agent.config import PROJECT_ROOT
from voice_agent.tools.base import CONFIRM, SAFE, Tool, ToolResult
from voice_agent.tools.mcp_client import McpClient, McpError, result_text

SEP = "__"                                  # calculator + add_numbers -> calculator__add_numbers
MAX_TEXT = 3000                             # longer tool text is cut (the reply is spoken)
SERVER_NAME = re.compile(r"^[a-z][a-z0-9_]{0,23}$")
KEYS = {"enabled", "command", "cwd", "env", "tools", "confirm", "safe", "timeout_s", "start_timeout_s", "hint"}


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    command: tuple[str, ...]
    cwd: Path = PROJECT_ROOT                # relative paths in command are relative to this
    env: dict = field(default_factory=dict)
    enabled: bool = True
    tools: tuple[str, ...] = ()             # allow-list of MCP tool names; empty = all
    confirm: tuple[str, ...] = ()           # always ask "yes or no"
    safe: tuple[str, ...] = ()              # never ask
    timeout_s: float = 30.0                 # per tool call
    start_timeout_s: float = 60.0           # first start may install packages
    hint: str = ""                          # one line for the LLM: when to use these tools


def _names(raw, key: str, server: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise ValueError(f"mcp_servers.{server}.{key} must be a list of tool names")
    return tuple(raw)


def parse_servers(raw: dict | None, root: Path = PROJECT_ROOT) -> list[McpServerConfig]:
    """config.yaml tools.mcp_servers -> checked server settings. Raises ValueError with the reason."""
    if not raw:
        return []
    if not isinstance(raw, dict):
        raise ValueError("tools.mcp_servers must be a mapping: name -> settings")
    out = []
    for name, d in raw.items():
        if not isinstance(name, str) or not SERVER_NAME.match(name):
            raise ValueError(f"mcp_servers: bad server name {name!r} (use lowercase letters, digits, _; "
                             "start with a letter; max 24)")
        d = d or {}
        if not isinstance(d, dict):
            raise ValueError(f"mcp_servers.{name} must be a mapping")
        unknown = set(d) - KEYS
        if unknown:
            raise ValueError(f"mcp_servers.{name}: unknown setting(s) {sorted(unknown)}; allowed: {sorted(KEYS)}")
        cmd = d.get("command")
        if not (isinstance(cmd, list) and cmd and all(isinstance(c, (str, int, float)) for c in cmd)):
            raise ValueError(f"mcp_servers.{name}.command must be a list, e.g. [uv, run, my-server]")
        cwd = Path(str(d.get("cwd") or root)).expanduser()
        env = d.get("env") or {}
        if not isinstance(env, dict):
            raise ValueError(f"mcp_servers.{name}.env must be a mapping")
        out.append(McpServerConfig(
            name=name, command=tuple(str(c) for c in cmd), cwd=cwd if cwd.is_absolute() else (root / cwd),
            env={str(k): (None if v is None else str(v)) for k, v in env.items()},
            enabled=bool(d.get("enabled", True)),
            tools=_names(d.get("tools"), "tools", name), confirm=_names(d.get("confirm"), "confirm", name),
            safe=_names(d.get("safe"), "safe", name),
            timeout_s=float(d.get("timeout_s", 30)), start_timeout_s=float(d.get("start_timeout_s", 60)),
            hint=str(d.get("hint") or "").strip()))
    return out


# ---------------------------------------------------------------- one MCP tool -> one Tool

def agent_tool_name(server: str, tool: str) -> str:
    """LLM function names allow [A-Za-z0-9_-], max 64 characters."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", f"{server}{SEP}{tool}")[:64]


def tool_risk(spec: McpServerConfig, mcp_tool: dict) -> str:
    name = mcp_tool["name"]
    if name in spec.confirm:
        return CONFIRM
    if name in spec.safe:
        return SAFE
    ann = mcp_tool.get("annotations") or {}
    return SAFE if ann.get("readOnlyHint") is True else CONFIRM


def to_tool_result(result: dict) -> ToolResult:
    """MCP tools/call result -> ToolResult (what the LLM and the browser log see)."""
    text = result_text(result)
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT] + " ...(cut)"
    if result.get("isError"):
        # drop links such as "For further information visit https://errors.pydantic.dev/..."
        text = "\n".join(ln for ln in text.splitlines() if "For further information visit" not in ln)
        return ToolResult(False, error=text.strip() or "the tool reported an error")
    structured = result.get("structuredContent")
    data = dict(structured) if isinstance(structured, dict) and structured else {"result": text}
    others = [c.get("type") for c in result.get("content", []) if c.get("type") != "text"]
    if others:
        data["other_content"] = others          # images/audio/resources are not shown yet
    return ToolResult(True, data)


def bridge_tool(spec: McpServerConfig, client: McpClient, mcp_tool: dict) -> Tool:
    schema = mcp_tool.get("inputSchema") or {}
    mcp_name = mcp_tool["name"]

    def handler(**args) -> ToolResult:
        try:
            return to_tool_result(client.call_tool(mcp_name, args, spec.timeout_s))
        except McpError as e:
            return ToolResult(False, error=f"{spec.name} server: {e}")

    description = (mcp_tool.get("description") or mcp_tool.get("title") or mcp_name).strip()
    return Tool(
        name=agent_tool_name(spec.name, mcp_name),
        description=description,
        parameters=schema.get("properties") or {},
        handler=handler,
        required=list(schema.get("required") or []),
        risk=tool_risk(spec, mcp_tool),
        timeout_s=spec.timeout_s + 2,           # the client's own timeout answers first, with a reason
        defs=schema.get("$defs") or {},
    )


# ---------------------------------------------------------------- all servers

@dataclass
class ServerStatus:
    name: str
    state: str = "not started"              # not started | starting | ready | failed | disabled
    error: str = ""
    seconds: float = 0.0
    server_info: dict = field(default_factory=dict)
    tools: list[dict] = field(default_factory=list)   # MCP tool descriptions (after the allow-list)
    missing: list[str] = field(default_factory=list)  # allow-listed names the server does not have


class McpHub:
    """Starts the configured MCP servers once and hands out their tools as Tool objects."""

    def __init__(self, specs: list[McpServerConfig]):
        self.specs = {s.name: s for s in specs}
        self.status = {s.name: ServerStatus(s.name, "not started" if s.enabled else "disabled") for s in specs}
        self.clients: dict[str, McpClient] = {}
        self._done = threading.Event()
        self._lock = threading.Lock()
        if not any(s.enabled for s in specs):
            self._done.set()

    def start_one(self, name: str) -> ServerStatus:
        spec, st = self.specs[name], self.status[name]
        if not spec.enabled:
            return st
        st.state, st.error = "starting", ""
        t0 = time.perf_counter()
        client = McpClient(list(spec.command), cwd=str(spec.cwd),
                           # a uv server must use ITS .venv, not HAZEM's activated one
                           env={"VIRTUAL_ENV": None, **spec.env}, start_timeout_s=spec.start_timeout_s)
        try:
            tools = client.list_tools()
        except McpError as e:
            client.close()
            st.state, st.error, st.seconds = "failed", str(e), time.perf_counter() - t0
            return st
        names = {t.get("name") for t in tools}
        st.missing = [n for n in spec.tools if n not in names]
        st.tools = [t for t in tools if t.get("name") and (not spec.tools or t["name"] in spec.tools)]
        st.server_info = client.server_info
        st.seconds = time.perf_counter() - t0
        st.state = "ready"
        with self._lock:
            self.clients[name] = client
        return st

    def start_all(self) -> list[ServerStatus]:
        """Start every enabled server in parallel and wait for all of them."""
        threads = [threading.Thread(target=self.start_one, args=(n,), daemon=True)
                   for n, s in self.specs.items() if s.enabled]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self._done.set()
        return list(self.status.values())

    def start_background(self, on_done=None) -> threading.Thread:
        def run():
            res = self.start_all()
            if on_done is not None:
                on_done(res)
        t = threading.Thread(target=run, daemon=True, name="mcp-start")
        t.start()
        return t

    def wait(self, timeout_s: float) -> bool:
        return self._done.wait(timeout_s)

    def tools(self) -> list[Tool]:
        """Tools of the servers that are ready now."""
        out = []
        with self._lock:
            clients = dict(self.clients)
        for name, client in clients.items():
            spec = self.specs[name]
            out += [bridge_tool(spec, client, t) for t in self.status[name].tools]
        return out

    def hints(self) -> list[str]:
        return [self.specs[n].hint for n in self.clients if self.specs[n].hint]

    def close(self) -> None:
        with self._lock:
            clients, self.clients = self.clients, {}
        for c in clients.values():
            c.close()


_hubs: dict[str, McpHub | None] = {}
_hubs_lock = threading.Lock()


def shared_hub(raw: dict | None) -> McpHub | None:
    """One hub per config for the whole process (all browser tabs share the servers).

    None when no servers are configured. A bad config raises ValueError once per process.
    """
    if not raw:
        return None
    key = json.dumps(raw, sort_keys=True, default=str)
    with _hubs_lock:
        if key not in _hubs:
            _hubs[key] = McpHub(parse_servers(raw))
        return _hubs[key]
