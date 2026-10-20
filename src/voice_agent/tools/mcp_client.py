"""A small MCP client: talks to one MCP server over stdio (JSON-RPC, one JSON message per line).

MCP (Model Context Protocol) servers offer tools in a standard way. We start the
server as a child process, ask for its tools, and call them. Only stdio is used:
nothing goes over the network.

  client = McpClient(["npx", "-y", "@modelcontextprotocol/server-filesystem", "C:/Users/me/Documents/HAZEM"])
  client.call("list_directory", {"path": "C:/Users/me/Documents/HAZEM"})      # -> (ok, text)
  client.call_tool("list_directory", {"path": "..."})                        # -> full MCP result dict

Used by tools/file_access.py (file tools) and tools/mcp_bridge.py (servers from config.yaml).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections import deque

PROTOCOL_VERSION = "2025-06-18"
_CLOSED = {"_closed": True}          # reply put in every waiting slot when the server stops


class McpError(RuntimeError):
    pass


class McpClient:
    def __init__(self, command: list[str], cwd: str | None = None, env: dict | None = None,
                 start_timeout_s: float = 60.0):
        """env: added to this process's environment; a value of None removes that variable."""
        self.command = list(command)
        self.cwd = cwd
        self.env = env
        self.start_timeout_s = start_timeout_s      # first npx run may download the server
        self.server_info: dict = {}
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()               # one start at a time
        self._write_lock = threading.Lock()
        self._next_id = 0
        self._waiting: dict[int, dict] = {}         # request id -> {"event", "reply"}
        self._stderr: deque[str] = deque(maxlen=20)
        self._stderr_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ process

    def _resolve(self) -> list[str]:
        exe = shutil.which(self.command[0])         # finds npx.cmd on Windows
        if exe is None:
            hint = "Install Node.js (it brings npx)" if self.command[0] in ("npx", "node") else "Install it"
            raise McpError(f"{self.command[0]!r} not found on PATH. {hint}, or fix the command in config.yaml")
        return [exe] + self.command[1:]

    def start(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            try:
                self._start_locked()
            except McpError as e:
                # npx --prefer-offline trusts npm's cached package list without asking the
                # registry. If that list predates the pinned version, npm says "notarget"
                # although the version exists. Ask the registry once, then the cache is fresh.
                if "--prefer-offline" not in self.command or not any(
                        k in str(e) for k in ("notarget", "ETARGET", "No matching version")):
                    raise
                self.command = [c for c in self.command if c != "--prefer-offline"]
                self._start_locked()

    def _start_locked(self) -> None:
        env = {**os.environ, **(self.env or {})}
        env = {k: str(v) for k, v in env.items() if v is not None}
        self._stderr.clear()
        self._proc = subprocess.Popen(
            self._resolve(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=self.cwd, env=env, bufsize=0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True).start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True)
        self._stderr_thread.start()
        try:
            result = self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                "clientInfo": {"name": "voice-agent", "version": "0.1"}}, self.start_timeout_s)
            if "serverInfo" not in result:
                raise McpError(f"MCP server gave no valid initialize reply. {self._exit_reason()}")
            self.server_info = result["serverInfo"]
            self._notify("notifications/initialized")
        except McpError:
            self.close()                    # do not keep a half-started server; next call retries
            raise

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=3)
        except Exception:  # noqa: BLE001  (stdin of a dead server: EINVAL / EPIPE)
            try:
                proc.kill()
            except OSError:
                pass

    # ------------------------------------------------------------------ messages

    def _send(self, msg: dict) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            raise McpError(f"MCP server is not running. {self._exit_reason()}")
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        with self._write_lock:
            try:
                proc.stdin.write(data)
                proc.stdin.flush()
            except OSError as e:
                # The server closed its stdin (it exited). Windows reports this as
                # [Errno 22] Invalid argument, Linux/macOS as [Errno 32] Broken pipe.
                # The useful reason is what the server printed on stderr.
                raise McpError(f"MCP server closed the connection ({e}). {self._exit_reason()}") from e

    def _notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def _request(self, method: str, params: dict, timeout_s: float) -> dict:
        with self._write_lock:
            self._next_id += 1
            rid = self._next_id
        slot = {"event": threading.Event(), "reply": None}
        self._waiting[rid] = slot
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            if not slot["event"].wait(timeout_s):
                raise McpError(f"MCP server did not answer {method} within {timeout_s:.0f} s. {self._last_error()}")
        finally:
            self._waiting.pop(rid, None)
        reply = slot["reply"] or {}
        if reply is _CLOSED:
            raise McpError(f"MCP server stopped before answering {method}. {self._exit_reason()}")
        if "error" in reply:
            raise McpError(reply["error"].get("message", str(reply["error"])))
        return reply.get("result", {})

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue                      # not JSON: ignore
            if "id" in msg and ("result" in msg or "error" in msg):
                slot = self._waiting.get(msg["id"])
                if slot is not None:
                    slot["reply"] = msg
                    slot["event"].set()
            elif "id" in msg and "method" in msg:
                self._answer_server(msg)
        for slot in list(self._waiting.values()):   # the server stopped: wake every caller
            if slot["reply"] is None:
                slot["reply"] = _CLOSED
            slot["event"].set()

    def _answer_server(self, msg: dict) -> None:
        # The server may ask us things. We answer ping; anything else is "not supported".
        if msg["method"] == "ping":
            reply = {"jsonrpc": "2.0", "id": msg["id"], "result": {}}
        else:
            reply = {"jsonrpc": "2.0", "id": msg["id"],
                     "error": {"code": -32601, "message": f"{msg['method']} not supported"}}
        try:
            self._send(reply)
        except McpError:
            pass

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        for line in proc.stderr:
            self._stderr.append(line.decode("utf-8", "replace").rstrip())

    def _last_error(self) -> str:
        return ("Server said: " + " | ".join(list(self._stderr)[-3:])) if self._stderr else ""

    def _exit_reason(self) -> str:
        """Exit code and last stderr lines of a server that stopped (waits briefly for them)."""
        proc = self._proc
        code = None
        if proc is not None:
            try:
                code = proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        t = self._stderr_thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1)               # stderr is read by a thread: let it catch up
        parts = [f"Exit code {code}." if code is not None else "", self._last_error()]
        if not self._stderr:
            parts.append(f"Command: {' '.join(self.command)}")
        return " ".join(p for p in parts if p)

    # ------------------------------------------------------------------ MCP

    def list_tools(self) -> list[dict]:
        self.start()
        tools, params = [], {}
        for _ in range(50):                         # servers may page the list (nextCursor)
            page = self._request("tools/list", params, 15.0)
            tools += page.get("tools", [])
            if not page.get("nextCursor"):
                break
            params = {"cursor": page["nextCursor"]}
        return tools

    def call_tool(self, name: str, arguments: dict, timeout_s: float = 30.0) -> dict:
        """The full MCP result: {"content": [...], "isError": bool, "structuredContent": {...}?}."""
        self.start()
        return self._request("tools/call", {"name": name, "arguments": arguments}, timeout_s)

    def call(self, name: str, arguments: dict, timeout_s: float = 30.0) -> tuple[bool, str]:
        """(ok, text). The text joins the text parts of the result."""
        result = self.call_tool(name, arguments, timeout_s)
        return not result.get("isError", False), result_text(result)


def result_text(result: dict) -> str:
    """Join the text parts of an MCP tool result."""
    parts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
    return "\n".join(p for p in parts if p)
