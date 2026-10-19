"""A small MCP client: talks to one MCP server over stdio (JSON-RPC, one JSON message per line).

MCP (Model Context Protocol) servers offer tools in a standard way. We start the
server as a child process, ask for its tools, and call them. Only stdio is used:
nothing goes over the network.

  client = McpClient(["npx", "-y", "@modelcontextprotocol/server-filesystem", "C:/Users/me/Documents/HAZEM"])
  client.call("list_directory", {"path": "C:/Users/me/Documents/HAZEM"})
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections import deque

PROTOCOL_VERSION = "2025-06-18"


class McpError(RuntimeError):
    pass


class McpClient:
    def __init__(self, command: list[str], cwd: str | None = None, env: dict | None = None,
                 start_timeout_s: float = 60.0):
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

    # ------------------------------------------------------------------ process

    def _resolve(self) -> list[str]:
        exe = shutil.which(self.command[0])         # finds npx.cmd on Windows
        if exe is None:
            raise McpError(f"{self.command[0]!r} not found. Install Node.js (it brings npx), "
                           "or set tools.files.command in config.yaml")
        return [exe] + self.command[1:]

    def start(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            env = {**os.environ, **(self.env or {})}
            self._proc = subprocess.Popen(
                self._resolve(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=self.cwd, env=env, bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True).start()
            threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True).start()
            result = self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                "clientInfo": {"name": "voice-agent", "version": "0.1"}}, self.start_timeout_s)
            self.server_info = result.get("serverInfo", {})
            self._notify("notifications/initialized")

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            proc.kill()

    # ------------------------------------------------------------------ messages

    def _send(self, msg: dict) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            raise McpError(f"MCP server is not running. {self._last_error()}")
        
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        
        with self._write_lock:
            try:
                proc.stdin.write(data)
                proc.stdin.flush() # Ensure data is actually sent
            except (OSError, BrokenPipeError) as e:
                # If the pipe is broken, the server likely crashed or closed
                raise McpError(f"Failed to send message to MCP server: {e}")


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

    # ------------------------------------------------------------------ MCP

    def list_tools(self) -> list[dict]:
        self.start()
        return self._request("tools/list", {}, 15.0).get("tools", [])

    def call(self, name: str, arguments: dict, timeout_s: float = 30.0) -> tuple[bool, str]:
        """(ok, text). The text joins the text parts of the result."""
        self.start()
        result = self._request("tools/call", {"name": name, "arguments": arguments}, timeout_s)
        parts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
        return not result.get("isError", False), "\n".join(p for p in parts if p)
