"""Tool framework: what the assistant can DO, not just say."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

SAFE, CONFIRM = "safe", "confirm"


@dataclass
class ToolResult:
    ok: bool
    data: dict = field(default_factory=dict)   # facts for the LLM 
    error: str = ""
    image: str | None = None                   # path of a screenshot to show in the UI
    ms: float = 0.0

    def for_llm(self) -> str:
        d = {"ok": self.ok, **self.data}
        if self.error:
            d["error"] = self.error
        return json.dumps(d, ensure_ascii=False)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                                   # JSON schema "properties"
    handler: Callable[..., ToolResult]                 # blocking; runs in a worker thread
    required: list[str] = field(default_factory=list)
    risk: str | Callable[[dict], str] = SAFE           # or a function of the arguments
    timeout_s: float = 20.0

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.parameters, "required": self.required}}}

    def risk_of(self, args: dict) -> str:
        return self.risk(args) if callable(self.risk) else self.risk


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def needs_confirmation(self, name: str, args: dict) -> bool:
        t = self.get(name)
        return t is not None and t.risk_of(args) == CONFIRM

    def run_sync(self, name: str, args: dict | None) -> ToolResult:
        t0 = time.perf_counter()
        tool = self.get(name)
        if tool is None:
            return ToolResult(False, error=f"unknown tool {name!r}")
        args = clean_args(tool, args or {})
        try:
            res = tool.handler(**args)
        except TypeError as e:
            res = ToolResult(False, error=f"bad arguments: {e}")
        except Exception as e:  # noqa: BLE001  (a tool must never kill the turn)
            res = ToolResult(False, error=f"{type(e).__name__}: {e}")
        res.ms = (time.perf_counter() - t0) * 1000
        return res

    async def run(self, name: str, args: dict | None) -> ToolResult:
        tool = self.get(name)
        timeout = tool.timeout_s if tool else 5
        try:
            return await asyncio.wait_for(asyncio.to_thread(self.run_sync, name, args), timeout)
        except asyncio.TimeoutError:
            return ToolResult(False, error=f"{name} timed out after {timeout:.0f}s")


def clean_args(tool: Tool, args: Any) -> dict:
    """LLMs sometimes send arguments as a JSON string, extra keys or wrong types."""
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        return {}
    out = {}
    for k, v in args.items():
        spec = tool.parameters.get(k)
        if spec is None:
            continue
        typ = spec.get("type")
        try:
            if typ == "integer" and not isinstance(v, bool):
                v = int(float(v))
            elif typ == "number" and not isinstance(v, bool):
                v = float(v)
            elif typ == "boolean" and isinstance(v, str):
                v = v.strip().lower() in ("true", "1", "yes")
            elif typ == "string" and not isinstance(v, str):
                v = str(v)
        except (TypeError, ValueError):
            continue
        if "enum" in spec and v not in spec["enum"]:
            continue
        out[k] = v
    return out
