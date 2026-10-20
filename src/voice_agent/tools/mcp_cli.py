"""mcp-test: check the MCP servers in config.yaml without the voice loop.

  uv run mcp-test                                   # start every server, list its tools
  uv run mcp-test calculator                        # only this server
  uv run mcp-test calculator add_numbers a=2 b=3    # call a tool (key=value arguments)
  uv run mcp-test calculator add_numbers "{\\"a\\": 2, \\"b\\": 3}"   # or one JSON object

Calls go through the same path as the voice agent (bridge -> ToolRegistry), so the
printed result is exactly what the LLM receives. Exit code 0 = OK, 1 = a problem.
"""

from __future__ import annotations

import argparse
import json
import sys

from voice_agent.config import DEFAULT_CONFIG, load_config
from voice_agent.tools.base import ToolRegistry
from voice_agent.tools.mcp_bridge import McpHub, agent_tool_name, parse_servers, tool_risk


def parse_args_list(items: list[str]) -> dict:
    """["a=2", "b=x"] or ['{"a": 2}'] -> {"a": 2, "b": "x"}. Values are JSON when they parse."""
    if len(items) == 1 and items[0].lstrip().startswith("{"):
        value = json.loads(items[0])
        if not isinstance(value, dict):
            raise ValueError("the JSON arguments must be an object")
        return value
    out = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"bad argument {item!r}: use key=value")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw                       # plain text
    return out


def _print_server(hub: McpHub, name: str) -> bool:
    st, spec = hub.status[name], hub.specs[name]
    if st.state == "disabled":
        print(f"[SKIP] {name}: enabled: false")
        return True
    if st.state != "ready":
        print(f"[FAIL] {name}: {st.error}")
        print(f"       command: {' '.join(spec.command)}   (cwd {spec.cwd})")
        return False
    info = st.server_info
    print(f"[ OK ] {name}: {info.get('name', '?')} {info.get('version', '')} ready in {st.seconds:.1f}s, "
          f"{len(st.tools)} tool(s)")
    for t in st.tools:
        props = ", ".join((t.get("inputSchema") or {}).get("properties", {})) or "no arguments"
        first_line = ((t.get("description") or "").strip().splitlines() or [""])[0][:90]
        print(f"       {agent_tool_name(name, t['name']):<40} risk={tool_risk(spec, t):<8} ({props})")
        if first_line:
            print(f"       {'':<40} {first_line}")
    for m in st.missing:
        print(f"[WARN] {name}: allow-list names {m!r}, but the server has no such tool")
    return not st.missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check MCP servers from config.yaml (tools.mcp_servers)")
    ap.add_argument("server", nargs="?", help="server name (default: all)")
    ap.add_argument("tool", nargs="?",
                    help="tool to call: MCP name (add_numbers) or agent name (calculator__add_numbers)")
    ap.add_argument("arguments", nargs="*", help="key=value pairs, or one JSON object")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="config.yaml path")
    a = ap.parse_args(argv)

    try:
        specs = parse_servers(load_config(a.config).tools.mcp_servers)
    except ValueError as e:
        print(f"[FAIL] config: {e}")
        return 1
    if not specs:
        print("No MCP servers in config.yaml (tools.mcp_servers).")
        return 1
    if a.server:
        specs = [s for s in specs if s.name == a.server]
        if not specs:
            print(f"[FAIL] no server {a.server!r} in config.yaml")
            return 1
    hub = McpHub(specs)
    try:
        print(f"starting {len(specs)} server(s) ...", flush=True)
        hub.start_all()
        ok = all([_print_server(hub, s.name) for s in specs])
        if not a.tool:
            return 0 if ok else 1
        try:
            args = parse_args_list(a.arguments)
        except ValueError as e:                       # json.JSONDecodeError is a ValueError
            print(f"[FAIL] arguments: {e}")
            return 1
        reg = ToolRegistry(hub.tools())
        name = a.tool if reg.get(a.tool) else agent_tool_name(a.server, a.tool)
        tool = reg.get(name)
        if tool is None:
            print(f"[FAIL] no tool {a.tool!r}. Available: {reg.names()}")
            return 1
        print(f"\ncalling {name}({json.dumps(args, ensure_ascii=False)})   risk={tool.risk_of(args)}")
        res = reg.run_sync(name, args)
        print(f"[{' OK ' if res.ok else 'FAIL'}] {res.ms:.0f} ms")
        print(f"LLM sees: {res.for_llm()}")
        return 0 if res.ok else 1
    finally:
        hub.close()


if __name__ == "__main__":
    sys.exit(main())
