"""A minimal MCP server: one tool, add_numbers, over stdio.

Run it by hand (it waits for JSON-RPC messages on stdin; Ctrl+C to stop):
    uv run --directory examples/mcp-calculator mcp-calculator

Test it without HAZEM (opens a web page to list and call the tools; needs Node.js):
    npx @modelcontextprotocol/inspector uv run --directory examples/mcp-calculator mcp-calculator

Rules for stdio servers:
  - stdout carries the protocol. Never print() to stdout; log to stderr instead.
  - For a failure you expect (bad input), raise ToolError("reason"). The caller gets
    isError=true and your reason. Any other exception is treated as a crash: the caller
    only sees "Error executing tool add_numbers" and the traceback goes to stderr.
  - Return a TypedDict (or pydantic model): the SDK then publishes an output schema
    and sends the result as structuredContent (JSON) as well as text.
"""

from __future__ import annotations

import logging
import math
import sys
from typing import TypedDict

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="mcp-calculator: %(message)s")
log = logging.getLogger("mcp-calculator")

mcp = MCPServer(
    "calculator",
    instructions="Adds numbers. Use add_numbers to add two numbers.",
)


class AddResult(TypedDict):
    a: int | float
    b: int | float
    sum: int | float


def _tidy(x: float) -> int | float:
    """2.0 -> 2 (sounds better when spoken); other values unchanged."""
    return int(x) if x.is_integer() and abs(x) < 1e15 else x


@mcp.tool(annotations=ToolAnnotations(title="Add numbers", readOnlyHint=True, idempotentHint=True,
                                      openWorldHint=False))
def add_numbers(a: float, b: float) -> AddResult:
    """Add two numbers and return their sum."""
    total = a + b
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(total)):
        raise ToolError("the numbers are too large or not finite")
    log.info("add_numbers(%s, %s) = %s", a, b, total)
    return {"a": _tidy(a), "b": _tidy(b), "sum": _tidy(total)}


def main() -> None:
    mcp.run()                       # stdio transport (default)


if __name__ == "__main__":
    main()
