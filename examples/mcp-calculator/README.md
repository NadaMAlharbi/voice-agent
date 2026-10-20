# mcp-calculator: the smallest useful MCP server

An independent project with one MCP tool, `add_numbers(a, b)`. HAZEM starts it as a
child process and talks to it over stdio. Copy this folder to start a new MCP server.

```
examples/mcp-calculator/
  pyproject.toml                  its own dependencies (only "mcp"), its own uv.lock
  src/mcp_calculator/server.py    the server: one decorated function = one tool
```

## Run and test it on its own

```powershell
cd examples\mcp-calculator
uv sync                                   # first time only, needs internet (installs "mcp")
uv run mcp-calculator                     # starts and waits on stdin; nothing is printed. Ctrl+C to stop.
```

Interactive test page (needs Node.js), no HAZEM involved:

```powershell
npx @modelcontextprotocol/inspector uv run --directory examples/mcp-calculator mcp-calculator
```

From the HAZEM project folder, through HAZEM's own MCP client:

```powershell
uv run mcp-test                                            # lists servers and tools
uv run mcp-test calculator add_numbers "{\"a\": 2, \"b\": 3}"
```

## What the tool returns

| Call | Result |
|---|---|
| `add_numbers(a=2, b=3)` | `isError: false`, `structuredContent: {"a": 2, "b": 3, "sum": 5}` |
| `add_numbers(a="x", b=3)` | `isError: true`, "Input should be a valid number" (checked by the SDK from the type hints) |
| `add_numbers(a=1e308, b=1e308)` | `isError: true`, "the numbers are too large or not finite" (our `ToolError`) |

## Make your own tool

```python
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))   # readOnlyHint: HAZEM may skip "yes or no"
def my_tool(city: str, days: int = 1) -> MyResult:           # type hints become the input schema
    """One clear sentence: what it does and when to use it."""  # the docstring becomes the description
    ...
    raise ToolError("reason the user should hear")            # expected failure
    return {...}                                              # TypedDict -> structuredContent
```
