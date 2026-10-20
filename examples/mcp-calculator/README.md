# mcp-calculator

A small, independent MCP server with one tool: `add_numbers(a, b)`.
It shows how another project can give HAZEM new tools. Copy this folder to start your own server.

```
examples/mcp-calculator/
  pyproject.toml                  its own dependencies (only "mcp") and its own uv.lock
  src/mcp_calculator/server.py    the server: one decorated function = one tool
```

## Run it

```powershell
cd examples\mcp-calculator
uv sync                  # once, needs internet (installs "mcp")
uv run mcp-calculator    # starts and waits for MCP messages on stdin. Ctrl+C to stop.
```

Nothing is printed on start. This is normal: stdout is used only for MCP messages.

## Check it

On its own, with the MCP Inspector web page (needs Node.js):

```powershell
npx @modelcontextprotocol/inspector uv run --directory examples/mcp-calculator mcp-calculator
```

Through HAZEM, from the HAZEM project folder:

```powershell
uv run mcp-test calculator                      # starts the server, lists its tools
uv run mcp-test calculator add_numbers a=2 b=3  # calls the tool
```

## What the tool returns

| Call | Result |
|---|---|
| `add_numbers(a=2, b=3)` | `{"a": 2, "b": 3, "sum": 5}` |
| `add_numbers(a="x", b=3)` | Error: "Input should be a valid number" |
| `add_numbers(a=1e308, b=1e308)` | Error: "the numbers are too large or not finite" |

## Write your own tool

```python
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))   # read-only: HAZEM does not ask "yes or no"
def my_tool(city: str, days: int = 1) -> MyResult:           # type hints become the input schema
    """One clear sentence: what it does and when to use it."""  # the docstring is what the LLM reads
    ...
    raise ToolError("reason the user should hear")            # expected failure
    return {...}                                              # a TypedDict becomes JSON for HAZEM
```

To connect it to HAZEM, follow [MCP_GUIDE.md](../../MCP_GUIDE.md).
