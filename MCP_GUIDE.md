# MCP Guide: connect your project to HAZEM

This guide is for developers who want HAZEM, the voice agent, to use tools from their own project.

**In short:** your project ships an MCP server. You add a few lines to HAZEM's
`config/config.yaml`. HAZEM starts your server, reads its tools, and the LLM can use them
by voice in English and Arabic. You do not change any HAZEM code.

The example used in every step is the calculator server in `examples/mcp-calculator`.

Contents

1. [How it works](#1-how-it-works)
2. [The MCP files in HAZEM](#2-the-mcp-files-in-hazem)
3. [Step by step with the calculator](#3-step-by-step-with-the-calculator)
4. [Connect your own project](#4-connect-your-own-project)
5. [Rules for a good MCP server](#5-rules-for-a-good-mcp-server)
6. [Configuration reference](#6-configuration-reference)
7. [Troubleshooting](#7-troubleshooting)
8. [Limits](#8-limits)

---

## 1. How it works

MCP (Model Context Protocol) is a standard way for an AI app to use tools that live in another
program. HAZEM starts your server as a child process and talks to it with JSON messages over
stdin/stdout (stdio). Nothing goes over the network.

```text
HAZEM                                          Your MCP server (separate process)
  |-- initialize ----------------------------->  "I am calculator"
  |-- tools/list ----------------------------->  add_numbers(a, b)
  |   HAZEM shows the LLM a tool named calculator__add_numbers
  |
User: "Add 2 and 3 with the MCP calculator"
  |-- tools/call add_numbers {a: 2, b: 3} ---->
  |<-- {"a": 2, "b": 3, "sum": 5} -------------
HAZEM: "2 plus 3 is 5."
```

Three things to know:

| Rule | Example |
|---|---|
| Each tool is named `<server name>__<tool name>` | `calculator__add_numbers` |
| A tool asks "yes or no" first unless it is read-only | `readOnlyHint=True` or `safe:` in config skips the question |
| A broken server never stops HAZEM | It is skipped and logged; all other tools still work |

---

## 2. The MCP files in HAZEM

| File | Role | Who uses it |
|---|---|---|
| `src/voice_agent/tools/mcp_client.py` | **The connection.** Starts one MCP server process and speaks the protocol with it: `initialize`, `tools/list`, `tools/call`. | `mcp_bridge.py` and the file tools |
| `src/voice_agent/tools/mcp_bridge.py` | **The adapter.** Reads `tools.mcp_servers` from `config.yaml`, starts one client per server, turns each MCP tool into a normal HAZEM tool. | HAZEM and `mcp_cli.py` |
| `src/voice_agent/tools/mcp_cli.py` | **The check command** (`uv run mcp-test`). Starts the servers from `config.yaml`, lists their tools, and can call one, without voice or LLM. | You, the developer |
| `config/config.yaml` | Where you add your server (`tools.mcp_servers`). | You |

### `mcp_client.py` vs `mcp_cli.py`

| | `mcp_client.py` | `mcp_cli.py` |
|---|---|---|
| What it is | A library class, `McpClient` | A command-line program, `uv run mcp-test` |
| Level | Low: one process, raw JSON-RPC messages | High: all servers in `config.yaml`, through the bridge |
| Used by | HAZEM at run time (every MCP tool call goes through it) | A person, to check a server before using it by voice |
| Input | A command list, e.g. `["uv", "run", "mcp-calculator"]` | Server name, tool name, `key=value` arguments |
| Output | The raw MCP result (`content`, `isError`, `structuredContent`) | Printed status and the exact result the LLM will see |
| Knows about `config.yaml`? | No | Yes |

The chain is: `mcp_cli.py` -> `mcp_bridge.py` -> `mcp_client.py` -> your server.
HAZEM uses the same chain without `mcp_cli.py`. So if `mcp-test` works, your
server works in HAZEM too.

---

## 3. Step by step with the calculator

Run all commands in PowerShell from the HAZEM project folder.

### Step 1. Look at the server

`examples/mcp-calculator` is an independent project with its own `pyproject.toml`.
Its only dependency is the official MCP Python SDK (`mcp>=2.3,<3`). The whole tool is one function:

```python
# examples/mcp-calculator/src/mcp_calculator/server.py
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

mcp = MCPServer("calculator")

@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def add_numbers(a: float, b: float) -> AddResult:
    """Add two numbers and return their sum."""
    total = a + b
    if not math.isfinite(total):
        raise ToolError("the numbers are too large or not finite")
    return {"a": a, "b": b, "sum": total}

def main() -> None:
    mcp.run()          # stdio
```

| Part | Becomes |
|---|---|
| Function name | Tool name (`add_numbers`) |
| Docstring | Tool description (the LLM reads it to decide when to use the tool) |
| Type hints | Input schema (`a: number, b: number`) |
| `readOnlyHint=True` | HAZEM does not ask "yes or no" |
| `ToolError("...")` | An error message the user hears |
| Returned dict | The JSON result HAZEM gets |

`pyproject.toml` defines the start command:

```toml
[project.scripts]
mcp-calculator = "mcp_calculator.server:main"
```

### Step 2. Install the server

```powershell
uv sync --directory examples/mcp-calculator      # once, needs internet
```

This creates `examples/mcp-calculator/.venv`. The server's packages stay separate from HAZEM's.

### Step 3. Add the server to `config/config.yaml`

The calculator is already there. This is the entry:

```yaml
tools:
  mcp_servers:
    calculator:                  # server name = prefix of its tool names
      enabled: true
      command: [uv, run, --directory, examples/mcp-calculator, mcp-calculator]
      tools: [add_numbers]       # which tools HAZEM may use
      confirm: []                # tools that always ask "yes or no"
      safe: []                   # tools that never ask
      timeout_s: 30              # max seconds per call
      start_timeout_s: 120       # first start may install packages
      hint: "To add two numbers with the MCP calculator, use calculator__add_numbers."
```

`command` is the same command you would type to start the server. Relative paths are
relative to the HAZEM folder.

### Step 4. Check the server with `mcp-test`

```powershell
uv run mcp-test calculator
uv run mcp-test calculator add_numbers a=2 b=3
```

Expected output:

```text
starting 1 server(s) ...
[ OK ] calculator: calculator  ready in 0.8s, 1 tool(s)
       calculator__add_numbers                  risk=safe     (a, b)
                                                Add two numbers and return their sum.

calling calculator__add_numbers({"a": 2, "b": 3})   risk=safe
[ OK ] 2 ms
LLM sees: {"ok": true, "a": 2, "b": 3, "sum": 5}
```

`[ OK ]` means the server is ready. `LLM sees:` is exactly what the LLM will receive.

### Step 5. Start HAZEM

```powershell
uv run serve --mode assistant
```

Look for this line in the log:

```text
tools: MCP calculator ready in 1.0s (calculator): calculator__add_numbers
```

### Step 6. Use it by voice

Open http://127.0.0.1:8000 and say: **"Add 25 and 17 with the MCP calculator."**

The server log shows:

```text
tool calculator__add_numbers({'a': 25, 'b': 17}) via llm -> ok=True {'a': 25, 'b': 17, 'sum': 42}
```

The browser's activity panel shows the same call.

Note: for "how much is 2 plus 3", HAZEM uses its built-in `calculate` tool (Windows Calculator).
Say "with the MCP calculator" to use the MCP server.

---

## 4. Connect your own project

Example: an AIS project in `C:\work\ais-mcp` with three tools.

### On your project's side

1. Copy `examples/mcp-calculator` and rename it (folder, package, script name).
2. Replace `add_numbers` with your tools. Follow the [rules](#5-rules-for-a-good-mcp-server).
3. Install it: `uv sync --directory C:/work/ais-mcp`.
4. Check it starts: `uv run --directory C:/work/ais-mcp ais-mcp` (it waits silently; Ctrl+C).

### On HAZEM's side

1. Add an entry to `config/config.yaml`:

   ```yaml
   tools:
     mcp_servers:
       ais:
         command: [uv, run, --directory, C:/work/ais-mcp, ais-mcp]
         tools: [get_anomalies, vessel_info, acknowledge_alert]
         confirm: [acknowledge_alert]          # changes data: always ask
         hint: "For vessel anomalies or AIS alerts, use the ais__ tools."
   ```

2. Check it:

   ```powershell
   uv run mcp-test ais
   uv run mcp-test ais get_anomalies area=Jeddah
   ```

3. Restart HAZEM (`uv run serve --mode assistant`) and look for `tools: MCP ais ready ...`.
4. Try it by voice in English and Arabic. Check the activity panel for the tool call.
5. If the LLM picks the wrong tool, improve the tool's docstring or the `hint:`, and keep
   the `tools:` list short.

To connect several projects, add one entry per server. Each runs in its own process.

### Other ways to start a server

| Server type | `command` |
|---|---|
| Python project with uv | `[uv, run, --directory, C:/work/proj, proj-server]` |
| Python with an existing venv | `[C:/work/proj/.venv/Scripts/python.exe, -m, proj.server]` |
| Node package | `[npx, -y, --prefer-offline, "@scope/server@1.2.3"]` |
| Compiled program | `[C:/tools/my-server.exe, --stdio]` |

---

## 5. Rules for a good MCP server

| Rule | Why |
|---|---|
| Use the stdio transport | HAZEM supports stdio only |
| Never `print()` to stdout; log to stderr | stdout carries the MCP messages |
| One clear sentence per docstring: what it does and when to use it | The LLM chooses tools by their description |
| Use simple arguments (strings, numbers, booleans) | A 9B model fills simple arguments best |
| Return short results | The answer is spoken |
| Answer within a few seconds | Voice users wait; the default limit is 30 s |
| Raise `ToolError("reason")` for expected failures | The user hears the reason; other exceptions only say "Error executing tool" |
| Set `readOnlyHint=True` on tools that only read | Otherwise HAZEM asks "yes or no" each time |
| Return a `TypedDict` | HAZEM gets clean JSON (`structuredContent`) |

Give the HAZEM side: the folder or package, the start command, the tool list (which ones
change data), and any environment variables.

---

## 6. Configuration reference

```yaml
tools:
  mcp_wait_s: 15                 # at startup, wait this long for servers before warming the LLM
  mcp_servers:
    <name>:                      # lowercase letters, digits, _; starts with a letter; max 24
      command: [...]             # REQUIRED: how to start the server, as a list
      enabled: true              # false = skip this server
      cwd: .                     # folder the command runs in (default: HAZEM folder)
      env: {API_URL: "http://localhost:9000", DEBUG: null}   # add variables; null removes one
      tools: [tool_a, tool_b]    # allow-list (default: every tool)
      confirm: [tool_b]          # always ask "yes or no"
      safe: [tool_a]             # never ask
      timeout_s: 30              # max seconds per call
      start_timeout_s: 60        # max seconds to start
      hint: "One line telling the LLM when to use these tools."
```

When HAZEM asks "yes or no":

| Situation | Asks? |
|---|---|
| Tool is in `confirm:` | Always |
| Tool is in `safe:` | Never |
| Server marks the tool `readOnlyHint: true` | No |
| Anything else | Yes |

What the LLM gets back:

| Server answer | LLM gets |
|---|---|
| JSON result (`structuredContent`) | `{"ok": true, ...result...}` |
| Text only | `{"ok": true, "result": "<text, max 3000 chars>"}` |
| `ToolError` | `{"ok": false, "error": "<your message>"}` |
| Server crashed or timed out | `{"ok": false, "error": "<server>: <reason>"}` |

Typos in the config are reported at startup, for example `unknown setting(s) ['comand']`.

---

## 7. Troubleshooting

Start with `uv run mcp-test <name>`. It shows the same error as the HAZEM log.

| You see | Fix |
|---|---|
| `'uv' not found on PATH` | Install the program, or use its full path in `command` |
| `MCP server stopped before answering initialize ... Server said: ...` | The server crashed at start. Read "Server said", or run the command by hand |
| `No module named 'mcp'` | Install the server's packages: `uv sync --directory <server folder>` |
| `did not answer initialize within 60 s` | First start is slow: raise `start_timeout_s` or run the command by hand once |
| `did not answer tools/call within 30 s` | Make the tool faster, or raise `timeout_s` |
| `has no tool(s) ['x']` | A name in `tools:` is wrong. `mcp-test <name>` shows the real names |
| HAZEM asks "yes or no" for a read-only tool | Set `readOnlyHint=True` in the server, or add the tool to `safe:` |
| The LLM never uses the tool | Improve the docstring, add a `hint:`, shorten `tools:` |
| The Inspector shows parse errors | The server prints to stdout. Log to stderr instead |
| A fixed server is still missing | Restart HAZEM. A server that failed at start is not retried |

---

## 8. Limits

- stdio servers only (no HTTP servers yet).
- Tools are read when the server starts. If you add tools, restart HAZEM.
- A server that fails at start stays off until HAZEM restarts. A server that crashes later
  is restarted on the next call.
- Only text and JSON results are used; images from a tool are not shown.
- MCP tools are used through the LLM, not the fast router.
- Every tool adds to the LLM's tool list. Keep `tools:` short: a 9B model chooses worse with many tools.
- MCP servers start only with the real Windows desktop (not with `tools.desktop: dry-run`).
