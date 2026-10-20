# MCP User Guide for HAZEM

This guide explains how HAZEM (the voice agent) uses MCP to get tools from other
projects, how to run and test it, and how to connect a new project later.
Short version: **another project ships an MCP server, you add a few lines to
`config/config.yaml`, and HAZEM can use its tools. No HAZEM code changes.**

Contents

1. [MCP in 5 minutes](#1-mcp-in-5-minutes)
2. [How HAZEM uses MCP (architecture)](#2-how-hazem-uses-mcp)
3. [New and changed files](#3-new-and-changed-files)
4. [Setup and run (Windows)](#4-setup-and-run-windows)
5. [Configuration reference](#5-configuration-reference)
6. [Testing](#6-testing)
7. [Troubleshooting](#7-troubleshooting)
8. [Connecting a future project](#8-connecting-a-future-project)
9. [Config only, or code change?](#9-config-only-or-code-change)
10. [Current limits](#10-current-limits)

The checklist to give other developers is in [MCP_INTEGRATION_CHECKLIST.md](MCP_INTEGRATION_CHECKLIST.md).

---

## 1. MCP in 5 minutes

MCP (Model Context Protocol) is a standard way for an AI app to use tools that live in
**another program**. It is just JSON messages. The LLM never sees MCP: HAZEM talks MCP to
the tool programs and shows their tools to the LLM like its own tools.

| Word | Meaning | In HAZEM |
|---|---|---|
| Host | The AI app with the LLM and the user | HAZEM |
| Server | A separate program that offers tools | `examples/mcp-calculator`, the Filesystem server, a future project |
| Client | HAZEM's connection to one server | `McpClient` in `tools/mcp_client.py` (one per server) |
| Tool | One function a server offers: name, description, input schema | `add_numbers(a, b)` |
| stdio | HAZEM starts the server as a child process and talks over its stdin/stdout | All servers today (no network) |

What happens with the calculator (you can see these messages with the Inspector, section 6):

```text
HAZEM → server   initialize                         "hello, I speak MCP 2025-06-18"
server → HAZEM   serverInfo: calculator              "hello, I am calculator"
HAZEM → server   tools/list
server → HAZEM   add_numbers(a: number, b: number)   + description, readOnlyHint=true
                 ... HAZEM shows the LLM a tool called calculator__add_numbers ...
user: "add 2 and 3 with the MCP calculator"
LLM  → HAZEM     call calculator__add_numbers {a: 2, b: 3}
HAZEM → server   tools/call add_numbers {a: 2, b: 3}
server → HAZEM   structuredContent {a: 2, b: 3, sum: 5}, isError=false
LLM  → user      "2 plus 3 is 5."
```

---

## 2. How HAZEM uses MCP

HAZEM has three kinds of tools. Only the third one is new.

| Kind | Where it lives | Add one by | Example |
|---|---|---|---|
| Native tools | `src/voice_agent/tools/*.py`, inside HAZEM | Writing Python in HAZEM | open_app, set_timer, calculate (Windows Calculator) |
| Hand-made MCP layer | `tools/files.py` over the Filesystem MCP server | Writing Python in HAZEM | list_files, read_file, delete_file |
| **Bridged MCP tools** | Any MCP server, any project | **Adding it to config.yaml** | calculator__add_numbers |

```text
                       ┌──────────────────────────── HAZEM ────────────────────────────┐
 voice → STT → router ─┤ LLM ← tool list ← ToolRegistry (per browser tab, tools/base.py) │
                       │                     ├─ native tools (tools/*.py)              │
                       │                     ├─ files.py ── McpClient ── stdio ────────┼─→ Filesystem server (npx)
                       │                     └─ McpHub (tools/mcp_bridge.py)           │
                       │                          ├─ McpClient ── stdio ───────────────┼─→ calculator (examples/mcp-calculator)
                       │                          └─ McpClient ── stdio ───────────────┼─→ your next project
                       │  "yes or no" confirmation, timeouts, logs: same for all tools │
                       └───────────────────────────────────────────────────────────────┘
```

### What happens at startup (`uv run serve --mode assistant`)

1. `app.py` reads `tools.mcp_servers` and creates one `McpHub` for the whole process.
2. The hub starts every enabled server **in parallel, in the background**, and asks each one for its tools.
3. Startup waits up to `tools.mcp_wait_s` (15 s) for them, so the LLM warm-up already knows their tools.
4. Log lines tell you the result:
   `tools: MCP calculator ready in 1.0s (calculator): calculator__add_numbers`
   or `tools: MCP calculator failed to start (<reason>)`.
5. A server that fails is skipped. Everything else (native tools, other servers) still works.
6. On shutdown (Ctrl+C) the hub stops all server processes.

### What happens on each call

1. The LLM picks `calculator__add_numbers`. The name is always `<server name>__<tool name>`.
2. The registry cleans the arguments (drops unknown keys, turns `"2"` into `2` for number fields).
3. If the tool is "confirm" risk, HAZEM first asks "Should I go ahead? Say yes or no."
4. The bridge sends `tools/call` and waits up to `timeout_s`.
5. The result becomes a normal `ToolResult`:

| Server answer | What the LLM gets |
|---|---|
| `structuredContent` (JSON object) | `{"ok": true, ...that object...}` |
| Text only | `{"ok": true, "result": "<text, max 3000 chars>"}` |
| `isError: true` | `{"ok": false, "error": "<server's message>"}` |
| Server crashed / timed out / not running | `{"ok": false, "error": "calculator server: <reason + server's stderr>"}` |

If a server crashes, the next call starts it again automatically.

### Safety rule (who asks "yes or no")

| Situation | Result |
|---|---|
| Tool listed in `confirm:` | Always asks |
| Tool listed in `safe:` | Never asks |
| Server marks the tool `readOnlyHint: true` | Does not ask |
| Anything else | **Asks** (safe default) |

Server hints are only hints: for a project you do not fully trust, put its write tools in `confirm:`.

---

## 3. New and changed files

### New

| File | What it does |
|---|---|
| `src/voice_agent/tools/mcp_bridge.py` | The generic bridge. Reads `tools.mcp_servers`, starts servers (`McpHub`), converts each MCP tool to a `Tool`, applies the allow-list, risk rule and timeout, converts results and errors. |
| `src/voice_agent/tools/mcp_cli.py` | The `uv run mcp-test` command: list servers and tools, call a tool, without the voice loop. |
| `examples/mcp-calculator/` | A complete independent MCP project: own `pyproject.toml`, own `uv.lock`, `src/mcp_calculator/server.py` with `add_numbers`. Copy it to start a new server. |
| `tests/test_mcp_bridge.py` | Tests for config checks, conversion rules, discovery, calls, errors, crash and restart, timeout, failure isolation, Toolbox/Agent wiring, the real calculator and the CLI. |
| `docs/MCP_GUIDE.md`, `docs/MCP_INTEGRATION_CHECKLIST.md` | This guide and the checklist. |

### Changed (small, existing behaviour kept)

| File | Change |
|---|---|
| `tools/mcp_client.py` | New `call_tool()` returns the full MCP result (needed for `structuredContent`); `call()` unchanged for files.py. Reads paged tool lists (`nextCursor`). `env` values of `None` remove a variable (the bridge removes `VIRTUAL_ENV`, so a uv server uses its own `.venv`). Clearer "not found on PATH" message. |
| `tools/base.py` | `Tool` has an optional `defs` field, so input schemas with `$defs` (nested types) reach the LLM complete. Native tools are unaffected (empty by default). |
| `tools/toolbox.py` | `Toolbox(..., mcp=hub)` adds the bridged tools after the native ones. A native tool with the same name always wins. |
| `server/agent.py` | The system note lists the `hint:` lines of ready MCP servers. |
| `server/pipelines.py` | `make_agent()` passes the shared hub to each new `Toolbox`. |
| `server/app.py` | `_start_mcp_servers()` starts the hub in the background and logs each server; the lifespan waits up to `mcp_wait_s` before the LLM warm-up and closes the servers on shutdown. |
| `config.py` | `tools.mcp_servers` (default empty) and `tools.mcp_wait_s` (15). |
| `config/config.yaml` | A `mcp_servers:` section with the `calculator` example. |
| `pyproject.toml` | New script `mcp-test`. **No new dependency for HAZEM**: its MCP client is its own small file. |

### Dependencies

| Who | Needs | Why |
|---|---|---|
| HAZEM | nothing new | `tools/mcp_client.py` speaks MCP itself |
| `examples/mcp-calculator` | `mcp>=2.3,<3` (official MCP Python SDK), in its **own** project | Makes writing a server easy. In SDK 2.x the server class is `MCPServer` (it was `FastMCP` in 1.x) |
| You | `uv` on PATH (already used for HAZEM) | HAZEM starts the example with `uv run` |
| Optional | Node.js | Only for the MCP Inspector test page (and the existing file tools) |

---

## 4. Setup and run (Windows)

Run these in PowerShell in the HAZEM folder (`C:\Users\...\voice-agent`).

```powershell
# 1. HAZEM itself (once, after updating the code)
uv sync

# 2. The example server (once, needs internet: installs the "mcp" package into examples\mcp-calculator\.venv)
uv sync --directory examples/mcp-calculator

# 3. Check the server through HAZEM's own MCP client (no voice, no LLM)
uv run mcp-test
uv run mcp-test calculator add_numbers a=2 b=3

# 4. Start HAZEM
uv run serve --mode assistant
```

Expected output of step 3:

```text
starting 1 server(s) ...
[ OK ] calculator: calculator  ready in 0.8s, 1 tool(s)
       calculator__add_numbers                  risk=safe     (a, b)
                                                Add two numbers and return their sum.

calling calculator__add_numbers({"a": 2, "b": 3})   risk=safe
[ OK ] 2 ms
LLM sees: {"ok": true, "a": 2, "b": 3, "sum": 5}
```

Expected log line in step 4:

```text
tools: MCP calculator ready in 1.0s (calculator): calculator__add_numbers
```

Then open http://127.0.0.1:8000 and say: **"Add 25 and 17 with the MCP calculator."**
The server log (the PowerShell window) shows
`tool calculator__add_numbers({'a': 25, 'b': 17}) via llm -> ok=True {'a': 25, 'b': 17, 'sum': 42}`,
and the browser's activity log shows the tool call and its result.

Note: for a general question like "how much is 2 plus 3", the LLM is told to use the
built-in `calculate` tool (Windows Calculator). Both give the right answer; say "with the
MCP calculator" to test the MCP path. To turn the example off: `enabled: false`.

---

## 5. Configuration reference

```yaml
tools:
  mcp_wait_s: 15                 # startup waits this long for servers before warming the LLM
  mcp_servers:
    <name>:                      # lowercase letters, digits, _; starts with a letter; max 24
      enabled: true              # false = skip it
      command: [uv, run, --directory, C:/work/other-project, other-server]   # REQUIRED, a list
      cwd: .                     # folder the command runs in (default: HAZEM's folder)
      env: {API_URL: "http://localhost:9000", DEBUG: null}   # add variables; null removes one
      tools: [tool_a, tool_b]    # allow-list (default: every tool of the server)
      confirm: [tool_b]          # always ask "yes or no"
      safe: [tool_a]             # never ask
      timeout_s: 30              # per call
      start_timeout_s: 60        # first start may install packages
      hint: "For vessel anomalies, use ais__get_anomalies."   # one line for the LLM
```

Mistakes are caught at startup with a clear message, for example
`unknown setting(s) ['comand']` or `command must be a list, e.g. [uv, run, my-server]`.

Command examples:

| Server type | command |
|---|---|
| Python project with uv (like the example) | `[uv, run, --directory, C:/work/proj, proj-server]` |
| Python file, project already has a .venv | `[C:/work/proj/.venv/Scripts/python.exe, -m, proj.server]` |
| Node package (pinned) | `[npx, -y, --prefer-offline, "@scope/server@1.2.3"]` |
| Compiled program | `[C:/tools/my-server.exe, --stdio]` |

---

## 6. Testing

| What | Command | Needs |
|---|---|---|
| All MCP tests (fake server + real calculator + CLI) | `uv run pytest -q tests/test_mcp_bridge.py` | uv; internet once for the example |
| File tools + MCP client tests | `uv run pytest -q tests/test_mcp_files.py` | Node.js for the real Filesystem server tests (skipped without it) |
| Everything | `uv run pytest -q` | |
| One server by hand | `uv run mcp-test <name>` and `uv run mcp-test <name> <tool> key=value ...` | |
| A server alone, with a web page | `npx @modelcontextprotocol/inspector uv run --directory examples/mcp-calculator mcp-calculator` | Node.js |
| By voice | Start `serve`, say the request, read the activity log | Ollama, models |

What the tests check:

| Area | Tests |
|---|---|
| Config | defaults, relative `cwd`, rejects bad names, string commands, typos, wrong types |
| Discovery | tool list across pages, allow-list, warning for allow-listed tools that do not exist, `$defs` kept |
| Invocation | arguments cleaned, Arabic text round trip, structured and text results |
| Errors | `isError` results, unknown tool, missing/invalid arguments, number overflow, server crash with its stderr, automatic restart, call timeout |
| Isolation | a broken server, a missing command and a disabled server do not stop a good one |
| Wiring | Toolbox has all native tools plus MCP tools; the agent's system note gets the hint |
| CLI | `mcp-test` arguments, exit codes, printed result |

---

## 7. Troubleshooting

Start with `uv run mcp-test <name>`: it shows the same error the server log shows.

| You see | Cause | Fix |
|---|---|---|
| `'uv' not found on PATH` (or any command) | The command is not installed or not on PATH | Install it, or use a full path in `command` |
| `MCP server stopped before answering initialize. Exit code 1. Server said: ...` | The server crashed at startup | Read the "Server said" part. Run the command by hand in a terminal to see the full error |
| `... ModuleNotFoundError: No module named 'mcp'` | The server's packages are not installed | `uv sync --directory <server folder>` (needs internet once) |
| `did not answer initialize within 60 s` | Slow first install, or the server waits for something | Raise `start_timeout_s`, or run the command by hand once |
| `did not answer tools/call within 30 s` | The tool is slow | Raise `timeout_s`, or make the tool faster (voice needs ~2 s) |
| `bad config.yaml tools.mcp_servers: unknown setting(s) ...` | A typo in config | Fix the key name (the message lists allowed keys) |
| `has no tool(s) ['x'] (check tools: in config)` | The allow-list names a tool the server does not have | Run `mcp-test <name>` to see the real names |
| Server is ready but the LLM never uses the tool | The description or hint is unclear, or too many tools | Improve the tool's docstring, add a `hint:`, keep the allow-list short |
| HAZEM asks "yes or no" for a read-only tool | The server did not set `readOnlyHint` | Add the tool to `safe:` or fix the server |
| Works in HAZEM but the Inspector or another MCP host shows parse errors | The server `print()`s to **stdout** (HAZEM skips non-JSON lines, stricter hosts do not) | Servers must log to stderr; stdout is only for MCP messages |
| `VIRTUAL_ENV ... does not match the project environment` warning | HAZEM's venv was activated | Harmless; the bridge already removes `VIRTUAL_ENV` for servers |
| A fixed server is still missing after it failed at startup | Failed servers are not retried until restart | Restart `serve` (a server that crashes *later* is restarted automatically) |

---

## 8. Connecting a future project

### Their side (the other developer)

1. Build an MCP server with stdio transport. Easiest: copy `examples/mcp-calculator`, rename it, replace `add_numbers` with their tools.
2. Follow [the checklist](MCP_INTEGRATION_CHECKLIST.md): clear descriptions, simple arguments, short results, `ToolError` for expected failures, `readOnlyHint` on read tools, logs to stderr only.
3. Test with the MCP Inspector.
4. Give you: the folder (or package name and version), the start command, the tool list with which ones change data, needed environment variables.

### Your side (HAZEM)

1. Put the project somewhere stable, for example `C:\work\ais-mcp`, and install it once: `uv sync --directory C:/work/ais-mcp`.
2. Add it to `config/config.yaml`:

   ```yaml
   tools:
     mcp_servers:
       ais:
         command: [uv, run, --directory, C:/work/ais-mcp, ais-mcp]
         tools: [get_anomalies, vessel_info, acknowledge_alert]
         confirm: [acknowledge_alert]
         hint: "Questions about vessel anomalies or AIS alerts: use the ais__ tools."
   ```

3. Check: `uv run mcp-test ais`, then `uv run mcp-test ais get_anomalies area=Jeddah`.
4. Restart `uv run serve --mode assistant` and look for `tools: MCP ais ready ...`.
5. Try it by voice in English and Arabic. Check the activity log for the tool call.
6. If the LLM picks the wrong tool: improve the hint, shorten the allow-list.

Several projects at once is the same: one entry per server. Each runs as its own process;
one broken server does not affect the others.

---

## 9. Config only, or code change?

| You want to | Config only | Code change in HAZEM |
|---|---|---|
| Add, remove, enable or disable a server | ✓ | |
| Choose which tools are exposed, which ask "yes or no", timeouts | ✓ | |
| Pass settings to a server (environment variables, working folder) | ✓ | |
| Tell the LLM when to use a server (`hint:`) | ✓ | |
| A server adds or changes tools | ✓ (restart `serve`) | |
| Have the router run an MCP tool **without the LLM** (instant voice commands) | | ✓ `tools/router.py` |
| A custom spoken sentence after a confirmed MCP action (today: "Done.") | | ✓ `tools/speech.py` |
| A custom "yes or no" question for an MCP tool (today: "Should I go ahead?") | | ✓ `speech.confirm_question` |
| Change arguments or results for voice (like files.py does for the Filesystem server) | | ✓ a new `tools/<name>.py` wrapper |
| Connect a server over the network (HTTP), with login | | ✓ add Streamable HTTP to `mcp_client.py` |
| Show images returned by a tool in the browser | | ✓ `mcp_bridge.to_tool_result` |

---

## 10. Current limits

- **stdio only.** Network servers (Streamable HTTP) are the next step; the config and bridge are ready for a second transport.
- Tools are read when a server starts. If a server changes its tool list while running, restart `serve`.
- A server that fails at startup stays off until `serve` restarts. A server that crashes later is restarted on the next call.
- Only text and JSON results are used. Images or files from a tool are noted (`other_content`) but not shown.
- Bridged tools are used through the LLM; the fast router does not know them.
- Each server's tools add to the LLM's tool list (HAZEM already has 38). Keep allow-lists short: a 9B model chooses worse with very many tools.
- MCP servers start only with a real desktop (not with `desktop: dry-run`), the same rule as the file tools.
