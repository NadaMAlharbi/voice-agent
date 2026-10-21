# MCP Guide for HAZEM

This guide is for developers who want to give HAZEM new tools through MCP: how HAZEM uses MCP servers, how to create one, connect it, configure its tools and confirmation rules, and test it.

Short version: **another project ships an MCP server, you add a few lines to `config/config.yaml`, and HAZEM can use its tools. No HAZEM code changes.**

Contents

1. [MCP in 5 minutes](#1-mcp-in-5-minutes)
2. [How HAZEM uses MCP](#2-how-hazem-uses-mcp)
3. [Create an MCP server](#3-create-an-mcp-server)
4. [Connect it to HAZEM](#4-connect-it-to-hazem)
5. [Run and test](#5-run-and-test)
6. [Example: MCP Notes server](#6-example-mcp-notes-server)
7. [MCP Notes vs. built-in notes](#7-mcp-notes-vs-built-in-notes)
8. [Troubleshooting](#8-troubleshooting)
9. [Config only, or code change?](#9-config-only-or-code-change)
10. [Current limits](#10-current-limits)

---

## 1. MCP in 5 minutes

MCP (Model Context Protocol) is a standard way for an AI app to use tools that live in **another program**. It is just JSON messages. The LLM never sees MCP: HAZEM talks MCP to the tool programs and shows their tools to the LLM like its own tools.

| Word   | Meaning                                                      | In HAZEM                                      |
| ------ | ------------------------------------------------------------ | --------------------------------------------- |
| Host   | The AI app with the LLM and the user                         | HAZEM                                         |
| Server | A separate program that offers tools                         | `examples/mcp-calculator`, your next project  |
| Client | HAZEM's connection to one server                             | `McpClient` in `tools/mcp_client.py`          |
| Tool   | One function a server offers: name, description, input schema | `add_numbers(a, b)`                           |
| stdio  | HAZEM starts the server as a child process and talks over its stdin/stdout | All servers today (no network)  |

What happens with the calculator:

```text
HAZEM → server   initialize                         "hello, I speak MCP"
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

HAZEM has three kinds of tools:

| Kind              | Where it lives                                    | Add one by                 | Example                     |
| ----------------- | ------------------------------------------------- | -------------------------- | --------------------------- |
| Native tools      | `src/voice_agent/tools/*.py`, inside HAZEM        | Writing Python in HAZEM    | `open_app`, `add_note`      |
| File tools        | `tools/files.py` over the Filesystem MCP server   | Writing Python in HAZEM    | `list_files`, `read_file`   |
| **Bridged MCP tools** | Any MCP server, any project                   | **Adding it to `config.yaml`** | `calculator__add_numbers` |

The bridge is `src/voice_agent/tools/mcp_bridge.py`. HAZEM's MCP client (`tools/mcp_client.py`) is its own small file, so HAZEM needs no MCP package. Only the server projects use the `mcp` SDK.

### At startup (`uv run serve --mode assistant`)

1. HAZEM reads `tools.mcp_servers` and starts every enabled server **in parallel, in the background**.
2. It asks each server for its tools and waits up to `tools.mcp_wait_s` (15 s), so the LLM warm-up already knows them.
3. The log shows the result: `tools: MCP calculator ready in 1.0s (calculator): calculator__add_numbers` or `tools: MCP calculator failed to start (<reason>)`.
4. A server that fails is skipped. Native tools and other servers still work.
5. On shutdown (Ctrl+C), HAZEM stops all server processes.

### On each call

1. The LLM picks a tool. Its name is always `<server name>__<tool name>`.
2. HAZEM cleans the arguments (drops unknown keys, turns `"2"` into `2` for number fields).
3. If the tool is "confirm" risk, HAZEM first asks "Should I go ahead? Say yes or no."
4. HAZEM sends `tools/call` and waits up to `timeout_s`.
5. The result becomes what the LLM sees:

| Server answer                          | What the LLM gets                                   |
| -------------------------------------- | --------------------------------------------------- |
| `structuredContent` (JSON object)      | `{"ok": true, ...that object...}`                   |
| Text only                              | `{"ok": true, "result": "<text, max 3000 chars>"}`  |
| `isError: true`                        | `{"ok": false, "error": "<server's message>"}`      |
| Server crashed, timed out, not running | `{"ok": false, "error": "<server> server: <reason>"}` |

If a server crashes, the next call starts it again automatically.

### Who asks "yes or no"

| Situation                                   | Result                  |
| ------------------------------------------- | ----------------------- |
| Tool listed in `confirm:`                   | Always asks             |
| Tool listed in `safe:`                      | Never asks              |
| Server marks the tool `readOnlyHint: true`  | Does not ask            |
| Anything else                               | **Asks** (safe default) |

Server hints are only hints. For a project you do not fully trust, put its write tools in `confirm:`.

---

## 3. Create an MCP server

The easiest start is to copy `examples/mcp-calculator/`. It is a complete, independent project:

```
examples/mcp-calculator/
├── pyproject.toml                  # its own dependencies (only "mcp") and the start command
├── uv.lock
└── src/mcp_calculator/
    ├── __init__.py
    └── server.py                   # one decorated function = one tool
```

### How a tool is written

```python
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))   # readOnlyHint: HAZEM may skip "yes or no"
def my_tool(city: str, days: int = 1) -> MyResult:           # type hints become the input schema
    """One clear sentence: what it does and when to use it."""  # the docstring becomes the description
    ...
    raise ToolError("reason the user should hear")            # expected failure
    return {...}                                              # TypedDict -> structuredContent
```

| Part in the code                 | What it becomes                                                  |
| -------------------------------- | ---------------------------------------------------------------- |
| `@mcp.tool(...)`                 | An MCP tool                                                      |
| Type hints (`a: float`)          | The input schema the LLM fills in                                |
| Docstring                        | The tool description. The LLM chooses tools by it.               |
| `readOnlyHint=True`              | "I change nothing": HAZEM does not ask "yes or no"               |
| `-> TypedDict`                   | The result is sent as `structuredContent` (JSON)                 |
| `raise ToolError("...")`         | `isError: true` with your message                                |

### Rules for stdio servers

- **Never `print()` to stdout.** stdout carries the MCP messages. Log to stderr.
- Raise `ToolError("reason")` for expected failures (bad input, not found). Any other exception is treated as a crash.
- Keep arguments simple and results short: the reply is spoken.
- Mark read tools with `readOnlyHint=True`, and destructive tools with `destructiveHint=True`.
- The `mcp` SDK 2.x names the server class `MCPServer` (it was `FastMCP` in 1.x).

### The start command

`pyproject.toml` must define a script and a build system. Without `[build-system]`, `uv` does not install the project, the script is never created, and the server fails with `Failed to spawn`.

```toml
[project.scripts]
my-server = "my_server.server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/my_server"]
```

The package folder uses an underscore (`src/my_server`); the command uses a hyphen (`my-server`).

---

## 4. Connect it to HAZEM

1. Install the server once (needs internet): `uv sync --directory <server folder>`.
2. Add it under `tools.mcp_servers` in `config/config.yaml`.
3. Check it with `uv run mcp-test <name>`, then restart `serve`.

### Configuration reference

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

These are the only allowed keys. A typo is reported at startup, for example `unknown setting(s) ['comand']`.

| Server type                                  | `command`                                                    |
| -------------------------------------------- | ------------------------------------------------------------ |
| Python project with uv (like the examples)   | `[uv, run, --directory, C:/work/proj, proj-server]`          |
| Python file, project already has a `.venv`   | `[C:/work/proj/.venv/Scripts/python.exe, -m, proj.server]`   |
| Node package (pinned)                        | `[npx, -y, --prefer-offline, "@scope/server@1.2.3"]`         |
| Compiled program                             | `[C:/tools/my-server.exe, --stdio]`                          |

Tips:

- `uv run --directory <folder>` runs the server **inside that folder**, so relative paths in `env` are relative to the server folder, not to HAZEM.
- The bridge removes `VIRTUAL_ENV` for servers, so a uv server always uses its own `.venv`.
- Keep the allow-list short. HAZEM already has 38 native tools, and a 9B model chooses worse with very many.
- Several servers at once are fine: one entry per server, each in its own process. One broken server does not affect the others.

---

## 5. Run and test

Test in this order:

| Step | What                       | Command                                                                                 | Needs   |
| ---- | -------------------------- | --------------------------------------------------------------------------------------- | ------- |
| 1    | The server alone, by hand  | `uv run --directory <folder> <command>` (prints nothing and waits; Ctrl+C to stop)      | uv      |
| 2    | The server in a web page   | `npx @modelcontextprotocol/inspector uv run --directory <folder> <command>`             | Node.js |
| 3    | Through HAZEM's MCP client | `uv run mcp-test <name>` and `uv run mcp-test <name> <tool> key=value ...`              | uv      |
| 4    | By voice                   | `uv run serve --mode assistant`, then speak and read the activity log                   | Ollama  |

`mcp-test` goes through the same bridge as the voice agent, so `LLM sees:` is exactly what the LLM receives. It calls tools directly and does **not** ask "yes or no". Arguments are `key=value` pairs (values are parsed as JSON when possible), or one JSON object. You can use the MCP name (`add_numbers`) or the HAZEM name (`calculator__add_numbers`).

Expected output for the calculator:

```text
starting 1 server(s) ...
[ OK ] calculator: calculator  ready in 0.8s, 1 tool(s)
       calculator__add_numbers                  risk=safe     (a, b)
                                                Add two numbers and return their sum.

calling calculator__add_numbers({"a": 2, "b": 3})   risk=safe
[ OK ] 2 ms
LLM sees: {"ok": true, "a": 2, "b": 3, "sum": 5}
```

By voice, the server log shows each call, for example `tool calculator__add_numbers({'a': 25, 'b': 17}) via llm -> ok=True ...`. `via llm` means the MCP tool was used; `via router` means a built-in tool handled the request (see [Section 7](#7-mcp-notes-vs-built-in-notes)).

---

## 6. Example: MCP Notes server

A second example with three tools that save, list, and delete notes in a JSON file. It shows what the calculator does not: text arguments (including Arabic), a write tool, a destructive tool, saved state, and a setting passed through `env`.

| Tool          | Arguments       | Returns                                        | Risk in HAZEM         |
| ------------- | --------------- | ---------------------------------------------- | --------------------- |
| `add_note`    | `text: str`     | `{"saved": {"id", "text", "created"}}`         | confirm               |
| `list_notes`  | none            | `{"count", "notes": [...]}`                    | safe (`readOnlyHint`) |
| `delete_note` | `note_id: int`  | `{"deleted": {...}, "remaining"}`              | confirm               |

This server is **separate from HAZEM's built-in `add_note` and `read_notes`** tools. Read [Section 7](#7-mcp-notes-vs-built-in-notes) before testing by voice.

Run all commands in PowerShell in the HAZEM project folder.

### 6.1 Create the folders

```powershell
mkdir examples\mcp-notes\src\mcp_notes
```

### 6.2 `examples/mcp-notes/pyproject.toml`

```toml
[project]
name = "mcp-notes"
version = "0.1.0"
description = "Example MCP server (stdio): notes saved in a JSON file."
requires-python = ">=3.10"
dependencies = [
    "mcp>=2.3,<3",
]

[project.scripts]
mcp-notes = "mcp_notes.server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mcp_notes"]
```

### 6.3 `examples/mcp-notes/src/mcp_notes/__init__.py`

```python
"""Example MCP Notes server for the HAZEM voice agent."""
```

### 6.4 `examples/mcp-notes/src/mcp_notes/server.py`

```python
"""MCP Notes server: add_note, list_notes, delete_note over stdio.

Separate from HAZEM's built-in notes (data/notes.md). This server keeps its own
notes in a JSON file, set by the NOTES_FILE environment variable.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

# stdout carries the MCP messages: log to stderr only.
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="mcp-notes: %(message)s")
log = logging.getLogger("mcp-notes")

NOTES_FILE = Path(os.environ.get("NOTES_FILE", "notes.json"))
MAX_LEN = 500

mcp = MCPServer(
    "notes",
    instructions="Keeps the user's MCP notes. add_note saves, list_notes reads, delete_note removes.",
)


class Note(TypedDict):
    id: int
    text: str
    created: str


class AddResult(TypedDict):
    saved: Note


class ListResult(TypedDict):
    count: int
    notes: list[Note]


class DeleteResult(TypedDict):
    deleted: Note
    remaining: int


def _load() -> list[Note]:
    if not NOTES_FILE.exists():
        return []
    try:
        return json.loads(NOTES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise ToolError(f"cannot read the notes file: {e}")


def _save(notes: list[Note]) -> None:
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)       # recreates a deleted folder
    NOTES_FILE.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")


@mcp.tool(annotations=ToolAnnotations(title="Add MCP note", readOnlyHint=False,
                                      destructiveHint=False, openWorldHint=False))
def add_note(text: str) -> AddResult:
    """Save a note in the MCP notes list. Use only when the user asks for MCP notes."""
    text = text.strip()
    if not text:
        raise ToolError("the note is empty")
    if len(text) > MAX_LEN:
        raise ToolError(f"the note is too long (max {MAX_LEN} characters)")
    notes = _load()
    note: Note = {
        "id": max((n["id"] for n in notes), default=0) + 1,
        "text": text,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    notes.append(note)
    _save(notes)
    log.info("add_note id=%s", note["id"])
    return {"saved": note}


@mcp.tool(annotations=ToolAnnotations(title="List MCP notes", readOnlyHint=True,
                                      idempotentHint=True, openWorldHint=False))
def list_notes() -> ListResult:
    """List the notes in the MCP notes list, oldest first, with their ids."""
    notes = _load()
    return {"count": len(notes), "notes": notes}


@mcp.tool(annotations=ToolAnnotations(title="Delete MCP note", readOnlyHint=False,
                                      destructiveHint=True, openWorldHint=False))
def delete_note(note_id: int) -> DeleteResult:
    """Delete one note from the MCP notes list by its id. Call list_notes first to get the id."""
    notes = _load()
    for i, n in enumerate(notes):
        if n["id"] == note_id:
            removed = notes.pop(i)
            _save(notes)
            log.info("delete_note id=%s", note_id)
            return {"deleted": removed, "remaining": len(notes)}
    raise ToolError(f"there is no note with id {note_id}")


def main() -> None:
    mcp.run()                       # stdio transport (default)


if __name__ == "__main__":
    main()
```

The docstrings say "MCP notes" on purpose: the LLM sees both this server's tools and HAZEM's built-in `add_note` / `read_notes`, and chooses by description.

### 6.5 Check the folder and install

```powershell
tree /F examples\mcp-notes
```

The layout must be exactly this, with no second `mcp-notes` folder inside (a common mistake when unzipping a copy):

```
examples\mcp-notes
│   pyproject.toml
└───src
    └───mcp_notes
            __init__.py
            server.py
```

Then install the server (needs internet once) and start it by hand:

```powershell
uv sync --directory examples/mcp-notes
uv run --directory examples/mcp-notes mcp-notes
```

It prints nothing and waits for MCP messages. That means it works. Press Ctrl+C.

### 6.6 Connect it in `config/config.yaml`

Add this under `tools.mcp_servers`, at the same indentation as `calculator:`:

```yaml
    notes:
      enabled: true
      command: [uv, run, --directory, examples/mcp-notes, mcp-notes]
      env: {NOTES_FILE: "data/notes.json"}   # relative to examples/mcp-notes
      tools: [add_note, list_notes, delete_note]
      confirm: [add_note, delete_note]
      timeout_s: 10
      start_timeout_s: 120
      hint: "Only when the user says MCP notes, use notes__add_note, notes__list_notes and notes__delete_note."
```

The notes are saved in `examples\mcp-notes\data\notes.json`, because `uv run --directory` runs the server inside its folder. HAZEM's own notes stay in `data\notes.md`.

### 6.7 Test it

```powershell
uv run mcp-test notes
uv run mcp-test notes list_notes
uv run mcp-test notes add_note text="buy milk"
uv run mcp-test notes add_note text=" "
uv run mcp-test notes delete_note note_id=1
uv run mcp-test notes delete_note note_id=99
```

| Command                     | Expected                                                                       |
| --------------------------- | ------------------------------------------------------------------------------ |
| `mcp-test notes`            | `[ OK ] notes`, 3 tools: `add_note risk=confirm`, `list_notes risk=safe`, `delete_note risk=confirm` |
| `list_notes` (first time)   | `{"ok": true, "count": 0, "notes": []}`                                        |
| `add_note text="buy milk"`  | `{"ok": true, "saved": {"id": 1, "text": "buy milk", ...}}`                    |
| `add_note text=" "`         | `{"ok": false, "error": "Error executing tool add_note: the note is empty"}`   |
| `delete_note note_id=1`     | `{"ok": true, "deleted": {...}, "remaining": 0}`                               |
| `delete_note note_id=99`    | `{"ok": false, "error": "... there is no note with id 99"}`                    |

Arabic text works the same way (`text="اشتري حليب"`). If PowerShell shows it garbled, run `chcp 65001` first; the JSON file is always saved as UTF-8.

Then restart `uv run serve --mode assistant`, look for `tools: MCP notes ready ...`, and test by voice or with the typed command box using the phrases in [Section 7](#7-mcp-notes-vs-built-in-notes).

---

## 7. MCP Notes vs. built-in notes

HAZEM already has **built-in** notes tools in `src/voice_agent/tools/notes.py`. The MCP Notes server is a different tool set with its own file:

|                 | Built-in notes                        | MCP Notes server                                        |
| --------------- | ------------------------------------- | ------------------------------------------------------- |
| Tools (LLM names) | `add_note`, `read_notes`            | `notes__add_note`, `notes__list_notes`, `notes__delete_note` |
| Code            | `src/voice_agent/tools/notes.py`      | `examples/mcp-notes/src/mcp_notes/server.py`            |
| Storage         | `data/notes.md`                       | `examples/mcp-notes/data/notes.json`                    |
| Asks first      | No                                    | `add_note` and `delete_note`                            |
| Delete          | Not available                         | `delete_note`                                           |
| Fast router     | Yes                                   | No (LLM only)                                           |

### Tool names do not collide

The bridge prefixes every MCP tool with its server name, so `add_note` from the server becomes `notes__add_note`. Both tools exist side by side. If an MCP tool ever had exactly the same final name as a native tool, the native tool would win and the MCP tool would be skipped.

### Routing does

`tools/router.py` runs **before** the LLM and sends common note phrases straight to the built-in tools. The router does not know MCP tools, so those requests never reach the MCP server. Typed commands from the activity panel take the same path as speech.

These results come from running HAZEM's router on each phrase:

| Request                                        | Goes to                        |
| ---------------------------------------------- | ------------------------------ |
| "Save a note: meeting at 10"                   | Router → built-in `add_note`   |
| "Save a note in MCP notes: meeting at 10"      | Router → built-in `add_note` (starts with "save a note") |
| "What are my notes"                            | Router → built-in `read_notes` |
| "احفظ ملاحظة اجتماع الساعة 10"                 | Router → built-in `add_note`   |
| "وش ملاحظاتي"                                  | Router → built-in `read_notes` |
| "Use the MCP notes to save meeting at 10"      | LLM → can use `notes__add_note` |
| "Add meeting at 10 to my MCP notes"            | LLM → can use `notes__add_note` |
| "What are my MCP notes" / "List my MCP notes"  | LLM → can use `notes__list_notes` |
| "Delete note 1 from MCP notes"                 | LLM → can use `notes__delete_note` |
| "احفظ في ملاحظات MCP اجتماع الساعة 10"         | LLM → can use `notes__add_note` |
| "اعرض ملاحظات MCP"                             | LLM → can use `notes__list_notes` |

**How to reach the MCP server reliably:**

- Do not start the request with "save a note" / "احفظ ملاحظة". Name the server ("MCP notes") in the request.
- Keep the docstrings and the `hint:` explicit about "MCP notes", so the LLM does not pick the built-in tool.
- Check the server log: `tool notes__add_note(...) via llm` means the MCP server ran; `tool add_note(...) via router` means the built-in tool ran.
- To send every request to the LLM while testing, set `tools.router: false`. This also turns off all other fast commands.

---

## 8. Troubleshooting

Start with `uv run mcp-test <name>`: it shows the same error the server log shows.

| You see                                                   | Cause                                                       | Fix                                                                      |
| --------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------ |
| `Failed to spawn: <command>` / program not found          | The start command was never installed                       | Check `[project.scripts]` and `[build-system]` in `pyproject.toml`, check the folder layout (no nested copy), then delete the server's `.venv` and run `uv sync --directory <folder>` |
| `'uv' not found on PATH` (or any command)                 | The command is not installed or not on PATH                 | Install it, or use a full path in `command`                              |
| `MCP server stopped before answering initialize ... Server said: ...` | The server crashed at startup                    | Read the "Server said" part, or run the command by hand                  |
| `ModuleNotFoundError: No module named 'mcp'`              | The server's packages are not installed                     | `uv sync --directory <folder>` (needs internet once)                     |
| `No module named '<package>'`                             | The package folder name does not match                      | Use an underscore: `src/mcp_notes`, not `src/mcp-notes`                  |
| `did not answer initialize within N s`                    | Slow first install, or the server waits for something       | Raise `start_timeout_s`, or run the command by hand once                 |
| `did not answer tools/call within N s`                    | The tool is slow                                            | Raise `timeout_s`, or make the tool faster (voice needs about 2 s)       |
| `bad config.yaml tools.mcp_servers: unknown setting(s) ...` | A typo in the config                                      | Fix the key name (the message lists the allowed keys)                    |
| `has no tool(s) ['x'] (check tools: in config)`           | The allow-list names a tool the server does not have        | Run `mcp-test <name>` to see the real names                              |
| The server is ready but the LLM never uses the tool       | The router handled it, or the description is unclear        | Check `via router` in the log ([Section 7](#7-mcp-notes-vs-built-in-notes)); improve the docstring and `hint:`; keep the allow-list short |
| HAZEM asks "yes or no" for a read-only tool               | The server did not set `readOnlyHint`                       | Add the tool to `safe:` or fix the server                                |
| Works in HAZEM, but the Inspector shows parse errors      | The server prints to stdout                                 | Log to stderr only                                                       |
| A fixed server is still missing after it failed at startup | Failed servers are not retried until restart               | Restart `serve`                                                          |

---

## 9. Config only, or code change?

| You want to                                                       | Config only         | Code change in HAZEM                    |
| ----------------------------------------------------------------- | ------------------- | --------------------------------------- |
| Add, remove, enable, or disable a server                          | ✓                   |                                         |
| Choose exposed tools, "yes or no" rules, timeouts                 | ✓                   |                                         |
| Pass settings to a server (`env`, `cwd`)                          | ✓                   |                                         |
| Tell the LLM when to use a server (`hint:`)                       | ✓                   |                                         |
| A server adds or changes tools                                    | ✓ (restart `serve`) |                                         |
| Run an MCP tool from the fast router, without the LLM             |                     | ✓ `tools/router.py`                     |
| A custom spoken sentence or "yes or no" question for an MCP tool  |                     | ✓ `tools/speech.py`                     |
| Connect a server over the network (HTTP)                          |                     | ✓ `tools/mcp_client.py`                 |
| Show images returned by a tool in the browser                     |                     | ✓ `to_tool_result` in `tools/mcp_bridge.py` |

---

## 10. Current limits

- **stdio only.** Network servers (Streamable HTTP) are not supported yet.
- Tools are read when a server starts. If a server changes its tools, restart `serve`.
- A server that fails at startup stays off until `serve` restarts. A server that crashes later is restarted on the next call.
- Only text and JSON results are used. Images or files from a tool are noted (`other_content`) but not shown.
- Bridged tools are used through the LLM. The fast router does not know them.
- MCP servers start only with a real desktop (not with `tools.desktop: dry-run`), the same rule as the file tools.
