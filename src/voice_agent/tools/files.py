"""File tools: list, find, read, write, edit, move and delete files in ONE folder.

The work is done by the official Filesystem MCP server (@modelcontextprotocol/server-filesystem),
started with access to that folder only (tools.files.root, default Documents/HAZEM).
These tools are a small, voice-friendly layer on top of it:
  - paths are relative to the folder ("notes/todo.txt"); anything outside is refused
  - writing, editing, moving and deleting ask the user "yes or no" first
  - delete moves the file to the folder's ".deleted" subfolder, so it can be restored
Nothing is sent over the network: the server is a local child process.
"""

from __future__ import annotations

import time
from functools import partial

from voice_agent.tools.base import CONFIRM, Tool, ToolResult
from voice_agent.tools.file_access import Files
from voice_agent.tools.mcp_client import McpError

MAX_READ_CHARS = 3000        # longer files are cut, so the reply stays short and fast
MAX_LIST = 40
TRASH = ".deleted"


def _run(fn):
    """Turn path and MCP errors into a ToolResult the LLM can explain."""
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except ValueError as e:
            return ToolResult(False, error=str(e))
        except McpError as e:
            return ToolResult(False, error=f"file server: {e}")
    return wrapper


@_run
def list_files(fs: Files, folder: str = "", pattern: str = "") -> ToolResult:
    base = fs.path(folder)
    if pattern:
        glob = pattern if any(c in pattern for c in "*?[") else f"**/*{pattern}*"
        ok, text = fs.call("search_files", path=str(base), pattern=glob)
        found = [fs.rel(line) for line in text.splitlines() if line.strip() and not line.startswith("No matches")]
    else:
        ok, text = fs.call("list_directory", path=str(base))
        # lines look like "[FILE] a.txt" / "[DIR] notes"
        found = [ln.replace("[FILE] ", "").replace("[DIR] ", "") + ("/" if ln.startswith("[DIR]") else "")
                 for ln in text.splitlines() if ln.startswith(("[FILE]", "[DIR]"))]
        found = [f for f in found if f.rstrip("/") != TRASH]
    if not ok:
        return ToolResult(False, error=text)
    return ToolResult(True, {"folder": fs.rel(base), "files": found[:MAX_LIST], "total": len(found)})


@_run
def read_file(fs: Files, path: str) -> ToolResult:
    p = fs.path(path)
    ok, text = fs.call("read_text_file", path=str(p))
    if not ok:
        return ToolResult(False, error=text)
    cut = len(text) > MAX_READ_CHARS
    return ToolResult(True, {"file": fs.rel(p), "text": text[:MAX_READ_CHARS], "cut": cut, "chars": len(text)})


@_run
def write_file(fs: Files, path: str, text: str) -> ToolResult:
    p = fs.path(path)
    fs.call("create_directory", path=str(p.parent))
    ok, msg = fs.call("write_file", path=str(p), content=text)
    return ToolResult(True, {"written": fs.rel(p), "chars": len(text)}) if ok else ToolResult(False, error=msg)


@_run
def edit_file(fs: Files, path: str, find: str, replace: str) -> ToolResult:
    p = fs.path(path)
    ok, msg = fs.call("edit_file", path=str(p), edits=[{"oldText": find, "newText": replace}])
    return ToolResult(True, {"edited": fs.rel(p)}) if ok else ToolResult(False, error=msg)


@_run
def move_file(fs: Files, source: str, destination: str) -> ToolResult:
    src, dst = fs.path(source), fs.path(destination)
    fs.call("create_directory", path=str(dst.parent))
    ok, msg = fs.call("move_file", source=str(src), destination=str(dst))
    return ToolResult(True, {"moved": fs.rel(src), "to": fs.rel(dst)}) if ok else ToolResult(False, error=msg)


@_run
def delete_file(fs: Files, path: str) -> ToolResult:
    p = fs.path(path)
    if p == fs.root:
        return ToolResult(False, error="cannot delete the whole folder")
    trash = fs.root / TRASH
    fs.call("create_directory", path=str(trash))
    dst = trash / f"{time.strftime('%Y%m%d-%H%M%S')}-{p.name}"
    ok, msg = fs.call("move_file", source=str(p), destination=str(dst))
    if not ok:
        return ToolResult(False, error=msg)
    return ToolResult(True, {"deleted": fs.rel(p), "restore_from": fs.rel(dst)})


def tools(tb) -> list[Tool]:
    fs: Files | None = getattr(tb, "files", None)
    if fs is None:
        return []
    s = {"type": "string"}
    where = f"Paths are relative to the user's {fs.root.name} folder."
    return [
        Tool("list_files", f"List files in the user's file folder, or find them by name (pattern). {where}",
             {"folder": s, "pattern": s}, partial(list_files, fs), timeout_s=60),
        Tool("read_file", f"Read a text file. {where}", {"path": s}, partial(read_file, fs), ["path"],
             timeout_s=60),
        Tool("write_file", f"Create or overwrite a text file. {where}", {"path": s, "text": s},
             partial(write_file, fs), ["path", "text"], risk=CONFIRM, timeout_s=60),
        Tool("edit_file", f"Replace exact text in a file. {where}", {"path": s, "find": s, "replace": s},
             partial(edit_file, fs), ["path", "find", "replace"], risk=CONFIRM, timeout_s=60),
        Tool("move_file", f"Move or rename a file. {where}", {"source": s, "destination": s},
             partial(move_file, fs), ["source", "destination"], risk=CONFIRM, timeout_s=60),
        Tool("delete_file", f"Delete a file (it can be restored from .deleted). {where}", {"path": s},
             partial(delete_file, fs), ["path"], risk=CONFIRM, timeout_s=60),
    ]
