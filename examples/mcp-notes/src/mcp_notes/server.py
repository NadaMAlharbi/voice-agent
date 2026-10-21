
"""Notes MCP server: add_note and list_notes, saved in a JSON file. stdio transport."""

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

# Logs go to stderr only because stdout is reserved for MCP messages.
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="mcp-notes: %(message)s")
log = logging.getLogger("mcp-notes")

# The file path is provided through the environment by config.yaml.
NOTES_FILE = Path(os.environ.get("NOTES_FILE", "notes.json"))
MAX_LEN = 500

mcp = MCPServer(
    "notes",
    instructions="Saves and lists short notes. Use add_note to save, list_notes to read.",
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


# class DeleteResult(TypedDict):
#     deleted: Note
#     remaining: int

def _load() -> list[Note]:
    if not NOTES_FILE.exists():
        return []
    try:
        return json.loads(NOTES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise ToolError(f"cannot read the notes file: {e}")


def _save(notes: list[Note]) -> None:
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False preserves Arabic text and other Unicode characters as-is.
    NOTES_FILE.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")


# Write tool: readOnlyHint=False.
@mcp.tool(annotations=ToolAnnotations(title="Add note", readOnlyHint=False,
                                      destructiveHint=False, openWorldHint=False))
def add_note(text: str) -> AddResult:
    """Save a short note. Use when the user asks to remember, note or write something down."""
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


# Read-only tool: readOnlyHint=True.
@mcp.tool(annotations=ToolAnnotations(title="List notes", readOnlyHint=True,
                                      idempotentHint=True, openWorldHint=False))
def list_notes() -> ListResult:
    """List all saved notes, oldest first."""
    notes = _load()
    return {"count": len(notes), "notes": notes}


# @mcp.tool(annotations=ToolAnnotations(title="Delete note", readOnlyHint=False,
#                                       destructiveHint=True, openWorldHint=False))
# def delete_note(note_id: int) -> DeleteResult:
#     """Delete one note by its id. Call list_notes first to find the id."""
#     notes = _load()
#     for i, n in enumerate(notes):
#         if n["id"] == note_id:
#             removed = notes.pop(i)
#             _save(notes)
#             log.info("delete_note id=%s", note_id)
#             return {"deleted": removed, "remaining": len(notes)}
#     raise ToolError(f"there is no note with id {note_id}")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()