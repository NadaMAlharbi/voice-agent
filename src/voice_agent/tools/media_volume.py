"""System volume and media playback keys."""

from __future__ import annotations

from functools import partial

from voice_agent.tools.base import Tool, ToolResult


def volume(tb, action: str, steps: int = 5) -> ToolResult:
    # This function turns the volume up, down, or mutes it.
    tb.desktop.volume(action, max(1, min(25, steps)))
    return ToolResult(True, {"volume": action})


def media(tb, action: str) -> ToolResult:
    # This function sends play/pause, next, previous or stop.
    tb.desktop.media(action)
    return ToolResult(True, {"media": action})


def tools(tb) -> list[Tool]:
    S = {"type": "string"}
    return [
        Tool("volume", "Change the system volume.", {"action": {**S, "enum": ["up", "down", "mute"]},
                                                     "steps": {"type": "integer"}}, partial(volume, tb), ["action"]),
        Tool("media", "Control media playback.",
             {"action": {**S, "enum": ["play_pause", "next", "previous", "stop"]}}, partial(media, tb), ["action"]),
    ]
