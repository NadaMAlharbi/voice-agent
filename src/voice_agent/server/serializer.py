"""Wire protocol between browser and server (one WebSocket).

client -> server
  binary : PCM16 little-endian mono @ 16 kHz, 20 ms frames (640 bytes)
  text   : JSON control, e.g. {"type": "ping", "t": 123.4}
           {"type": "set_voice", "gender"}  {"type": "set_language", "language": "auto|ar|en"}
           {"type": "stop"}  (Stop button)  {"type": "text", "text"}  {"type": "confirm", "answer"}

server -> client
  binary : PCM16 little-endian mono @ 24 kHz (assistant audio / loopback)
  text   : JSON events:
           {"type": "interrupt"}                        flush playback NOW (barge-in)
           {"type": "transcript", "text", "language", "stt_ms"}
           {"type": "user_started_speaking"} / {"type": "user_stopped_speaking"}
           {"type": "user_turn", "text"}                end of user turn (Smart Turn)
           {"type": "pong", "t"}
"""

from __future__ import annotations

import json

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

IN_SR = 16000


class RawPCMSerializer(FrameSerializer):
    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, InterruptionFrame):
            return json.dumps({"type": "interrupt"})
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio  # transport already resampled to its output rate
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message, ensure_ascii=False)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, (bytes, bytearray)):
            if not data:
                return None
            return InputAudioRawFrame(audio=bytes(data), sample_rate=IN_SR, num_channels=1)
        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            return None
        return InputTransportMessageFrame(message=msg) if isinstance(msg, dict) else None
