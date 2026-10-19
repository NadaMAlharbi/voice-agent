"""Small Pipecat processors used by the pipelines."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    DataFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    LLMContextFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from voice_agent.stt.base import STTEngine


@dataclass
class SpeechTextFrame(DataFrame):
    """What STT heard in one speech segment, with its language. The response code uses it
    to start the LLM early and to know the spoken language."""
    text: str = ""
    language: str = "en"


@dataclass
class StopReplyFrame(DataFrame):
    """The browser's Stop button: end the current reply (LLM and speech) now."""


@dataclass
class UserTextFrame(DataFrame):
    """A typed command or a confirm button press from the browser (same path as a spoken turn)."""
    text: str = ""
    source: str = "typed"          # typed | button


def message(payload: dict) -> OutputTransportMessageUrgentFrame:
    """JSON event for the browser (urgent = not queued behind audio)."""
    return OutputTransportMessageUrgentFrame(message=payload)


def voice_info(voices: dict) -> dict:
    # This function lists the voices for the browser (model voice behind each name).
    if not voices:
        return {}
    return {"voices": [{"name": v.name, "language": v.language, "gender": v.gender,
                        "engine": v.engine, "voice": v.voice_id} for v in voices.values()]}


class ClientMessageHandler(FrameProcessor):
    """Browser control messages: ping (RTT), set_voice (male/female), typed commands, confirm buttons."""

    def __init__(self, session=None, voices=None, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._voices = voices or {}

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputTransportMessageFrame):
            msg = frame.message
            if msg.get("type") == "ping":
                await self.push_frame(message({"type": "pong", "t": msg.get("t")}))
            elif msg.get("type") == "client_metric":
                logger.info(f"turn {msg.get('turn_id')}: first sound in browser {msg.get('first_sound_ms')} ms "
                            f"after the user's turn ended")
            elif msg.get("type") == "text" and str(msg.get("text", "")).strip():
                await self.push_frame(UserTextFrame(text=str(msg["text"]).strip()[:500]))
            elif msg.get("type") == "stop":
                await self.push_frame(StopReplyFrame())
            elif msg.get("type") == "set_language" and self._session is not None:
                if msg.get("language") in ("auto", "ar", "en"):
                    self._session.stt_language = msg["language"]
                await self.push_frame(message({"type": "language",
                                               "language": self._session.stt_language or "auto"}))
            elif msg.get("type") == "confirm":
                await self.push_frame(UserTextFrame(text="yes" if msg.get("answer") else "no", source="button"))
            elif msg.get("type") == "set_voice" and self._session is not None:
                if msg.get("gender") in ("female", "male"):
                    self._session.gender = msg["gender"]
                await self.push_frame(message({"type": "voice", "gender": self._session.gender,
                                               **voice_info(self._voices)}))
            return  # consumed
        await self.push_frame(frame, direction)


class EchoProcessor(FrameProcessor):
    """Sends the mic audio straight back to the speaker (loopback mode)."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            await self.push_frame(
                OutputAudioRawFrame(frame.audio, frame.sample_rate, frame.num_channels)
            )
            return
        await self.push_frame(frame, direction)


class LocalSTTService(SegmentedSTTService):
    """Wraps our STT engine as a Pipecat segmented STT service.

    Pipecat collects the speech between VAD start and stop and calls run_stt once per
    segment. The engine runs in a thread so audio keeps flowing, and the detected
    language is reported.

    Each result goes to the browser ("transcript" or "stt_skip" with the reason).
    With record_dir set, each segment is also saved as wav + json, including whether
    the assistant was speaking (text heard while it speaks is usually echo).
    """

    def __init__(self, engine: STTEngine, language: str = "auto", record_dir=None, session=None, **kwargs):
        super().__init__(
            settings=STTSettings(model=engine.name, language=None),
            ttfs_p99_latency=0.6,
            **kwargs,
        )
        self._engine = engine
        self._language = language
        self._session = session          # the browser may choose the language (Session.stt_language)
        self._record_dir = Path(record_dir) if record_dir else None
        self._bot_speaking = False
        self._bot_spoke_since_segment = False
        self._n = 0

    @property
    def wants_wav_segments(self) -> bool:
        return False  # raw PCM16

    def can_generate_metrics(self) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = self._bot_spoke_since_segment = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        await super().process_frame(frame, direction)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        during_bot = self._bot_speaking or self._bot_spoke_since_segment
        self._bot_spoke_since_segment = self._bot_speaking
        await self.start_processing_metrics()
        try:
            lang = getattr(self._session, "stt_language", None) or self._language
            t = await asyncio.to_thread(self._engine.transcribe, pcm, lang)
        except Exception as e:  # noqa: BLE001
            yield ErrorFrame(f"STT failed: {e}")
            return
        await self.stop_processing_metrics()
        self._n += 1
        diag = {k: v for k, v in t.info.items() if k != "rejected"}
        if self._record_dir:
            self._record(pcm, t, during_bot)
        if t.text:
            yield message({"type": "transcript", "text": t.text, "language": t.language, "stt_ms": round(t.ms),
                           "during_bot_speech": during_bot, "level_lufs": loudness_lufs(pcm), **diag})
            yield SpeechTextFrame(text=t.text, language=t.language)
            yield TranscriptionFrame(t.text, self._user_id, time_now_iso8601(), _lang(t.language))
        else:
            yield message({"type": "stt_skip", "reason": t.info.get("skip", "empty"), "audio_s": round(t.audio_s, 2),
                           "during_bot_speech": during_bot, "rejected": t.info.get("rejected", []), **diag})

    def _record(self, pcm: np.ndarray, t, during_bot: bool) -> None:
        import json

        from voice_agent.audio_io import save_wav

        stem = self._record_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{self._n:03d}"
        save_wav(stem.with_suffix(".wav"), pcm, 16000)
        stem.with_suffix(".json").write_text(json.dumps({
            "text": t.text, "language": t.language, "stt_ms": round(t.ms), "audio_s": round(t.audio_s, 2),
            "during_bot_speech": during_bot, "rms": round(float(np.sqrt(np.mean(pcm ** 2))), 4), **t.info,
        }, ensure_ascii=False, indent=1), encoding="utf-8")


def loudness_lufs(pcm: np.ndarray) -> float | None:
    """Loudness of a speech segment (LUFS, the scale of vad.min_volume: volume = (LUFS + 110) / 100).
    Shown in the browser log so vad.min_volume can be matched to your microphone."""
    try:
        import loudness

        v = loudness.integrated_loudness(pcm.astype(np.float32), 16000)
        return round(float(v), 1) if np.isfinite(v) else None
    except Exception:  # noqa: BLE001  (too short, or library missing)
        return None


def _lang(code: str) -> Language | None:
    try:
        return Language(code)
    except ValueError:
        return None


class TurnReporter(FrameProcessor):
    """Tells the browser about user turns.

    When Smart Turn decides the user has finished, the turn text is sent to the browser.
    `swallow_context=True` (transcribe mode, no LLM) stops the frame here.
    """

    def __init__(self, swallow_context: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._swallow = swallow_context

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            await self.push_frame(message({"type": "user_started_speaking"}))
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self.push_frame(message({"type": "user_stopped_speaking"}))
        elif isinstance(frame, LLMContextFrame):
            msgs = frame.context.get_messages()
            last = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
            text = _content_text(last.get("content")) if last else ""
            await self.push_frame(message({"type": "user_turn", "text": text}))
            if self._swallow:
                return
        await self.push_frame(frame, direction)


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""
