"""The assistant's reply: LLM text -> speech chunks -> TTS -> audio, with barge-in.

Each finished user turn starts a new response with its own turn_id:

    Ollama stream -> SpeechChunker -> queue -> TTS worker -> audio frames

The LLM and the TTS run at the same time, so speech starts while the LLM is still writing.

When the user interrupts (barge-in):
  1. The turn_id is cleared, so nothing from the old turn is sent any more.
  2. TTS is cancelled and the Ollama stream is closed (Ollama stops generating).
  3. The browser gets "interrupt" and "cancel_turn" and drops that turn's audio.
  4. History keeps only what the user actually heard (+ "..." if it was cut off).

Tools: with an Agent attached, each utterance goes to the agent first (sleep/wake,
yes/no confirmation, fast router). A router match runs the tool and speaks a short
fixed sentence, without the LLM. Otherwise the LLM gets the tool list and tools run
between LLM steps. Risky tools always wait for a yes/no. Tool calls stay in the
history even if speech was cut off, because the actions really happened.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voice_agent.audio_io import float_to_pcm16
from voice_agent.chunker import SpeechChunker, speakable
from voice_agent.config import Chunker as ChunkerCfg
from voice_agent.knowledge import Intent, build_messages, classify, fetch_live, turn_note
from voice_agent.language import LanguageGuard, directive, reminder, text_language
from voice_agent.server.agent import is_command, needs_tool_retry, retry_note
from voice_agent.server.processors import SpeechTextFrame, StopReplyFrame, UserTextFrame
from voice_agent.server.tts_budget import TTSCostModel, plan_chunk
from voice_agent.tools import speech
from voice_agent.tools.base import ToolResult
from voice_agent.tools.router import is_sleep, is_wake, yes_no
from voice_agent.tts.base import Voice, cancel, pick_voice, reset_cancel
from voice_agent.tts.base import synthesize as tts_synthesize

FLUSH = object()   # marker: speak the buffered text now (e.g. before a slow tool runs)
OPTIMISTIC = {"open_app", "web_search", "youtube_search", "open_url", "type_text", "focus_window", "open_calendar"}  # speak while the tool runs


@dataclass
class Session:
    """Per-connection settings the browser can change."""
    gender: str = "female"
    language: str = "en"        # used before the user has spoken (e.g. timer announcements)
    stt_language: str | None = None   # browser choice: auto | ar | en (None = config stt.language)


class Prefetch:
    """The LLM's first step, started before the user's turn is confirmed.

    Smart Turn may wait for more speech (up to turn.stop_secs). The transcript is
    already known then, so the LLM starts on it. Its output is only kept in memory.
    When the turn ends with the same text, the reply uses it and starts sooner.
    Otherwise it is cancelled (this also stops Ollama). Tools never run here.
    """

    def __init__(self, text: str, history: list[dict], language: str, stream):
        self.text, self.history, self.language = text, history, language
        self.items: list = []
        self.done = False
        self.error: Exception | None = None
        self.used = False
        self._new = asyncio.Event()
        self.task = asyncio.get_running_loop().create_task(self._run(stream))

    async def _run(self, stream) -> None:
        try:
            async for ev in stream:
                self.items.append(ev)
                self._new.set()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001  (Ollama down: the reply reports it)
            self.error = e
        finally:
            self.done = True
            self._new.set()

    def matches(self, text: str, history: list[dict]) -> bool:
        return _norm(text) == _norm(self.text) and history == self.history

    async def events(self):
        """Everything the LLM sent so far, then the rest as it arrives."""
        self.used = True
        i = 0
        while True:
            while i < len(self.items):
                yield self.items[i]
                i += 1
            if self.done:
                if self.error is not None:
                    raise self.error
                return
            self._new.clear()
            await self._new.wait()

    def cancel(self) -> None:
        if not self.task.done():
            self.task.cancel()


def _norm(text: str) -> str:
    return " ".join(text.split())


class PlaybackTracker:
    """Estimates what the user actually heard.

    Audio plays in real time, so chunk i starts at max(push time, end of chunk i-1).
    On interruption we keep the fully played chunks plus the played share (by words)
    of the current one.
    """

    def __init__(self):
        self.chunks: list[tuple[str, float, float]] = []  # text, start, end
        self._end = 0.0
        self.gap_s = 0.0

    def add(self, text: str, duration_s: float, pushed_at: float, new_chunk: bool = True) -> float:
        """new_chunk=False: another audio piece of the same chunk (streaming TTS).

        Returns the estimated start time of this piece. A piece that arrives after the
        previous audio ended means the listener heard a gap (TTS was too slow).
        """
        if self.chunks and pushed_at > self._end:
            self.gap_s += pushed_at - self._end
        start = max(pushed_at, self._end)
        self._end = start + duration_s
        if new_chunk or not self.chunks:
            self.chunks.append((text, start, self._end))
        else:
            t, s0, _ = self.chunks[-1]
            self.chunks[-1] = (t, s0, self._end)
        return start

    @property
    def ends_at(self) -> float:
        return self._end

    def spoken(self, now: float) -> tuple[str, bool]:
        """(text heard so far, whether everything pushed has finished playing)."""
        parts: list[str] = []
        for text, start, end in self.chunks:
            if end <= now:
                parts.append(text)
                continue
            if start < now and text:
                words = text.split()
                n = int(len(words) * (now - start) / (end - start))
                parts.append(" ".join(words[:n]))
            return _join(parts), False
        return _join(parts), True


def _join(parts: list[str]) -> str:
    return " ".join(p for p in parts if p).strip()


@dataclass
class _Response:
    turn_id: int
    context: object                 # pipecat LLMContext (shared with the user aggregator)
    insert_at: int                  # where the assistant message goes in the history
    started: float
    user_stopped: float | None
    tracker: PlaybackTracker = field(default_factory=PlaybackTracker)
    text: str = ""                  # everything the LLM wrote
    done: bool = False              # LLM + TTS finished
    ttft: float | None = None
    first_chunk: float | None = None
    first_audio: float | None = None
    chunks: int = 0
    language: str = "en"            # reply language = language of the user's current utterance
    user_text: str = ""             # the current utterance (last user message)
    intent: Intent = field(default_factory=lambda: Intent("general"))
    history: list = field(default_factory=list)
    tool_msgs: list = field(default_factory=list)    # assistant tool_calls + tool results (kept in history)
    actions: list = field(default_factory=list)      # [{"tool", "ok", "ms", "source"}] for metrics/UI
    announce: str | None = None                      # proactive speech (timer), no user message
    live_data: bool | None = None
    tts: list = field(default_factory=list)          # per chunk: words, queue/synth ms, engine stats
    loop_lag_ms: float = 0.0                          # worst event-loop stall during the reply
    speech_end_wall: float | None = None             # when the user really stopped talking (time.time())
    first_audio_wall: float | None = None
    prefetch: Prefetch | None = None                 # LLM started early on the same words
    produced_all: bool = False                       # the text source has ended
    llm_prompt: dict = field(default_factory=dict)   # Ollama prompt stats of the first LLM step


class ResponseProcessor(FrameProcessor):
    def __init__(
        self,
        llm,                                   # voice_agent.llm.OllamaClient (or compatible)
        tts=None,                              # TTSEngine, or None = text-only ("chat" mode)
        voices: dict[str, Voice] | None = None,
        session: Session | None = None,
        chunker_cfg: ChunkerCfg | None = None,
        max_history: int = 20,
        clock: Callable[[], float] = time.monotonic,
        live_provider=None,                    # knowledge.LiveDataProvider (optional, online)
        monitor=None,                          # bargein.PlaybackMonitor (echo reference)
        now: Callable[[], datetime] = datetime.now,
        agent=None,                            # server.agent.Agent (tools, router, confirmations)
        context=None,                          # the LLMContext shared with the user aggregator
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._llm, self._tts, self._voices = llm, tts, voices or {}
        self._session = session or Session()
        self._chunker_cfg = chunker_cfg or ChunkerCfg()
        self._max_history = max_history
        self._clock = clock
        self._live = live_provider
        self._monitor = monitor
        self._cost = TTSCostModel()          # learns this GPU's TTS speed (gap-free scheduling)
        self._now = now
        self._turn_id = 0
        self._active: int | None = None     # turn_id allowed to emit frames
        self._resp: _Response | None = None
        self._task: asyncio.Task | None = None
        self._tts_lock = threading.Lock()   # one synthesis at a time on the GPU
        self._user_stopped_at: float | None = None
        self._agent = agent
        self._context = context
        self._user_speaking = False
        self._timers: dict[str, asyncio.Task] = {}
        self._timer_n = 0
        self._last_lang = getattr(session, "language", "en")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._vad_speaking = False
        self._speech_end_wall: float | None = None    # last VAD stop, minus its silence wait
        self._turn_texts: list[tuple[str, str]] = []  # (text, STT language) of the current user turn
        self._prefetch: Prefetch | None = None
        self._merge_next = False                      # next utterance continues a cut-off one
        self._vad_start_at: float | None = None       # when the user's latest speech began
        if agent is not None and agent.toolbox is not None:
            agent.toolbox.scheduler = self.schedule_timer

    # ------------------------------------------------------------------ frames

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._interrupt("barge-in")
            await self.push_frame(frame, direction)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._interrupt("shutdown", notify=False)
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._interrupt("new turn")  # normally already done by the InterruptionFrame
            self._context = frame.context
            if _has_new_utterance(frame.context):
                self._start(frame.context)     # consumed here
            else:
                logger.warning("context without a new user utterance: not answering (no stale re-answers)")
        elif isinstance(frame, UserTextFrame):
            await self._typed(frame)
        elif isinstance(frame, StopReplyFrame):
            self._drop_prefetch()
            await self._interrupt("stop")
            # Like a barge-in, tell the output transport to drop the audio it still has queued.
            # Without this it keeps sending the stopped reply (Kokoro makes a whole reply in a
            # second or two, so many seconds can be queued), still counts as "bot speaking"
            # (your next words then face the stricter barge-in checks), and the next reply's
            # audio waits behind the old one. The browser also gets "interrupt" and flushes.
            await self.push_frame(InterruptionFrame())
        elif isinstance(frame, SpeechTextFrame):
            if self._user_speaking:            # part of a real user turn (not echo or noise)
                self._turn_texts.append((frame.text, frame.language))
                self._start_prefetch()
        else:
            if isinstance(frame, UserStoppedSpeakingFrame):
                self._user_stopped_at = self._clock()
                self._user_speaking = False
            elif isinstance(frame, UserStartedSpeakingFrame):
                self._user_speaking = True
                self._turn_texts = []
                self._drop_prefetch()
            elif isinstance(frame, VADUserStartedSpeakingFrame):
                self._vad_speaking = True
                self._vad_start_at = self._clock()
                self._drop_prefetch()          # the user goes on: the words will change
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                self._vad_speaking = False
                self._speech_end_wall = frame.timestamp - frame.stop_secs
            await self.push_frame(frame, direction)

    def _current(self, turn_id: int) -> bool:
        return self._active == turn_id

    async def _send(self, turn_id: int, payload: dict, ordered: bool = False) -> None:
        """JSON to the browser. ordered=True queues it with the audio (arrives in sync)."""
        if not self._current(turn_id):
            return
        cls = OutputTransportMessageFrame if ordered else OutputTransportMessageUrgentFrame
        await self.push_frame(cls(message=payload))

    # ------------------------------------------------------------------ turn lifecycle

    def _start(self, context, spoken: bool = True) -> None:
        self._turn_id += 1
        tid = self._active = self._turn_id
        if self._merge_next:
            _merge_last_user_messages(context)
            self._merge_next = False
        history = context.get_messages()
        resp = self._resp = _Response(
            tid, context, insert_at=len(history), started=self._clock(), user_stopped=self._user_stopped_at
        )
        last_user = next((m for m in reversed(history) if m.get("role") == "user"), None)
        resp.user_text = _text(last_user.get("content")) if last_user else ""
        hint = _stt_language(self._turn_texts) if spoken else None
        resp.language = text_language(resp.user_text, default=self._last_lang, hint=hint)
        resp.intent = classify(resp.user_text)
        resp.history = list(history)
        resp.speech_end_wall = self._speech_end_wall if spoken else None
        p, self._prefetch = self._prefetch, None
        if p is not None and spoken and p.language == resp.language and p.matches(resp.user_text, list(history[:-1])):
            resp.prefetch = p
        elif p is not None:
            p.cancel()
        self._turn_texts = []
        self._task = self.create_task(self._respond(resp), f"response-{tid}")

    # ------------------------------------------------------------------ early LLM start (prefetch)

    def _drop_prefetch(self) -> None:
        if self._prefetch is not None:
            self._prefetch.cancel()
            self._prefetch = None

    def _start_prefetch(self) -> None:
        """Start the LLM on the words heard so far, while the end of the turn is not sure yet."""
        self._drop_prefetch()
        events = getattr(self._llm, "stream_events", None)
        text = " ".join(t for t, _ in self._turn_texts).strip()
        if self._vad_speaking or events is None or not text or self._context is None:
            return
        ag = self._agent
        lang = text_language(text, default=self._last_lang, hint=_stt_language(self._turn_texts))
        if ag is not None:
            if ag.asleep or ag.pending is not None or is_sleep(text) or ag.route(text, lang) is not None:
                return                         # no LLM needed, or the answer depends on state
        intent = classify(text)
        if intent.live and self._live is not None:
            return
        history = list(self._context.get_messages())
        if history and history[-1].get("role") == "user" and _norm(_text(history[-1].get("content"))) == _norm(text):
            return                             # the turn has just ended: the normal reply starts now
        msgs = self._llm_messages(history + [{"role": "user", "content": text}], lang, turn_note(intent, None))
        tools = ag.schemas() if ag is not None and ag.max_steps > 1 else None
        self._prefetch = Prefetch(text, history, lang, events(msgs, tools))

    async def _typed(self, frame: UserTextFrame) -> None:
        """Typed command / confirm button: same handling as a spoken utterance."""
        await self._interrupt("new turn")
        if self._context is None:
            from pipecat.processors.aggregators.llm_context import LLMContext
            self._context = LLMContext()
        msgs = list(self._context.get_messages())
        msgs.append({"role": "user", "content": frame.text})
        self._context.set_messages(msgs)
        await self.push_frame(OutputTransportMessageUrgentFrame(
            message={"type": "user_turn", "text": frame.text, "source": frame.source}))
        self._user_stopped_at = self._clock()
        self._merge_next = False
        self._start(self._context, spoken=False)

    # ------------------------------------------------------------------ timers (proactive speech)

    def schedule_timer(self, seconds: float, label: str = "") -> str:
        """Called from a tool thread. Returns the timer id."""
        self._timer_n += 1
        tid = f"t{self._timer_n}"
        lang = self._last_lang
        loop = self._loop
        if loop is None:
            raise RuntimeError("no running pipeline")

        def _go():
            self._timers[tid] = loop.create_task(self._timer(tid, seconds, label, lang))
        loop.call_soon_threadsafe(_go)
        return tid

    async def _timer(self, tid: str, seconds: float, label: str, lang: str) -> None:
        try:
            await asyncio.sleep(seconds)
            waited = 0.0
            while (self._busy() or self._user_speaking) and waited < 30:
                await asyncio.sleep(0.25)          # don't talk over the user or another reply
                waited += 0.25
            await self._interrupt("timer")
            await self.push_frame(OutputTransportMessageUrgentFrame(
                message={"type": "timer_done", "timer_id": tid, "label": label}))
            self._start_announcement(speech.timer_done(label, lang), lang)
        except asyncio.CancelledError:
            pass
        finally:
            self._timers.pop(tid, None)

    def _busy(self) -> bool:
        """A reply is being generated or its audio is still playing."""
        r = self._resp
        return r is not None and self._active == r.turn_id and (not r.done or r.tracker.ends_at > self._clock())

    def _start_announcement(self, text: str, lang: str) -> None:
        self._turn_id += 1
        tid = self._active = self._turn_id
        ctx = self._context
        n = len(ctx.get_messages()) if ctx is not None else 0
        resp = self._resp = _Response(tid, ctx, insert_at=n, started=self._clock(), user_stopped=None)
        resp.language, resp.announce = lang, text
        self._task = self.create_task(self._respond(resp), f"announce-{tid}")

    def _llm_messages(self, history: list[dict], language: str, note: str = "") -> list[dict]:
        """System prompt, device clock, per-turn notes and recent history.

        The language and knowledge notes come from the current utterance only.
        """
        msgs = []
        for m in history[-self._max_history:]:
            role = m.get("role")
            if role in ("user", "assistant") and not m.get("tool_calls"):
                msgs.append({"role": role, "content": _text(m.get("content"))})
            elif role == "assistant" or role == "tool":
                msgs.append(dict(m))                # tool calls / results, as Ollama expects them
        while msgs and msgs[0]["role"] != "user":  # never start with an orphan tool result
            msgs.pop(0)
        base = getattr(self._llm.cfg, "system_prompt", "")
        if self._agent is not None and self._agent.system_note():
            base = f"{base}\n\n{self._agent.system_note()}"
        cache = bool(getattr(self._llm.cfg, "cache_friendly", False)) and self._agent is not None
        return build_messages(base, msgs, None, self._now(), note, directive(language), cache)

    async def _interrupt(self, reason: str, notify: bool = True) -> None:
        resp = self._resp
        if resp is None:
            return
        self._active = None                    # 1. invalidate the turn
        if self._monitor is not None:
            self._monitor.truncate(self._clock())  # flushed audio will never be heard
        if self._tts is not None:
            cancel(self._tts)                  # 2. stop TTS mid-chunk (if supported)
        if self._task and not self._task.done():
            await self.cancel_task(self._task)  # 3. cancels the LLM stream too
        self._task, self._resp = None, None

        if self._tts is None:                  # text-only: the user saw everything written
            spoken, all_played = resp.text.strip(), True
        else:
            spoken, all_played = resp.tracker.spoken(self._clock())
        complete = resp.done and all_played
        # The user went on talking before the reply started: the turn was cut at a pause.
        # The next utterance is joined to it, so the reply sees the whole sentence, and the
        # few words of the reply (not listened to) are left out of the history.
        # (VAD reports speech 0.2 s late, and nobody answers a reply within 0.3 s of hearing it)
        went_on = self._vad_start_at is not None and (resp.first_audio is None
                                                      or self._vad_start_at <= resp.first_audio + 0.5)
        self._merge_next = (reason == "barge-in" and not resp.tool_msgs and resp.announce is None
                            and self._clock() - resp.started < 4.0 and (not spoken or went_on))
        if self._merge_next:
            spoken = ""
        content = spoken if complete else (f"{spoken}..." if spoken else "")
        add = list(resp.tool_msgs) + ([{"role": "assistant", "content": content}] if content else [])
        if add and resp.context is not None:   # 5. history = what was heard (+ actions that really ran)
            msgs = list(resp.context.get_messages())
            msgs[resp.insert_at:resp.insert_at] = add
            resp.context.set_messages(msgs)
        if not complete and notify:            # 4. tell the browser to drop this turn
            await self.push_frame(OutputTransportMessageUrgentFrame(message={
                "type": "cancel_turn", "turn_id": resp.turn_id, "reason": reason, "heard": spoken}))
            logger.info(f"turn {resp.turn_id} cancelled ({reason}); heard: {spoken!r}")

    async def cleanup(self):
        await super().cleanup()
        for task in list(self._timers.values()):
            task.cancel()
        if self._task and not self._task.done():
            await self.cancel_task(self._task)

    # ------------------------------------------------------------------ the response

    async def _respond(self, resp: _Response) -> None:
        tid = resp.turn_id
        self._loop = asyncio.get_running_loop()
        self._last_lang = resp.language
        try:
            source = await self._plan(resp)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception(f"turn {tid}: planning failed")
            source = _fixed(f"Error: {e}")
        if source is None:                     # nothing to say (asleep): no reply at all
            resp.done = True
            return
        await self._send(tid, {"type": "assistant_start", "turn_id": tid, "language": resp.language,
                               "user_text": resp.user_text, "intent": resp.intent.kind,
                               "live_data": resp.live_data, "proactive": resp.announce is not None})
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        producer = asyncio.create_task(self._produce(resp, source, queue))
        lag = asyncio.create_task(self._watch_loop_lag(resp))
        try:
            await self._consume(resp, queue)
            await producer
            resp.done = True
            metrics = self._turn_metrics(resp)
            logger.info(f"turn {tid} done: {metrics}")
            await self._send(tid, {"type": "assistant_done", "turn_id": tid, "text": resp.text, **metrics},
                             ordered=True)  # arrives when the audio has been played
            if resp.announce is not None:
                self._finish_announcement(resp)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001  (Ollama down, TTS error, ...)
            logger.exception(f"turn {tid} failed")
            await self._send(tid, {"type": "error", "turn_id": tid, "message": f"{type(e).__name__}: {e}"})
            resp.done = True
        finally:
            if not producer.done():
                producer.cancel()
            lag.cancel()
            if resp.prefetch is not None and not resp.prefetch.used:
                resp.prefetch.cancel()

    def _finish_announcement(self, resp: _Response) -> None:
        """A completed proactive message stays in history like any reply."""
        if resp.context is not None and resp.text.strip():
            msgs = list(resp.context.get_messages())
            msgs.append({"role": "assistant", "content": resp.text.strip()})
            resp.context.set_messages(msgs)
        self._active = None
        self._resp = None

    async def _plan(self, resp: _Response):
        """What to say: an async source of text (or None = stay silent)."""
        if resp.announce is not None:
            return _fixed(resp.announce)
        ag = self._agent
        text = resp.user_text
        if ag is not None:
            if ag.asleep:
                if not is_wake(text):
                    self._drop_last_user(resp)
                    await self._send(resp.turn_id, {"type": "asleep_ignored", "text": text})
                    return None
                ag.asleep = False
                await self._send(resp.turn_id, {"type": "mode", "asleep": False})
                rest = _after_wake_word(text)
                if not rest:
                    return _fixed(speech.waking(resp.language))
                resp.user_text = text = rest
            if is_sleep(text):
                ag.asleep, ag.pending = True, None
                await self._send(resp.turn_id, {"type": "mode", "asleep": True})
                return _fixed(speech.sleeping(resp.language))
            pending = ag.take_pending()
            if pending is not None:
                answer = yes_no(text)
                if answer is not None:
                    resp.language = pending.language       # reply in the language of the question
                    await self._send(resp.turn_id, {"type": "confirm_resolved", "tool": pending.tool,
                                                    "answer": answer})
                    if not answer:
                        return _fixed(speech.cancelled(resp.language))
                    return self._routed_source(resp, pending.tool, pending.args, "confirmed")
            routed = ag.route(text, resp.language)
            if routed is not None:
                if ag.registry.needs_confirmation(routed.tool, routed.args):
                    return _fixed(await self._ask_confirmation(resp, routed.tool, routed.args, "router"))
                return self._routed_source(resp, routed.tool, routed.args, "router")
        facts = await fetch_live(self._live, resp.intent, resp.user_text, resp.language) if resp.intent.live else None
        resp.live_data = bool(facts) if resp.intent.live else None
        msgs = self._llm_messages(resp.history, resp.language, turn_note(resp.intent, facts))
        return self._llm_source(resp, msgs)

    def _drop_last_user(self, resp: _Response) -> None:
        if resp.context is None:
            return
        msgs = list(resp.context.get_messages())
        if msgs and msgs[-1].get("role") == "user":
            msgs.pop()
            resp.context.set_messages(msgs)

    async def _ask_confirmation(self, resp: _Response, tool: str, args: dict, source: str) -> str:
        from voice_agent.server.agent import Pending

        self._agent.pending = Pending(tool, args, resp.language)
        q = speech.confirm_question(tool, args, resp.language)
        await self._send(resp.turn_id, {"type": "confirm_request", "turn_id": resp.turn_id, "tool": tool,
                                        "args": args, "question": q, "source": source})
        return q

    async def _run_tool(self, resp: _Response, name: str, args: dict, source: str) -> ToolResult:
        reg = self._agent.registry
        tool = reg.get(name)
        args = dict(args or {})
        if tool is not None and "language" in tool.parameters and "language" not in args:
            args["language"] = resp.language
        await self._send(resp.turn_id, {"type": "tool_start", "turn_id": resp.turn_id, "tool": name,
                                        "args": args, "source": source})
        res = await reg.run(name, args)
        resp.actions.append({"tool": name, "ok": res.ok, "ms": round(res.ms), "source": source})
        logger.info(f"turn {resp.turn_id}: tool {name}({args}) via {source} -> ok={res.ok} "
                    f"{res.error or res.data} [{res.ms:.0f} ms]")
        evt = {"type": "tool_result", "turn_id": resp.turn_id, "tool": name, "ok": res.ok, "ms": round(res.ms),
               "data": res.data, "error": res.error}
        if res.image:
            evt["image"] = "/shots/" + res.image.replace("\\", "/").rsplit("/", 1)[-1]
        # tool results reach the browser even if the user already interrupted the reply
        await self.push_frame(OutputTransportMessageUrgentFrame(message=evt))
        return res

    async def _routed_source(self, resp: _Response, tool: str, args: dict, source: str):
        """Router/confirmed action: run the tool, speak a template (no LLM)."""
        call = {"function": {"name": tool, "arguments": args}}
        resp.tool_msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
        task = asyncio.create_task(self._run_tool(resp, tool, args, source))
        try:
            if tool in speech.PRE:                  # slow tool: acknowledge now, outcome later
                yield speech.pre(tool, args, resp.language)
                yield FLUSH
                res = await task
                more = speech.follow_up(tool, args, res, resp.language)
                if more:
                    yield " " + more
            elif tool in OPTIMISTIC:
                yield speech.say(tool, args, ToolResult(True), resp.language)
                yield FLUSH
                res = await task
                if not res.ok:
                    yield " " + speech.say(tool, args, res, resp.language)
            else:
                res = await task
                yield speech.say(tool, args, res, resp.language)
        finally:
            if not task.done():
                task.cancel()
        if task.done() and not task.cancelled():
            res = task.result()
            resp.tool_msgs.append({"role": "tool", "tool_name": tool, "content": res.for_llm()})

    async def _llm_source(self, resp: _Response, msgs: list[dict]):
        """LLM text, with tool calls executed between steps (agent loop)."""
        tools = self._agent.schemas() if self._agent is not None else None
        events = getattr(self._llm, "stream_events", None)
        for step in range(self._agent.max_steps if self._agent else 1):
            use_tools = tools if step < (self._agent.max_steps - 1 if self._agent else 0) else None
            if events is None or not tools:
                async for d in self._llm.stream_chat(msgs):
                    yield d
                return
            # A command must end in a tool call, so its text is held back first. A reply like
            # "I'll lower it" with no tool call is retried once and never spoken.
            hold = step == 0 and is_command(resp.user_text or "")
            calls, step_text = [], ""
            stream = resp.prefetch.events() if step == 0 and resp.prefetch is not None else events(msgs, use_tools)
            async for ev in stream:
                if isinstance(ev, str):
                    step_text += ev
                    if not hold:
                        yield ev
                elif self._agent.registry.get(ev.name) is not None:
                    calls.append(ev)
                else:
                    logger.warning(f"turn {resp.turn_id}: LLM called unknown tool {ev.name!r}")
            if hold and needs_tool_retry(resp.user_text or "", calls, step_text):
                logger.info(f"turn {resp.turn_id}: LLM claimed an action without a tool call; asking again")
                retry = msgs + [{"role": "assistant", "content": step_text},
                                {"role": "user", "content": retry_note(resp.language)}]
                step_text = ""
                async for ev in events(retry, use_tools):
                    if isinstance(ev, str):
                        step_text += ev
                    elif self._agent.registry.get(ev.name) is not None:
                        calls.append(ev)
                if not calls and needs_tool_retry(resp.user_text or "", calls, step_text):
                    step_text = speech.cannot_do(resp.language)   # never claim an action that did not happen
            if step == 0:
                st = getattr(self._llm, "stats", None)
                if st is not None and getattr(st, "prompt_tokens", 0):
                    resp.llm_prompt = {"tokens": st.prompt_tokens, "ms": round(st.prompt_ms)}
            if hold and step_text:
                yield step_text
            if not calls:
                return
            yield FLUSH                                     # speak "Sure, opening it." while the tool runs
            asked = None
            for c in calls:
                if self._agent.registry.needs_confirmation(c.name, c.arguments):
                    asked = c
                    break
            done_calls = calls if asked is None else calls[:calls.index(asked)]
            call_msg = {"role": "assistant", "content": step_text,
                        "tool_calls": [c.to_message() for c in (done_calls + ([asked] if asked else []))]}
            msgs.append(call_msg)
            resp.tool_msgs.append(call_msg)
            for c in done_calls:
                res = await self._run_tool(resp, c.name, c.arguments, "llm")
                tm = {"role": "tool", "tool_name": c.name, "content": res.for_llm()}
                msgs.append(tm)
                resp.tool_msgs.append(tm)
            if done_calls:
                # tool results are English JSON: repeat the reply language right after them
                msgs[-1] = {**msgs[-1], "content": _with_language(msgs[-1]["content"], resp.language)}
            if asked is not None:
                q = await self._ask_confirmation(resp, asked.name, asked.arguments, "llm")
                resp.tool_msgs.append({"role": "tool", "tool_name": asked.name,
                                       "content": '{"ok": false, "status": "waiting for the user to confirm"}'})
                yield (" " if step_text else "") + q
                return
            if step_text and not step_text.endswith(" "):
                yield " "

    @staticmethod
    async def _watch_loop_lag(resp: _Response, period: float = 0.02) -> None:
        """How late the event loop wakes up. High values mean something (e.g. the TTS
        thread) is delaying audio sends and turn handling."""
        loop = asyncio.get_running_loop()
        while True:
            t = loop.time()
            await asyncio.sleep(period)
            resp.loop_lag_ms = max(resp.loop_lag_ms, (loop.time() - t - period) * 1000)

    async def _produce(self, resp: _Response, source, queue: asyncio.Queue) -> None:
        """Text source (LLM deltas, tool templates) -> browser text + speakable chunks."""
        chunker = SpeechChunker(self._chunker_cfg, resp.language)
        guard = LanguageGuard()  # nothing outside Arabic/Latin script reaches TTS or the UI
        try:
            async for raw in source:
                if not self._current(resp.turn_id):
                    return
                if raw is FLUSH:
                    tail = guard.flush()
                    if tail:
                        resp.text += tail
                        await self._send(resp.turn_id, {"type": "llm_delta", "turn_id": resp.turn_id, "text": tail})
                    for chunk in chunker.push(tail) + chunker.flush():
                        resp.first_chunk = resp.first_chunk or self._clock()
                        await queue.put(chunk)
                    continue
                if resp.ttft is None:
                    resp.ttft = self._clock()
                delta = guard.feed(raw)
                if not delta:
                    continue
                resp.text += delta
                await self._send(resp.turn_id, {"type": "llm_delta", "turn_id": resp.turn_id, "text": delta})
                for chunk in chunker.push(delta):
                    resp.first_chunk = resp.first_chunk or self._clock()
                    await queue.put(chunk)
            tail = guard.flush()
            if tail:
                resp.text += tail
                for chunk in chunker.push(tail):
                    await queue.put(chunk)
            for chunk in chunker.flush():
                resp.first_chunk = resp.first_chunk or self._clock()
                await queue.put(chunk)
            if guard.removed:
                logger.warning(f"turn {resp.turn_id}: removed {guard.removed} chars of foreign script from LLM "
                               f"output ({guard.drift_ratio:.0%}); reply language {resp.language}")
                await self._send(resp.turn_id, {"type": "lang_guard", "turn_id": resp.turn_id,
                                                "removed": guard.removed})
        finally:
            resp.produced_all = True
            await queue.put(None)

    async def _consume(self, resp: _Response, queue: asyncio.Queue) -> None:
        """Chunks -> TTS -> audio frames, as soon as each chunk is ready."""
        carry: str | None = None
        while True:
            chunk = carry if carry is not None else await queue.get()
            carry = None
            if chunk is None:
                break
            if self._tts is None or not self._current(resp.turn_id):
                continue  # text-only mode: just drain
            # Speak only the part that is ready before the queued audio ends; the rest is
            # carried over. Same rule as latency-bench (tts_budget.plan_chunk).
            # one voice per reply, in the reply's language (not guessed per chunk)
            voice = pick_voice(self._voices, resp.language, self._session.gender)
            if not _is_streaming(self._tts, voice):
                # Speak only the part that is ready before the queued audio ends; the rest is
                # carried over. A streaming voice (XTTS) starts playing early anyway, and short
                # pieces sound choppy, so its chunks are not split.
                ahead = max(0.0, resp.tracker.ends_at - self._clock()) if resp.chunks else 0.0
                plan = plan_chunk(chunk, resp.language, self._cost, ahead)
                chunk, carry = plan.text, plan.carry
            final = carry is None and resp.produced_all and queue.qsize() <= 1
            say = speakable(chunk, resp.language, final)
            first_piece = True
            t_req = self._clock()
            stat = {"words": len(chunk.split())}
            audio_s = 0.0
            sr = self._tts.sample_rate
            async for audio in self._synthesize(resp.turn_id, say, voice):
                if not self._current(resp.turn_id):
                    return
                if resp.first_audio is None:
                    resp.first_audio = self._clock()
                    resp.first_audio_wall = time.time()
                    await self._send(resp.turn_id, {"type": "audio_start", "turn_id": resp.turn_id}, ordered=True)
                if first_piece:
                    stat["first_piece_ms"] = round((self._clock() - t_req) * 1000)
                    resp.chunks += 1
                    await self._send(resp.turn_id, {"type": "tts_chunk", "turn_id": resp.turn_id,
                                                    "text": chunk, "voice": voice.name,
                                                    "voice_id": getattr(voice, "voice_id", "")},
                                 ordered=True)
                start = resp.tracker.add(chunk, len(audio) / sr, self._clock(), new_chunk=first_piece)
                if self._monitor is not None:
                    self._monitor.add(audio, sr, start)
                    if first_piece:   # text span for echo matching: the whole chunk, estimated
                        est = max(len(audio) / sr, stat["words"] * self._cost.s_per_word.get(resp.language, 0.4))
                        self._monitor.add_text(chunk, start, start + est)
                await self.push_frame(TTSAudioRawFrame(float_to_pcm16(audio), sr, 1))
                audio_s += len(audio) / sr
                first_piece = False
            if first_piece or not self._current(resp.turn_id):
                continue                               # nothing synthesized (cancelled / empty)
            stat.update(synth_ms=round((self._clock() - t_req) * 1000), audio_s=round(audio_s, 2))
            engine_stats = getattr(self._tts, "last_stats", {}) or {}
            stat.update({k: v for k, v in engine_stats.items()
                         if k in ("first_piece_ms", "voice")})
            resp.tts.append(stat)
            # learn this machine's TTS speed (used to keep the audio gap-free)
            self._cost.observe(resp.language, stat["words"], stat["audio_s"], stat["synth_ms"] / 1000)

    async def _synthesize(self, turn_id: int, text: str, voice: Voice) -> AsyncIterator:
        """Run the blocking TTS in a worker thread and yield audio pieces as they arrive.

        The thread holds the TTS lock for the whole chunk, so a cancelled chunk stops
        (quickly, thanks to cancel()) before the next chunk starts.
        """
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def put(item) -> None:
            try:
                loop.call_soon_threadsafe(q.put_nowait, item)
            except RuntimeError:
                pass  # loop closed (shutdown)

        def work() -> None:
            with self._tts_lock:
                try:
                    reset_cancel(self._tts)          # clear first, then check: no lost cancel
                    if not self._current(turn_id):
                        return
                    for piece in tts_synthesize(self._tts, text, voice):
                        if not self._current(turn_id):
                            break
                        put(piece)
                except Exception as e:  # noqa: BLE001
                    put(e)
                finally:
                    put(None)

        loop.run_in_executor(None, work)
        while (item := await q.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item

    def _turn_metrics(self, r: _Response) -> dict:
        def ms(t):
            return None if t is None else round((t - r.started) * 1000)

        m = {"llm_ttft_ms": ms(r.ttft), "first_chunk_ms": ms(r.first_chunk),
             "first_audio_ms": ms(r.first_audio), "chunks": r.chunks}
        if r.user_stopped is not None and r.first_audio is not None:
            m["user_stop_to_audio_ms"] = round((r.first_audio - r.user_stopped) * 1000)
        if r.speech_end_wall is not None and r.first_audio_wall is not None:
            # the real wait: from the end of the user's speech, including VAD and Smart Turn
            m["speech_end_to_audio_ms"] = round((r.first_audio_wall - r.speech_end_wall) * 1000)
        if r.prefetch is not None:
            m["prefetched"] = True
        if r.llm_prompt:
            m["llm_prompt"] = r.llm_prompt
        if r.tts:
            m["first_chunk_words"] = r.tts[0]["words"]
            m["first_synth_ms"] = r.tts[0].get("first_piece_ms", r.tts[0]["synth_ms"])  # until first sound
            if r.tts[0].get("stream"):
                m["first_streamed"] = True
            m["tts"] = r.tts
        if r.actions:
            m["actions"] = r.actions
        m["gap_ms"] = round(r.tracker.gap_s * 1000)          # silence heard between chunks
        m["loop_lag_ms"] = round(r.loop_lag_ms)
        return m


def _with_language(tool_json: str, lang: str) -> str:
    """Add the reply language to a tool result (still valid JSON)."""
    import json

    try:
        d = json.loads(tool_json)
    except json.JSONDecodeError:
        return f"{tool_json}\n[{reminder(lang)}]"
    d["reply_in"] = "Arabic" if lang == "ar" else "English"
    return json.dumps(d, ensure_ascii=False)


def _is_streaming(tts, voice) -> bool:
    fn = getattr(tts, "is_streaming", None)
    return bool(fn(voice)) if fn else bool(getattr(tts, "streaming", False))


def _stt_language(parts: list[tuple[str, str]]) -> str | None:
    """The language STT heard most (by text length) in this turn."""
    score: dict[str, int] = {}
    for text, lang in parts:
        score[lang] = score.get(lang, 0) + len(text)
    return max(score, key=score.get) if score else None


def _merge_last_user_messages(context) -> None:
    """[..., user "I want to know", user "the weather"] -> [..., user "I want to know the weather"]."""
    msgs = list(context.get_messages())
    if len(msgs) >= 2 and msgs[-1].get("role") == "user" and msgs[-2].get("role") == "user":
        joined = f"{_text(msgs[-2].get('content'))} {_text(msgs[-1].get('content'))}".strip()
        context.set_messages(msgs[:-2] + [{"role": "user", "content": joined}])


async def _fixed(text: str):
    yield text


def _after_wake_word(text: str) -> str:
    """'HAZEM, open notepad' -> 'open notepad'; 'HAZEM' -> ''."""
    import re
    m = re.search(r"(?:\bHAZEM\b|حزم|حزم|حزم|\bwake up\b|استيقظ|اصحى|اصحي)[\s,،.!?؟]*", text, re.I)
    return text[m.end():].strip() if m else text


def _has_new_utterance(context) -> bool:
    """Reply only if the newest message is a non-empty user message (the current utterance)."""
    msgs = [m for m in context.get_messages() if m.get("role") != "system"]
    return bool(msgs) and msgs[-1].get("role") == "user" and bool(_text(msgs[-1].get("content")).strip())


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""
