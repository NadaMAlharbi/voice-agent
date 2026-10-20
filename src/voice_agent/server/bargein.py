"""Barge-in: the assistant's own voice (echo) and short noises must not interrupt; real speech must.

While the assistant is silent, a user turn starts as soon as VAD hears speech.
While it is speaking, VAD speech is only a candidate:

  1. Minimum length: the user must speak for `min_speech_s` (default 0.6 s).
     Coughs, "mm" and clicks are shorter and are ignored.
     Minimum loudness: quieter than `min_volume` (same scale as vad.min_volume) is a
     voice far away (another person, a TV) and is ignored.
  2. Echo check: the mic audio is compared with what the assistant was playing.
     A close match (score >= `echo_threshold`) is echo and is ignored. The check
     repeats every 100 ms, so a user talking over the echo still interrupts.
  3. Short commands ("stop", "توقف") still interrupt once their text arrives,
     unless they are backchannels ("mm", "okay", "اها") or echo.
  4. Text from ignored segments is dropped, so it never joins the next user turn.

The browser's own echo cancellation stays on; this is the second safety layer.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy

HOP = 0.01          # 10 ms analysis hop
SR = 16000
_HOP_N, _WIN_N = 160, 320
BACKCHANNELS = {
    "mm", "mmm", "mhm", "hmm", "hm", "uh", "um", "uhm", "ah", "oh", "aha", "uh-huh", "yeah", "yep",
    "ok", "okay", "right", "sure",
    "ام", "امم", "اممم", "ممم", "اه", "آه", "اها", "آها", "ايه", "إيه", "ايوه", "طيب", "اوكي", "تمام",
}


def is_backchannel(text: str) -> bool:
    words = [w.strip(".,!?؟،").lower() for w in text.split()]
    words = [w for w in words if w]
    return not words or all(w in BACKCHANNELS for w in words)


def log_spectrum(x: np.ndarray) -> np.ndarray:
    """(frames, 40) log band energies, 200 Hz - 4.2 kHz, 20 ms window, 10 ms hop at 16 kHz."""
    n = (len(x) - _WIN_N) // _HOP_N + 1
    if n <= 0:
        return np.zeros((0, 40))
    idx = np.arange(_WIN_N)[None, :] + _HOP_N * np.arange(n)[:, None]
    mag = np.abs(np.fft.rfft(x[idx] * np.hanning(_WIN_N), axis=1))[:, 4:84]
    return np.log(mag.reshape(n, 40, 2).mean(axis=2) ** 2 + 1e-8)


class PlaybackMonitor:
    """What the assistant is playing (16 kHz audio on a clock timeline) and saying (text).
    One per connection; used as the echo reference for barge-in."""

    def __init__(self, keep_s: float = 30.0):
        self._segs: deque[tuple[float, np.ndarray]] = deque()   # (start time, 16 kHz audio)
        self._texts: deque[tuple[float, float, str]] = deque()   # (start, end, text)
        self.keep_s = keep_s
        self.end = 0.0

    def add(self, audio: np.ndarray, sr: int, start: float) -> None:
        import soxr

        a = audio if sr == SR else soxr.resample(audio, sr, SR).astype(np.float32)
        self._segs.append((start, a))
        self.end = max(self.end, start + len(a) / SR)
        while self._segs and self._segs[0][0] + len(self._segs[0][1]) / SR < self.end - self.keep_s:
            self._segs.popleft()

    def add_text(self, text: str, start: float, end: float) -> None:
        self._texts.append((start, end, text))
        while self._texts and self._texts[0][1] < end - self.keep_s:
            self._texts.popleft()

    def truncate(self, at: float) -> None:
        """Barge-in: everything scheduled after `at` was flushed and never played."""
        kept: deque = deque()
        for start, a in self._segs:
            if start < at:
                kept.append((start, a[: max(0, int((at - start) * SR))]))
        self._segs = kept
        self._texts = deque(t for t in self._texts if t[0] < at)
        self.end = min(self.end, at)

    def audio(self, t0: float, t1: float) -> np.ndarray:
        """Assistant audio (16 kHz) scheduled between t0 and t1; zeros where silent."""
        out = np.zeros(max(0, int(round((t1 - t0) * SR))), np.float32)
        for start, a in self._segs:
            i0 = int(round((start - t0) * SR))
            lo, hi = max(0, i0), min(len(out), i0 + len(a))
            if lo < hi:
                out[lo:hi] += a[lo - i0: hi - i0]
        return out

    def text_overlap(self, text: str, since: float) -> float:
        """Share of the transcript's words the assistant said (in chunks playing after `since`)."""
        words = [w.strip(".,!?؟،").lower() for w in text.split()]
        words = [w for w in words if w]
        if not words:
            return 0.0
        spoken = {w.strip(".,!?؟،").lower() for s, e, t in self._texts if e >= since for w in t.split()}
        return sum(w in spoken for w in words) / len(words)


def echo_score(mic: np.ndarray, mic_end: float, monitor: PlaybackMonitor, max_lag_s: float = 0.6,
               min_active: float = 0.3) -> float:
    """How well the mic audio matches the assistant's own audio, at a delay of 0..max_lag.

    Echo is a filtered copy of what was played, so its spectrum over time matches;
    a person talking does not. Returns 0 when the assistant was mostly silent."""
    M = log_spectrum(mic)
    if len(M) < 10:
        return 0.0
    dur = len(mic) / SR
    ref = monitor.audio(mic_end - dur - max_lag_s, mic_end)
    R = log_spectrum(ref)
    if len(R) == 0:
        return 0.0
    idx = np.arange(_WIN_N)[None, :] + _HOP_N * np.arange(len(R))[:, None]
    active = np.sqrt(np.mean(ref[idx] ** 2, axis=1)) > 1e-3        # assistant audible in that frame
    lags = int(round(max_lag_s / HOP))
    Mc = M - M.mean(axis=0)
    best = 0.0
    for lag in range(lags + 1):
        off = lags - lag
        if off + len(M) > len(R):
            continue
        if np.mean(active[off: off + len(M)]) < min_active:
            continue
        seg = R[off: off + len(M)]
        Rc = seg - seg.mean(axis=0)
        den = np.sqrt((Mc * Mc).sum() * (Rc * Rc).sum())
        if den > 1e-9:
            best = max(best, float((Mc * Rc).sum() / den))
    return best


class BargeInStartStrategy(BaseUserTurnStartStrategy):
    """User turn start strategy: instant when the assistant is silent, guarded while it speaks."""

    def __init__(self, monitor: PlaybackMonitor | None, min_speech_s: float = 0.6,
                 echo_threshold: float = 0.6, max_echo_lag_s: float = 0.6, min_volume: float = 0.0,
                 clock: Callable[[], float] = time.monotonic, **kwargs):
        super().__init__(**kwargs)
        self._monitor = monitor
        self._min_speech_s = min_speech_s
        self._min_volume = min_volume
        self._echo_threshold = echo_threshold
        self._max_lag = max_echo_lag_s
        self._clock = clock
        self._bot_speaking = False
        self._in_turn = False
        self._candidate = False     # VAD speech while the bot talks, not decided yet
        self._rejected = False      # current VAD segment was judged echo / noise
        self._reject_reason = ""
        self._segment_end = 0.0
        self._last_eval = 0
        self._mic: list[np.ndarray] = []   # raw 16 kHz mic audio of the candidate segment
        self._mic_n = 0

    async def handle_user_turn_started(self):
        self._in_turn = True
        self._candidate = False

    async def handle_user_turn_stopped(self):
        self._in_turn = False

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            if self._candidate and not self._rejected:  # user kept talking after the bot finished
                return await self._start("bot finished while user speaking")
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._rejected, self._reject_reason = False, ""
            if not self._bot_speaking:
                return await self._start("vad")
            self._candidate = True
            self._mic, self._mic_n, self._last_eval = [], 0, 0
        elif isinstance(frame, InputAudioRawFrame) and self._candidate:
            return await self._on_candidate_audio(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._candidate and self._reject_reason in ("echo", "too_quiet"):
                self._candidate = False           # a stretch of echo / far voice ended; already reported
            elif self._candidate:
                self._candidate = False
                # too short for a reliable acoustic echo check: decided from its transcript
                self._rejected, self._reject_reason = True, "too_short"
                self._segment_end = self._clock()
                await self._report("too_short", duration_s=round(self._mic_n / SR, 2))
        elif isinstance(frame, TranscriptionFrame) and not self._in_turn:
            said_by_bot = (self._monitor is not None and
                           self._monitor.text_overlap(frame.text, self._segment_end - 10.0) >= 0.6)
            if self._reject_reason == "too_short" and not is_backchannel(frame.text) and not said_by_bot:
                # a short real command ("stop", "توقف"): interrupt now; the text is already aggregated
                self._reject_reason = ""
                return await self._start("short command", text=frame.text)
            # Text from a segment that never became a turn (echo / noise / backchannel):
            # drop it so it does not get glued to the next real user turn.
            await self.trigger_reset_aggregation()
            why = "echo (bot's own words)" if said_by_bot else (self._reject_reason or "no turn")
            await self._report("transcript_dropped", text=frame.text, why=why)
        return ProcessFrameResult.CONTINUE

    async def _on_candidate_audio(self, frame: InputAudioRawFrame) -> ProcessFrameResult:
        x = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
        self._mic.append(x)
        self._mic_n += len(x)
        self._mic_end = self._clock()
        if self._mic_n < self._min_speech_s * SR:
            return ProcessFrameResult.CONTINUE
        # Sliding window: VAD can stay on through long echo; if the user starts talking
        # over it, the latest window stops matching the assistant.
        if self._reject_reason in ("echo", "too_quiet") and self._mic_n - self._last_eval < 0.1 * SR:
            return ProcessFrameResult.CONTINUE
        self._last_eval = self._mic_n
        volume = self._volume()
        if volume < self._min_volume:
            if self._reject_reason != "too_quiet":
                self._rejected, self._reject_reason = True, "too_quiet"
                await self._report("too_quiet", volume=round(volume, 2))
            return ProcessFrameResult.CONTINUE
        score = self._score()
        if score >= self._echo_threshold:
            if self._reject_reason != "echo":
                self._rejected, self._reject_reason = True, "echo"
                await self._report("echo", echo_score=round(score, 2))
            return ProcessFrameResult.CONTINUE
        return await self._start("barge-in", echo_score=round(score, 2))

    def _volume(self) -> float:
        """Loudness of the last min_speech_s of mic audio, 0..1 (pipecat's VAD scale)."""
        if not self._min_volume or not self._mic:
            return 1.0
        from pipecat.audio.utils import calculate_audio_volume

        mic = np.concatenate(self._mic)[-int(self._min_speech_s * SR):]
        try:
            return calculate_audio_volume((np.clip(mic, -1, 1) * 32767).astype(np.int16).tobytes(), SR)
        except ValueError:          # shorter than one loudness block
            return 1.0

    def _score(self) -> float:
        if self._monitor is None or not self._mic:
            return 0.0
        mic = np.concatenate(self._mic)[-int(self._min_speech_s * SR):]
        return echo_score(mic, self._mic_end, self._monitor, self._max_lag)

    async def _start(self, reason: str, **info) -> ProcessFrameResult:
        self._candidate = False
        if reason == "barge-in":
            logger.info(f"barge-in accepted {info}")
        await self.trigger_user_turn_started()
        return ProcessFrameResult.STOP

    async def _report(self, reason: str, **info) -> None:
        logger.info(f"barge-in ignored: {reason} {info}")
        await self.push_frame(OutputTransportMessageUrgentFrame(
            message={"type": "barge_in_ignored", "reason": reason, **info}))
