"""Speech detection (Silero VAD) + end-of-turn detection (Smart Turn v3.2).

Uses Pipecat's own analyzers with the same settings as the live pipeline,
fed frame by frame from a file:

  VAD SPEAKING       -> speech_start
  VAD back to QUIET  -> speech_stop, then Smart Turn decides:
                        complete   -> turn_end   (hand audio to STT)
                        incomplete -> keep listening (user paused mid-sentence)
  silence >= turn.stop_secs        -> turn_end (fallback)

  uv run vad-test recordings/en_question.wav
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from voice_agent.config import Config, load_config
from voice_agent.models import enable_offline

SR = 16000
FRAME_MS = 20


@dataclass
class TurnEvent:
    t: float                 # position in the audio, seconds
    kind: str                # speech_start | speech_stop | turn_end | turn_incomplete
    probability: float | None = None
    infer_ms: float | None = None


@dataclass
class TurnStats:
    vad_ms_per_frame: float
    turn_infer_ms: list[float]


class TurnDetector:
    def __init__(self, cfg: Config):
        enable_offline()
        from loguru import logger

        logger.remove()  # keep CLI output clean; pipecat logs are debug/info-level
        logger.add(sys.stderr, level="WARNING")

        from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams, VADState

        self._VADState = VADState
        self.vad = SileroVADAnalyzer(sample_rate=SR, params=VADParams(**vars(cfg.vad)))
        self.vad.set_sample_rate(SR)
        self.turn = LocalSmartTurnAnalyzerV3(
            sample_rate=SR, params=SmartTurnParams(**vars(cfg.turn))
        )
        self.turn.set_sample_rate(SR)
        self.turn.update_vad_start_secs(cfg.vad.start_secs)

    async def process(self, frames: list[bytes]) -> tuple[list[TurnEvent], TurnStats]:
        from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState

        Q, SPEAKING = self._VADState.QUIET, self._VADState.SPEAKING
        events: list[TurnEvent] = []
        infer_ms: list[float] = []
        user_speaking = False
        vad_time = 0.0

        for i, frame in enumerate(frames):
            t = (i + 1) * FRAME_MS / 1000

            t0 = time.perf_counter()
            state = await self.vad.analyze_audio(frame)
            vad_time += time.perf_counter() - t0

            if state == SPEAKING and not user_speaking:
                user_speaking = True
                events.append(TurnEvent(t, "speech_start"))
            elif state == Q and user_speaking:
                user_speaking = False
                events.append(TurnEvent(t, "speech_stop"))
                self.turn.append_audio(frame, False)
                eot, metrics = await self.turn.analyze_end_of_turn()
                prob = getattr(metrics, "probability", None)
                ms = getattr(metrics, "e2e_processing_time_ms", None)
                if ms is not None:
                    infer_ms.append(ms)
                kind = "turn_end" if eot == EndOfTurnState.COMPLETE else "turn_incomplete"
                events.append(TurnEvent(t, kind, prob, ms))
                continue

            # Fallback: long silence after an incomplete turn ends it anyway.
            if self.turn.append_audio(frame, user_speaking) == EndOfTurnState.COMPLETE:
                events.append(TurnEvent(t, "turn_end"))

        stats = TurnStats(vad_time / max(len(frames), 1) * 1000, infer_ms)
        return events, stats


def detect_file(path: str | Path, cfg: Config | None = None):
    """Run detection on a file, as if streamed from a mic.

    Appends turn.stop_secs + 0.5 s of silence so the last turn always closes
    (by Smart Turn or by the silence fallback).
    """
    import numpy as np

    from voice_agent.audio_io import float_to_pcm16, load_wav, pcm16_frames

    cfg = cfg or load_config()
    audio = load_wav(path, SR)
    tail = np.zeros(int((cfg.turn.stop_secs + 0.5) * SR), np.float32)
    audio = np.concatenate([audio, tail])
    frames = list(pcm16_frames(float_to_pcm16(audio), SR, FRAME_MS))
    detector = TurnDetector(cfg)
    return asyncio.run(detector.process(frames))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run VAD + Smart Turn on audio files")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args(argv)

    for f in args.files:
        events, stats = detect_file(f)
        print(f"\n== {f}")
        for e in events:
            extra = ""
            if e.probability is not None:
                extra = f"  p(complete)={e.probability:.2f}  {e.infer_ms:.1f} ms"
            print(f"  {e.t:6.2f}s  {e.kind:<16}{extra}")
        if not events:
            print("  (no speech detected)")
        turns = [e for e in events if e.kind == "turn_end"]
        infer = f"{max(stats.turn_infer_ms):.1f} ms max" if stats.turn_infer_ms else "n/a"
        print(f"  turns: {len(turns)} | VAD {stats.vad_ms_per_frame:.2f} ms/frame | Smart Turn {infer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
