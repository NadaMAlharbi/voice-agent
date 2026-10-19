"""Headless test client: streams wav files to the server like a mic, prints what comes back.

  uv run ws-probe tests\\audio\\en_question.wav                       (any mode)
  uv run ws-probe recordings\\en_complete.wav --barge-in recordings\\ar_complete.wav --after 1.0
      -> in assistant mode: speaks the 2nd clip 1.0 s after the assistant starts talking

It uses the same turn gate as the browser (static/turn-gate.js), so it also reports
how much old audio would have been dropped after a barge-in.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from voice_agent.audio_io import float_to_pcm16, load_wav, pcm16_frames

SILENCE_FRAME = b"\x00" * 640


@dataclass
class TurnGate:
    """Mirror of static/turn-gate.js: which audio may be played."""
    accept: int | None = None      # turn currently allowed to play
    max_cancelled: int = 0
    accepted: dict[int, int] = field(default_factory=dict)  # turn -> bytes
    dropped: int = 0
    flushes: int = 0

    def on_event(self, ev: dict) -> None:
        t = ev.get("type")
        if t == "interrupt":
            self.accept = None
            self.flushes += 1
        elif t == "cancel_turn":
            self.max_cancelled = max(self.max_cancelled, ev["turn_id"])
            if self.accept is not None and self.accept <= ev["turn_id"]:
                self.accept = None
                self.flushes += 1
        elif t == "audio_start" and ev["turn_id"] > self.max_cancelled:
            self.accept = ev["turn_id"]

    def on_audio(self, nbytes: int) -> bool:
        if self.accept is None:
            self.dropped += nbytes
            return False
        self.accepted[self.accept] = self.accepted.get(self.accept, 0) + nbytes
        return True


async def probe(url: str, wav: str, tail_s: float = 4.0, realtime: bool = True,
                barge_in: str | None = None, barge_after_s: float = 1.0, gate_audio: bool = False,
                echo_gain: float = 0.0, echo_delay_s: float = 0.15) -> dict:
    """gate_audio=False (loopback) counts all audio; True applies the TurnGate.

    echo_gain > 0 simulates laptop speakers: audio the "browser" plays is mixed
    back into the mic stream after echo_delay_s (an acoustic loop, no AEC).
    """
    import websockets

    audio = load_wav(wav, 16000)
    speech_end_s = _last_speech_s(audio)
    audio = np.concatenate([audio, np.zeros(int(tail_s * 16000), np.float32)])
    frames = list(pcm16_frames(float_to_pcm16(audio), 16000, 20))
    barge_frames = list(pcm16_frames(float_to_pcm16(load_wav(barge_in, 16000)), 16000, 20)) if barge_in else []

    gate = TurnGate()
    result = {"events": [], "audio_bytes": 0, "first_audio_ms": None, "gate": gate, "barge_at_ms": None}
    sent_all = asyncio.Event()
    first_assistant_audio = asyncio.Event()
    speaker = np.zeros(0, np.float32)   # 16 kHz samples waiting to be "played" (echo source)

    def with_echo(frame: bytes) -> bytes:
        nonlocal speaker
        if echo_gain <= 0 or len(speaker) == 0:
            return frame
        n = len(frame) // 2
        echo, speaker = speaker[:n], speaker[n:]
        mic = np.frombuffer(frame, np.int16).astype(np.float32) / 32768.0
        mic[: len(echo)] += echo_gain * echo
        return float_to_pcm16(mic)

    async with websockets.connect(url, max_size=None) as ws:
        t0 = time.perf_counter()

        async def send_paced(fr: list[bytes]):
            start = time.perf_counter()
            for i, f in enumerate(fr):
                await ws.send(with_echo(f))
                if realtime:
                    await asyncio.sleep(max(0.0, start + (i + 1) * 0.02 - time.perf_counter()))

        async def sender():
            await send_paced(frames)
            if barge_frames:
                # keep the "mic" open with silence until the assistant talks, then interrupt
                while not first_assistant_audio.is_set() and time.perf_counter() - t0 < 60:
                    await send_paced([SILENCE_FRAME] * 5)
                await send_paced([SILENCE_FRAME] * int(barge_after_s / 0.02))
                result["barge_at_ms"] = round((time.perf_counter() - t0) * 1000)
                await send_paced(barge_frames + [SILENCE_FRAME] * int(tail_s / 0.02))
            await ws.send(json.dumps({"type": "ping", "t": time.perf_counter()}))
            sent_all.set()

        async def receiver():
            nonlocal speaker
            try:
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    except asyncio.TimeoutError:
                        if sent_all.is_set():
                            break  # quiet after everything was sent: done
                        continue
                    now = (time.perf_counter() - t0) * 1000
                    if isinstance(msg, bytes):
                        if result["first_audio_ms"] is None:
                            result["first_audio_ms"] = now
                        result["audio_bytes"] += len(msg)
                        played = gate.on_audio(len(msg)) if gate_audio else True
                        if echo_gain > 0 and played:
                            import soxr

                            pcm = np.frombuffer(msg, np.int16).astype(np.float32) / 32768.0
                            if len(speaker) == 0:  # playback (re)starts: speaker->mic delay
                                speaker = np.zeros(int(echo_delay_s * 16000), np.float32)
                            speaker = np.concatenate([speaker, soxr.resample(pcm, 24000, 16000).astype(np.float32)])
                    else:
                        ev = json.loads(msg)
                        if gate.on_event(ev) == "flush" or ev["type"] in ("interrupt", "cancel_turn"):
                            speaker = np.zeros(0, np.float32)   # browser flushed playback
                        if ev["type"] == "audio_start":
                            first_assistant_audio.set()
                        ev["at_ms"] = round(now)
                        result["events"].append(ev)
            except websockets.ConnectionClosed:
                pass

        rx = asyncio.create_task(receiver())
        await sender()
        await rx
    result["speech_end_ms"] = round(speech_end_s * 1000)
    return result


def _last_speech_s(audio: np.ndarray, thresh: float = 0.02) -> float:
    idx = np.nonzero(np.abs(audio) > thresh)[0]
    return float(idx[-1]) / 16000 if len(idx) else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stream a wav to the server and print results")
    ap.add_argument("wav")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--barge-in", default=None, help="second clip spoken while the assistant talks")
    ap.add_argument("--after", type=float, default=1.0, help="seconds of assistant audio before barging in")
    ap.add_argument("--show-deltas", action="store_true")
    args = ap.parse_args(argv)
    r = asyncio.run(probe(args.url, args.wav, barge_in=args.barge_in, barge_after_s=args.after,
                          gate_audio=True))

    print(f"speech ends at {r['speech_end_ms']} ms in the first clip")
    if r["barge_at_ms"]:
        print(f"barge-in clip starts at {r['barge_at_ms']} ms")
    for ev in r["events"]:
        at = ev.pop("at_ms")
        if ev["type"] == "llm_delta" and not args.show_deltas:
            continue
        if ev["type"] == "assistant_done":
            ev["text"] = ev["text"][:60] + ("..." if len(ev["text"]) > 60 else "")
        print(f"{at:7d} ms  {json.dumps(ev, ensure_ascii=False)}")
    g = r["gate"]
    if r["first_audio_ms"] is not None:
        per_turn = {k: f"{v / 48000:.1f}s" for k, v in g.accepted.items()}
        print(f"audio: {r['audio_bytes'] / 48000:.1f}s received, first after {r['first_audio_ms']:.0f} ms | played per turn {per_turn} | "
              f"stale dropped {g.dropped / 48000:.2f}s | flushes {g.flushes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
