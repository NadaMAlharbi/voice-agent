"""End-to-end latency: end of speech in a clip -> first reply audio.

The clips go through the real pipeline (Silero VAD, Smart Turn, Pipecat, our response
code), streamed in real time like a microphone.

  uv run latency-bench --e2e
      In-process server. VAD and Smart Turn are real. STT, LLM and TTS are simulated
      with fixed timings (SimTimes), so the numbers show the pipeline's own delays.
  uv run latency-bench --e2e --url ws://127.0.0.1:8000/ws
      Your running `uv run serve --mode assistant`: real models, real numbers.
      Command clips are skipped unless --include-commands (they would open apps).

Clips: tests/audio/bench/clips.json (make them with scripts/make_bench_clips.py).
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from voice_agent.config import PROJECT_ROOT, Config, load_config
from voice_agent.stt.base import Transcript

BENCH_DIR = PROJECT_ROOT / "tests" / "audio" / "bench"

REPLY = {
    "en": "Sure. Tokyo is the capital of Japan, and it is famous for food, technology and culture.",
    "ar": "بالتأكيد. طوكيو هي عاصمة اليابان، وهي مشهورة بالطعام والتقنية والثقافة.",
}


@dataclass
class SimTimes:
    """Rough model timings for an RTX 5090 laptop. Estimates, not measurements."""
    stt_base_s: float = 0.08        # Whisper turbo fp16: fixed cost per segment
    stt_per_s: float = 0.03         # ... plus this per second of speech
    llm_ttft_s: float = 0.20        # first token with the prompt cached
    llm_tok_s: float = 80.0         # tokens per second
    kokoro_first_s: float = 0.06    # Kokoro on GPU, per chunk
    xtts_first_s: float = 0.25      # XTTS first streamed piece
    xtts_rtf: float = 0.4           # XTTS seconds per audio second


def load_clips(include_commands: bool = True) -> list[dict]:
    clips = json.loads((BENCH_DIR / "clips.json").read_text(encoding="utf-8"))
    for c in clips:
        c["path"] = str((BENCH_DIR / c["file"]).resolve())
        c["name"] = Path(c["file"]).stem
        c["speech_s"] = speech_seconds(c["path"])
    return [c for c in clips if include_commands or "cmd" not in c["name"]]


def speech_seconds(path: str) -> float:
    """Voiced time in a clip (20 ms frames above a level), close to what VAD passes to STT."""
    from voice_agent.audio_io import load_wav

    a = load_wav(path, 16000)
    frames = a[: len(a) // 320 * 320].reshape(-1, 320)
    return float((np.abs(frames).max(axis=1) > 0.02).sum() * 0.02) + 0.2


class SimSTT:
    """Returns the clip's known text after a delay.

    VAD may cut a clip into several segments. Each call gets a share of the words
    that matches its share of the clip's speech time.
    """
    name = "sim-stt"

    def __init__(self, t: SimTimes):
        self.t = t
        self.words: list[str] = []
        self.speech_left_s = 0.0
        self.language = "en"

    def set_clip(self, text: str, speech_s: float, language: str) -> None:
        self.words, self.speech_left_s, self.language = text.split(), speech_s, language

    def load(self):
        pass

    def device_info(self):
        return "simulated"

    def transcribe(self, audio, language="auto"):
        dur = len(audio) / 16000
        time.sleep(self.t.stt_base_s + self.t.stt_per_s * dur)
        share = 1.0 if dur >= self.speech_left_s - 0.3 else dur / max(self.speech_left_s, 1e-3)
        n = max(1, round(len(self.words) * share)) if self.words else 0
        text, self.words = " ".join(self.words[:n]), self.words[n:]
        self.speech_left_s = max(0.0, self.speech_left_s - dur)
        return Transcript(text, self.language, 0.0, dur, {"speech_s": round(dur, 2)})


class SimLLM:
    """Streams a fixed reply in the user's language, word by word."""

    def __init__(self, t: SimTimes):
        self.t = t
        self.cfg = SimpleNamespace(system_prompt="SYS", cache_friendly=True)

    async def stream_events(self, messages, tools=None):
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        lang = "ar" if any("؀" <= c <= "ۿ" for c in user.split("]")[-1]) else "en"
        await asyncio.sleep(self.t.llm_ttft_s)
        for i, w in enumerate(REPLY[lang].split(" ")):
            if i:
                await asyncio.sleep(1.3 / self.t.llm_tok_s)    # ~1.3 tokens per word
            yield (" " if i else "") + w

    async def stream_chat(self, messages):
        async for ev in self.stream_events(messages):
            yield ev

    async def aclose(self):
        pass


class SimTTS:
    """Silent audio with Kokoro-like or XTTS-like timing (chosen by the voice's engine)."""
    name = "sim-tts"
    sample_rate = 24000

    def __init__(self, t: SimTimes):
        self.t = t
        self._cancel = threading.Event()

    def load(self, voices):
        pass

    def device_info(self):
        return "simulated"

    def cancel(self):
        self._cancel.set()

    def reset_cancel(self):
        self._cancel.clear()

    def synthesize(self, text, voice):
        audio_s = max(0.4, 0.38 * len(text.split()))
        if voice.engine != "xtts":
            time.sleep(self.t.kokoro_first_s)
            yield np.zeros(int(audio_s * self.sample_rate), np.float32)
            return
        time.sleep(self.t.xtts_first_s)
        done = 0.0
        while done < audio_s and not self._cancel.is_set():
            piece = min(0.9, audio_s - done)       # 20 tokens ~ 0.9 s of audio
            if done:
                time.sleep(piece * self.t.xtts_rtf)
            done += piece
            yield np.zeros(int(piece * self.sample_rate), np.float32)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_sim_server(cfg: Config, t: SimTimes):
    """Real server in a thread with simulated engines. Returns (ws url, stt)."""
    import uvicorn

    from voice_agent.server.app import create_app
    from voice_agent.tools.desktop import DryRunDesktop

    stt = SimSTT(t)
    app = create_app("assistant", cfg, stt_engine=stt, llm_factory=lambda: SimLLM(t), tts_engine=SimTTS(t),
                     desktop=DryRunDesktop())
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    return f"ws://127.0.0.1:{port}/ws", stt, srv


def _turn_ms(r: dict) -> list[int]:
    return [e["at_ms"] for e in r["events"] if e["type"] == "user_turn"]


async def run_clip(url: str, clip: dict, stt: SimSTT | None, tail_s: float) -> dict:
    from voice_agent.server.probe import probe

    if stt is not None:
        stt.set_clip(clip["text"], clip["speech_s"], clip["language"])
    r = await probe(url, clip["path"], tail_s=tail_s, gate_audio=True)
    turns = _turn_ms(r)
    end = r["speech_end_ms"]
    last = turns[-1] if turns else None
    # first audio of the reply to the last turn (the one that answers the whole sentence)
    first = next((e["at_ms"] for e in r["events"] if e["type"] == "audio_start" and last is not None
                  and e["at_ms"] >= last), None)
    done = [e for e in r["events"] if e["type"] == "assistant_done"]
    return {
        "clip": clip["name"], "language": clip["language"], "kind": clip["kind"],
        "turns": len(turns),
        "turn_ms": last - end if last is not None else None,      # end of speech -> turn decided
        "first_audio_ms": first - end if first is not None else None,
        "server_speech_end_to_audio_ms": done[-1].get("speech_end_to_audio_ms") if done else None,
        "replies": [e.get("text", "") for e in done],
    }


def pct(values: list[float], p: float) -> float | None:
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    return float(np.percentile(v, p))


def summarize(rows: list[dict]) -> dict:
    out = {}
    for lang in ("en", "ar", "all"):
        sel = [r for r in rows if r["kind"] == "complete" and (lang == "all" or r["language"] == lang)]
        vals = [r["first_audio_ms"] for r in sel]
        turn = [r["turn_ms"] for r in sel]
        out[lang] = {"n": len(sel), "p50": pct(vals, 50), "p90": pct(vals, 90),
                     "turn_p50": pct(turn, 50), "turn_p90": pct(turn, 90)}
    far = [r for r in rows if r["kind"] == "far"]
    pause = [r for r in rows if r["kind"] == "pause"]
    out["far_false_turns"] = f"{sum(r['turns'] > 0 for r in far)}/{len(far)}"
    out["pause_split_turns"] = f"{sum(r['turns'] > 1 for r in pause)}/{len(pause)}"
    out["complete_split_turns"] = f"{sum(r['turns'] > 1 for r in rows if r['kind'] == 'complete')}"
    return out


def print_report(rows: list[dict], summary: dict, label: str) -> None:
    print(f"\n{label}")
    print(f"{'clip':<13}{'lang':<5}{'kind':<10}{'turns':>6}{'turn ms':>9}{'1st audio ms':>14}")
    for r in rows:
        print(f"{r['clip']:<13}{r['language']:<5}{r['kind']:<10}{r['turns']:>6}"
              f"{r['turn_ms'] if r['turn_ms'] is not None else '-':>9}"
              f"{r['first_audio_ms'] if r['first_audio_ms'] is not None else '-':>14}")
    print("\ncomplete sentences: end of speech -> first audio (ms)")
    for lang in ("en", "ar", "all"):
        s = summary[lang]
        fmt = lambda x: "-" if x is None else f"{x:.0f}"  # noqa: E731
        print(f"  {lang:<4} n={s['n']:<3} p50 {fmt(s['p50']):>6}  p90 {fmt(s['p90']):>6}   "
              f"(turn decided: p50 {fmt(s['turn_p50'])}, p90 {fmt(s['turn_p90'])})")
    print(f"  far voices that started a turn: {summary['far_false_turns']}")
    print(f"  sentences with a pause that were split into 2+ turns: {summary['pause_split_turns']}")
    print(f"  complete sentences split into 2+ turns: {summary['complete_split_turns']}")


async def run_all(url: str, clips: list[dict], stt: SimSTT | None, tail_s: float) -> list[dict]:
    rows = []
    for c in clips:
        rows.append(await run_clip(url, c, stt, tail_s))
    return rows


def main(args) -> int:
    clips = load_clips(include_commands=args.url is None or args.include_commands)
    if args.only:
        clips = [c for c in clips if any(o in c["name"] for o in args.only.split(","))]
    cfg = load_config()
    if args.stop_secs is not None:
        cfg = replace(cfg, turn=replace(cfg.turn, stop_secs=args.stop_secs))
    srv = None
    if args.url:
        url, stt, label = args.url, None, f"REAL server at {args.url}"
    else:
        t = SimTimes()
        url, stt, srv = start_sim_server(cfg, t)
        label = (f"PIPELINE MEASURED, MODEL TIMES SIMULATED (turn.stop_secs={cfg.turn.stop_secs}, "
                 f"vad.min_volume={cfg.vad.min_volume}): {t}")
    rows = asyncio.run(run_all(url, clips, stt, args.tail_s))
    summary = summarize(rows)
    print_report(rows, summary, label)
    if args.json:
        Path(args.json).write_text(json.dumps({"label": label, "rows": rows, "summary": summary},
                                              ensure_ascii=False, indent=1), encoding="utf-8")
    if srv is not None:
        srv.should_exit = True
    return 0
