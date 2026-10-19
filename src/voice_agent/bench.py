"""Response latency benchmark: LLM token stream -> chunker -> TTS -> playback schedule.

Uses the same rules as the server (tts_budget.plan_chunk): the LLM and TTS run at the same
time and audio plays in real time.

  uv run latency-bench           # simulated LLM + simulated TTS speed (no GPU, no models)
  uv run latency-bench --real    # real Ollama + real Kokoro/XTTS on your machine
  uv run latency-bench --e2e     # whole pipeline: end of speech -> first audio (see e2e_bench.py)

Per reply: first audio (ms after the LLM request), first-chunk words, and silence gaps.
A per-chunk timeline is printed for replies with a gap (or for all with --timeline).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from voice_agent.chunker import SpeechChunker
from voice_agent.config import Chunker as ChunkerCfg
from voice_agent.config import load_config
from voice_agent.language import text_language

REPLIES = [
    "Riyadh is the capital of Saudi Arabia and it is also the largest city in the country, with millions of "
    "people living there. It is known for its modern skyline and its long history.",
    "الرياض هي عاصمة المملكة العربية السعودية وأكبر مدنها من حيث عدد السكان، وهي مدينة حديثة تتميز بناطحات "
    "السحاب والأسواق التاريخية. يزورها الكثير من الناس سنوياً.",
    "Sure. To sleep better, keep a regular schedule and avoid screens for an hour before bed.",
    "Yes, I can help with that. Open the settings, choose your account, and then turn on two step verification.",
]

PROMPTS = ["Tell me about Riyadh in two sentences.", "حدثني عن الرياض في جملتين.",
           "Give me one tip to sleep better."]


@dataclass
class Result:
    first_audio_ms: float
    first_words: int
    first_synth_ms: float
    gap_ms: float
    chunks: int


class CostModelTTS:
    """Simulated TTS: time = fixed + RTF x audio length. piece_s > 0 streams pieces like XTTS.
    The defaults are rough guesses, not measurements."""
    name, sample_rate = "cost-model", 24000

    def __init__(self, base_ms: float = 150, rtf: float = 0.3, s_per_word: float = 0.38, piece_s: float = 0.0):
        self.base_ms, self.rtf, self.s_per_word, self.piece_s = base_ms, rtf, s_per_word, piece_s

    def synthesize(self, text, voice):
        audio_s = max(0.4, len(text.split()) * self.s_per_word)
        time.sleep(self.base_ms / 1000)
        step = self.piece_s or audio_s
        done = 0.0
        while done < audio_s - 1e-6:
            d = min(step, audio_s - done)
            time.sleep(self.rtf * d)
            done += d
            yield np.zeros(int(d * self.sample_rate), np.float32)


async def replay(deltas: list[tuple[float, str]], tts, voice, chunker_cfg: ChunkerCfg, lang: str = "en",
                 cost=None, timeline: list | None = None) -> Result:
    """deltas = [(seconds since request, text)]. Same scheduling as the server.
    timeline: per-chunk rows are added to this list."""
    from voice_agent.server.tts_budget import TTSCostModel, plan_chunk

    cost = cost or TTSCostModel()
    t0 = time.perf_counter()
    queue: asyncio.Queue = asyncio.Queue()
    lock = threading.Lock()

    async def produce():
        ch = SpeechChunker(chunker_cfg, lang)
        for at, d in deltas:
            await asyncio.sleep(max(0.0, t0 + at - time.perf_counter()))
            for c in ch.push(d):
                await queue.put(c)
        for c in ch.flush():
            await queue.put(c)
        await queue.put(None)

    def synth(text, out_q, loop):
        # This function runs the blocking TTS in a thread and sends each audio piece back.
        with lock:
            for a in tts.synthesize(text, voice):
                loop.call_soon_threadsafe(out_q.put_nowait, (time.perf_counter(), a))
        loop.call_soon_threadsafe(out_q.put_nowait, None)

    prod = asyncio.create_task(produce())
    first_audio = first_synth = None
    first_words, n, gap, play_end = 0, 0, 0.0, 0.0
    carry = None
    loop = asyncio.get_running_loop()
    while True:
        chunk = carry if carry is not None else await queue.get()
        carry = None
        if chunk is None:
            break
        ahead = max(0.0, play_end - time.perf_counter()) if n else 0.0
        plan = plan_chunk(chunk, lang, cost, ahead)
        chunk, carry, ahead = plan.text, plan.carry, plan.ahead_s
        t_req = time.perf_counter()
        out_q: asyncio.Queue = asyncio.Queue()
        th = asyncio.create_task(asyncio.to_thread(synth, chunk, out_q, loop))
        dur, chunk_gap = 0.0, 0.0
        while (item := await out_q.get()) is not None:
            at, a = item
            d = len(a) / tts.sample_rate
            if first_audio is None:
                first_audio, first_synth, first_words = at - t0, at - t_req, len(chunk.split())
            elif at > play_end:
                chunk_gap += at - play_end          # silence: this piece came after the audio ran out
            play_end = max(at, play_end) + d
            dur += d
        gap += chunk_gap
        await th
        synth_s = time.perf_counter() - t_req
        if timeline is not None:
            timeline.append({"chunk": n + 1, "words": len(chunk.split()), "queued_ms": round(ahead * 1000),
                             "synth_ms": round(synth_s * 1000), "audio_ms": round(dur * 1000),
                             "gap_ms": round(chunk_gap * 1000), "text": chunk})
        cost.observe(lang, len(chunk.split()), dur, synth_s)
        n += 1
    await prod
    return Result(first_audio * 1000, first_words, first_synth * 1000, gap * 1000, n)


def simulated_stream(text: str, tok_s: float = 80.0, ttft_s: float = 0.15) -> list[tuple[float, str]]:
    words = text.split(" ")
    return [(ttft_s + i / tok_s * 1.3, (" " if i else "") + w) for i, w in enumerate(words)]  # ~1.3 tok/word


async def real_stream(llm, prompt: str) -> list[tuple[float, str]]:
    from voice_agent.llm import turn_messages

    _, msgs, _tools = turn_messages(llm, prompt)
    out, t0 = [], time.perf_counter()
    async for d in llm.stream_chat(msgs):
        out.append((time.perf_counter() - t0, d))
    return out


async def _run(args) -> int:
    from voice_agent.server.tts_budget import TTSCostModel

    cfg = load_config()
    if args.real:
        from voice_agent.llm import OllamaClient
        from voice_agent.tts.base import load_voices, make_tts, pick_voice

        voices = load_voices(cfg)
        tts = make_tts(cfg)
        tts.load(voices)
        llm = OllamaClient(cfg.llm)
        await llm.warmup()
        streams = [await real_stream(llm, p) for p in PROMPTS]
        await llm.aclose()
        pick = lambda text: pick_voice(voices, text_language(text), args.gender)  # noqa: E731
        print(f"real: {cfg.llm.model} + {tts.name} ({tts.device_info()})")
    else:
        tts = CostModelTTS(args.base_ms, args.rtf, piece_s=args.piece_s)
        streams = [simulated_stream(t) for t in REPLIES]
        pick = lambda text: None  # noqa: E731
        print(f"SIMULATED (not your machine): LLM ~80 tok/s, TTS {args.base_ms:.0f} ms + RTF {args.rtf}")

    cost = TTSCostModel()                     # kept across replies, like one server session
    print(f"{'reply':<7}{'first audio':>12}{'1st words':>10}{'1st sound':>10}{'gaps':>9}{'chunks':>7}")
    firsts, gaps, timelines = [], [], []
    for rnd in range(args.rounds):
        for i, deltas in enumerate(streams):
            text = "".join(d for _, d in deltas)
            tl: list = []
            r = await replay(deltas, tts, pick(text), cfg.chunker, lang=text_language(text), cost=cost, timeline=tl)
            firsts.append(r.first_audio_ms)
            gaps.append(r.gap_ms)
            if args.timeline or r.gap_ms > 0:
                timelines.append((rnd + 1, i + 1, r.gap_ms, tl))
            print(f"{i + 1:<7}{r.first_audio_ms:>9.0f} ms{r.first_words:>10}{r.first_synth_ms:>7.0f} ms"
                  f"{r.gap_ms:>6.0f} ms{r.chunks:>7}")
    v = sorted(firsts)
    print(f"\nfirst audio: mean {sum(v) / len(v):.0f} ms, worst {v[-1]:.0f} ms | gaps total {sum(gaps):.0f} ms, "
          f"replies with a gap {sum(g > 0 for g in gaps)}/{len(gaps)}")
    for rnd, i, g, tl in timelines:
        print(f"\nround {rnd} reply {i}: gaps {g:.0f} ms (queued = audio still playing when synthesis started)")
        for row in tl:
            print(f"  #{row['chunk']} {row['words']:>2} w  queued {row['queued_ms']:>5} ms  synth {row['synth_ms']:>5} ms"
                  f"  audio {row['audio_ms']:>5} ms  gap {row['gap_ms']:>4} ms  | {row['text']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Response latency benchmark (first audio, gaps)")
    ap.add_argument("--real", action="store_true", help="use real Ollama + TTS models")
    ap.add_argument("--gender", default="female", choices=["female", "male"], help="--real: voice gender")
    ap.add_argument("--base-ms", type=float, default=150.0, help="sim: TTS start-up ms per chunk")
    ap.add_argument("--rtf", type=float, default=0.3, help="sim: TTS seconds per audio second")
    ap.add_argument("--piece-s", type=float, default=0.0, help="sim: stream pieces of this length (0 = whole chunk)")
    ap.add_argument("--rounds", type=int, default=1, help="repeat all replies N times")
    ap.add_argument("--timeline", action="store_true", help="per-chunk timeline of every reply")
    ap.add_argument("--e2e", action="store_true", help="end of speech -> first audio, through the pipeline")
    ap.add_argument("--url", default=None, help="--e2e: a running server (real models)")
    ap.add_argument("--include-commands", action="store_true", help="--e2e --url: also play command clips")
    ap.add_argument("--only", default="", help="--e2e: clip names to run, comma separated")
    ap.add_argument("--stop-secs", type=float, default=None, help="--e2e: override turn.stop_secs")
    ap.add_argument("--tail-s", type=float, default=4.0, help="--e2e: silence after each clip")
    ap.add_argument("--json", default=None, help="--e2e: save results to this file")
    args = ap.parse_args(argv)
    if args.e2e:
        from voice_agent import e2e_bench

        return e2e_bench.main(args)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
