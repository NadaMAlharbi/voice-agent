"""Streaming bilingual LLM via a local Ollama server."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from voice_agent.config import Llm, load_config


@dataclass
class LLMStats:
    ttft_ms: float = 0.0              # request -> first text token
    total_ms: float = 0.0
    tokens: int = 0
    tokens_per_s: float = 0.0       
    text: str = ""
    tool_calls: int = 0
    prompt_tokens: int = 0            # prompt tokens Ollama had to process (not cached)
    prompt_ms: float = 0.0            # time spent on them
    load_ms: float = 0.0              # model load time (0 when already in VRAM)


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str | None = None

    @classmethod
    def from_ollama(cls, call: dict) -> "ToolCall":
        fn = call.get("function", call)
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        return cls(fn.get("name", ""), args if isinstance(args, dict) else {}, call.get("id"))

    def to_message(self) -> dict:
        d = {"function": {"name": self.name, "arguments": self.arguments}}
        if self.id:
            d["id"] = self.id
        return d


class _RawToolCallFilter:
    """Pulls <tool_call>{json}</tool_call> blocks out of streamed text."""

    OPEN, CLOSE = "<tool_call>", "</tool_call>"

    def __init__(self):
        self.buf = ""

    def feed(self, delta: str) -> tuple[str, list[ToolCall]]:
        self.buf += delta
        out, calls = "", []
        while True:
            i = self.buf.find(self.OPEN)
            if i < 0:
                # keep a possible partial "<tool_c" at the end
                keep = next((k for k in range(min(len(self.OPEN) - 1, len(self.buf)), 0, -1)
                             if self.OPEN.startswith(self.buf[-k:])), 0)
                out += self.buf[:len(self.buf) - keep]
                self.buf = self.buf[len(self.buf) - keep:]
                return out, calls
            out += self.buf[:i]
            j = self.buf.find(self.CLOSE, i)
            if j < 0:
                self.buf = self.buf[i:]
                return out, calls
            body = self.buf[i + len(self.OPEN):j].strip()
            self.buf = self.buf[j + len(self.CLOSE):]
            try:
                d = json.loads(body)
                calls.append(ToolCall(d.get("name", ""), d.get("arguments") or {}))
            except (json.JSONDecodeError, AttributeError):
                pass

    def flush(self) -> str:
        rest, self.buf = ("" if self.buf.startswith(self.OPEN) else self.buf), ""
        return rest


def options(cfg: Llm) -> dict:
    """Ollama sampling options. num_ctx must stay the same for every request (a change reloads the model)."""
    opts = {"temperature": cfg.temperature, "top_p": cfg.top_p, "top_k": cfg.top_k,
            "num_predict": cfg.max_tokens, "num_ctx": cfg.num_ctx}
    if cfg.presence_penalty:
        opts["presence_penalty"] = cfg.presence_penalty
    return opts


@dataclass
class OllamaClient:
    cfg: Llm
    client: httpx.AsyncClient | None = None
    stats: LLMStats = field(default_factory=LLMStats)

    def _http(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(base_url=self.cfg.host, timeout=httpx.Timeout(120, connect=5))
        return self.client

    def messages(self, user_text: str, history: list[dict] | None = None) -> list[dict]:
        msgs = [{"role": "system", "content": self.cfg.system_prompt}] if self.cfg.system_prompt else []
        return msgs + (history or []) + [{"role": "user", "content": user_text}]

    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        """Yield text deltas as they arrive. Stats land in self.stats when done."""
        async for ev in self.stream_events(messages):
            if isinstance(ev, str):
                yield ev

    async def stream_events(self, messages: list[dict], tools: list[dict] | None = None) -> AsyncIterator:
        """Yield text pieces (str) and ToolCall objects.

        If the model writes a raw <tool_call>{...}</tool_call> block as text instead of
        a real tool call, it is parsed here and never spoken.
        """
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": True,
            "think": self.cfg.think,
            "keep_alive": self.cfg.keep_alive,
            "options": options(self.cfg),
        }
        if tools:
            body["tools"] = tools
        self.stats = st = LLMStats()
        raw = _RawToolCallFilter()
        t0 = time.perf_counter()
        async with self._http().stream("POST", "/api/chat", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Ollama HTTP {r.status_code}: {(await r.aread()).decode(errors='ignore')}")
            async for line in r.aiter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if "error" in msg:
                    raise RuntimeError(f"Ollama: {msg['error']}")
                m = msg.get("message", {})
                for call in m.get("tool_calls") or []:
                    st.tool_calls += 1
                    yield ToolCall.from_ollama(call)
                delta = m.get("content", "")
                if delta:
                    if st.ttft_ms == 0.0:
                        st.ttft_ms = (time.perf_counter() - t0) * 1000
                    text, calls = raw.feed(delta)
                    for c in calls:
                        st.tool_calls += 1
                        yield c
                    if text:
                        st.text += text
                        yield text
                if msg.get("done"):
                    st.tokens = msg.get("eval_count", 0)
                    dur = msg.get("eval_duration", 0) / 1e9
                    st.tokens_per_s = st.tokens / dur if dur else 0.0
                    st.prompt_tokens = msg.get("prompt_eval_count", 0)
                    st.prompt_ms = msg.get("prompt_eval_duration", 0) / 1e6
                    st.load_ms = msg.get("load_duration", 0) / 1e6
        tail = raw.flush()
        if tail:
            st.text += tail
            yield tail
        st.total_ms = (time.perf_counter() - t0) * 1000

    async def gpu_share(self) -> tuple[str, float] | None:
        """(model, fraction of weights in VRAM) for the loaded model, from /api/ps."""
        r = await self._http().get("/api/ps")
        for m in r.json().get("models", []):
            if m.get("name", "").startswith(self.cfg.model.split(":")[0]):
                size = m.get("size") or 1
                return m["name"], m.get("size_vram", 0) / size
        return None

    async def warmup(self, messages: list[dict] | None = None, tools: list[dict] | None = None) -> float:
        """Load the model into GPU memory (can take seconds). With the real system prompt
        and tools it also fills Ollama's prompt cache, so the first turn is fast too."""
        t0 = time.perf_counter()
        async for _ in self.stream_events(messages or self.messages("hi"), tools):
            pass
        return (time.perf_counter() - t0) * 1000

    async def aclose(self) -> None:
        if self.client:
            await self.client.aclose()


from voice_agent.language import LanguageGuard, arabic_ratio, directive, text_language  # noqa: E402, F401

DEFAULT_PROMPTS = [
    "What is the capital of Japan, and what is it famous for?",
    "ما هي عاصمة اليابان وبماذا تشتهر؟",
    "Give me one tip to sleep better.",
    "أعطني نصيحة واحدة لنوم أفضل.",
]

# Questions that need live data or the clock: must not be answered with invented values.
REALTIME_CHECK_PROMPTS = [
    "What is the weather in Riyadh?",
    "كيف الطقس في الرياض اليوم؟",
    "What are today's top news headlines?",
    "ما سعر الذهب اليوم؟",
    "What time is it?",
    "ما هو تاريخ اليوم؟",
    "Explain how rain forms.",             # general: must be answered normally
]

# Prompts that tend to pull Qwen into Chinese or another language.
LANG_CHECK_PROMPTS = DEFAULT_PROMPTS + [
    "How do you say thank you in Chinese, Japanese and Korean?",
    "كيف أقول شكراً باللغة الصينية واليابانية؟",
    "Tell me about the Great Wall of China and Beijing.",
    "حدثني عن سور الصين العظيم.",
    "Translate 'good morning' into five languages.",
    "ok",
    "نعم",
]


def turn_messages(llm: "OllamaClient", user_text: str, history: list[dict] | None = None):
    """Exactly what the voice pipeline sends: system prompt + tool rules, the per-turn
    context in the user message, and the tool list. Returns (language, messages, tools)."""
    from datetime import datetime

    from voice_agent.config import load_config
    from voice_agent.knowledge import build_messages, classify, turn_note
    from voice_agent.server.pipelines import make_agent
    from voice_agent.tools.desktop import DryRunDesktop

    agent = make_agent(load_config(), DryRunDesktop())
    lang = text_language(user_text)
    base = llm.cfg.system_prompt + (f"\n\n{agent.system_note()}" if agent else "")
    msgs = build_messages(base, history or [], user_text, datetime.now(), turn_note(classify(user_text), None),
                          directive(lang), llm.cfg.cache_friendly)
    return lang, msgs, (agent.schemas() if agent else None)


_INVENTED = re.compile(r"\d+\s*(?:°|degrees|درجة|درجات)|\d+(?:\.\d+)?\s*(?:dollars?|riyals?|ريال|دولار|\$)")


async def _run(prompts: list[str]) -> int:
    cfg = load_config()
    llm = OllamaClient(cfg.llm)
    try:
        print(f"model {cfg.llm.model} @ {cfg.llm.host} | options {options(cfg.llm)}")
        w = await llm.warmup()
        share = await llm.gpu_share()
        if share:
            name, frac = share
            flag = "OK" if frac > 0.99 else "WARN: partly on CPU, will be slow"
            print(f"warmup {w:.0f} ms | {name}: {frac:.0%} in VRAM ({flag})")
        fails = drift_total = 0
        from voice_agent.knowledge import classify

        for p in prompts:
            want, msgs, tools = turn_messages(llm, p)
            guard = LanguageGuard()
            print(f"\n> {p}\n< ", end="", flush=True)
            calls = []
            async for ev in llm.stream_events(msgs, tools):
                if isinstance(ev, str):
                    print(guard.feed(ev), end="", flush=True)
                else:
                    calls.append(f"{ev.name}({json.dumps(ev.arguments, ensure_ascii=False)})")
            print(guard.flush(), end="")
            s = llm.stats
            got = text_language(s.text, default=want) if s.text.strip() else want
            lang_ok = got == want
            fails += not lang_ok
            drift_total += guard.removed
            notes = f" | tool calls: {', '.join(calls)}" if calls else ""
            if guard.removed:
                notes += f" | FOREIGN SCRIPT removed: {guard.removed} chars"
            if classify(p).live and _INVENTED.search(s.text):
                notes += " | WARN: contains a number that looks like a live value"
            words = len(s.text.split())
            if words > 45:
                notes += f" | WARN: long reply ({words} words)"
            print(f"\n  TTFT {s.ttft_ms:.0f} ms | {s.tokens_per_s:.0f} tok/s | {s.tokens} tokens, {words} words | "
                  f"prompt {s.prompt_tokens} tokens in {s.prompt_ms:.0f} ms | "
                  f"want {want}, got {got}: {'OK' if lang_ok else 'MISMATCH'}{notes}")
        print(f"\nRESULT: {len(prompts) - fails}/{len(prompts)} replies in the right language, "
              f"{drift_total} foreign-script chars removed by the guard")
        return 1 if fails else 0
    except httpx.ConnectError:
        print(f"Cannot reach Ollama at {cfg.llm.host}. Is it running? (`ollama serve`)")
        return 1
    finally:
        await llm.aclose()


async def _cache_check(turns: int = 4) -> int:
    """Does Ollama reuse the cached system prompt + tools? Later turns should process
    only the new tokens (tens), not the whole prompt (thousands)."""
    cfg = load_config()
    llm = OllamaClient(cfg.llm)
    history: list[dict] = []
    questions = ["What is the capital of Japan?", "ما هي أطول سلسلة جبال في العالم؟",
                 "Give me one tip to sleep better.", "كم عدد أيام السنة؟"]
    try:
        rows = []
        for i in range(turns):
            q = questions[i % len(questions)]
            _, msgs, tools = turn_messages(llm, q, history)
            text = ""
            async for ev in llm.stream_events(msgs, tools):
                if isinstance(ev, str):
                    text += ev
            s = llm.stats
            rows.append(s.prompt_tokens)
            print(f"turn {i + 1}: prompt {s.prompt_tokens:5d} tokens processed in {s.prompt_ms:6.0f} ms | "
                  f"TTFT {s.ttft_ms:5.0f} ms | load {s.load_ms:.0f} ms")
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": text}]
        ok = len(rows) > 1 and max(rows[1:]) < 0.5 * rows[0]
        print("RESULT:", "prompt cache works" if ok else
              "WARN: later turns process most of the prompt again (cache not reused)")
        return 0 if ok else 1
    except httpx.ConnectError:
        print(f"Cannot reach Ollama at {cfg.llm.host}. Is it running? (`ollama serve`)")
        return 1
    finally:
        await llm.aclose()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stream replies from the local LLM")
    ap.add_argument("prompts", nargs="*")
    ap.add_argument("--lang-check", action="store_true", help="run prompts that provoke other languages")
    ap.add_argument("--realtime-check", action="store_true", help="weather/news/prices/time questions")
    ap.add_argument("--cache-check", action="store_true", help="is the system prompt + tools cache reused?")
    args = ap.parse_args(argv)
    if args.cache_check:
        return asyncio.run(_cache_check())
    prompts = args.prompts or (LANG_CHECK_PROMPTS if args.lang_check else
                               REALTIME_CHECK_PROMPTS if args.realtime_check else DEFAULT_PROMPTS)
    return asyncio.run(_run(prompts))


if __name__ == "__main__":
    sys.exit(main())
