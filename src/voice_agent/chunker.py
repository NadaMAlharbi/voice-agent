"""Split a streaming LLM reply into short chunks for TTS.

Goal: start speaking as early as possible, then keep a natural rhythm.
  - First chunk: may end at a comma once it has `first_min_chars` characters, and at a
    sentence end only once it has `first_min_words` words ("Sure." alone is too short).
  - Word caps (`first_max_words`, `second_max_words`): a long first sentence is cut after
    a few words so speech can start sooner; the second chunk is capped so it is ready
    before the first one ends. 0 turns a cap off.
  - Later chunks end at sentence punctuation. Short sentences are merged up to
    `min_chars`; long ones may end at a comma after `soft_min_chars`.
  - No chunk is longer than `max_chars` (split at the last space).
  - "3.5", "e.g." and "Dr." are not split. Markdown symbols are removed.

  uv run chunk-test "Hello there, how are you? I'm fine. Thanks!"
  uv run chunk-test --llm "Tell me about Riyadh in three sentences."
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time

from voice_agent.config import Chunker as ChunkerCfg
from voice_agent.config import load_config

SENTENCE_END = set(".!?؟…؛\n")
SOFT_END = set(",،;:")
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "e.g", "i.e", "no", "jr", "sr"}
_MARKDOWN = re.compile(r"[*_`#>|]+")
_SPACES = re.compile(r"[ \t]+")
_WORD_AND_SPACE = re.compile(r"\S+\s+")

# Word lists for forced cuts. Each chunk is spoken as its own sentence, so a cut after
# "the" or "of" sounds broken, while a cut after a comma or before "and"/"في" sounds
# like a natural pause.
_BAD_END = set("""a an the of to in on at for with from by about as into than that which who whom whose and or
but so nor if because while is are was were be been am do does did have has had can could will would shall
should may might must not no my your his her its our their this these those i you he she it we they me him
us them how what when where why very more most some any each every i'm it's you're we're they're there
في من إلى الى على عن مع أن ان إن لأن لكن و ثم أو او التي الذي الذين هذا هذه ذلك تلك كل بعض حيث هو هي هم
قد لم لن لا ما""".split())
_BREAK_BEFORE = set("""and or but so because which who that when where while if than to of in on at for with
from by about into during after before until although though since as
في من إلى الى على عن مع لأن لكن ثم أو او التي الذي الذين حيث بينما""".split())
_PUNCT_END = tuple(",،;:.!?؟…؛")
_SENTENCE_END = tuple(".!?؟…؛")


def _bare(word: str) -> str:
    return word.strip("\"'()[]«»“”").rstrip(",،;:.!?؟…؛").lower()


def phrase_break(words: list[str], lo: int, hi: int, target: int | None = None) -> int:
    """Best number of words (lo..hi) to keep before a forced cut, close to `target`.
    Prefers cutting after punctuation or before a conjunction, never after a small word
    like "the"."""
    lo, hi = max(1, min(lo, hi)), max(1, hi)
    target = hi if target is None else target
    best, best_score = min(hi, target), -1e9
    for k in range(hi, lo - 1, -1):
        if k >= len(words):
            continue
        last, nxt = words[k - 1], words[k]
        score = -0.5 * (target - k) if k <= target else -0.75 * (k - target)
        if last.endswith(_PUNCT_END):
            score += 3
        b = _bare(nxt)
        if b in _BREAK_BEFORE or (len(b) > 3 and b[0] in "وف"):   # Arabic wa-/fa- (and, so) prefix
            score += 2
        if _bare(last) in _BAD_END:
            score -= 3
        if any(w.endswith(_SENTENCE_END) for w in words[k:k + 3]):
            score -= 2                                   # would orphan 1-3 words before a sentence end
        if score > best_score:
            best, best_score = k, score
    return best


def speakable(text: str, lang: str, final: bool) -> str:
    """Text as sent to TTS: it must end with a punctuation mark.

    TTS models (XTTS most of all) end a chunk without punctuation badly: the last word
    is cut or trails off. Mid-reply chunks get a comma (voice keeps going), the last
    chunk gets a full stop.
    """
    text = text.strip()
    if not text or text.endswith(_PUNCT_END) or text[-1] in "\"')]»”":
        return text
    if final:
        return text + "."
    return text + ("،" if lang == "ar" else ",")


def clean(text: str) -> str:
    return _MARKDOWN.sub("", text)


class SpeechChunker:
    def __init__(self, cfg: ChunkerCfg | None = None, language: str = "en"):
        self.cfg = cfg or ChunkerCfg()
        # Arabic words take longer to say, so Arabic uses smaller word caps for the same audio length.
        ar = language == "ar"
        self._first_cap = (self.cfg.first_max_words_ar if ar and self.cfg.first_max_words_ar else
                           self.cfg.first_max_words)
        self._second_cap = (self.cfg.second_max_words_ar if ar and self.cfg.second_max_words_ar else
                            self.cfg.second_max_words)
        self._max_chars = self.cfg.max_chars_ar if ar and self.cfg.max_chars_ar else self.cfg.max_chars
        self._first_min_chars = (self.cfg.first_min_chars_ar if ar and self.cfg.first_min_chars_ar else
                                 self.cfg.first_min_chars)
        self._min_chars = self.cfg.min_chars_ar if ar and self.cfg.min_chars_ar else self.cfg.min_chars
        self.buf = ""
        self.emitted = 0
        self._first_words = 0

    def push(self, delta: str) -> list[str]:
        """Add streamed text; return chunks that are ready to speak."""
        self.buf += clean(delta)
        out = []
        while (chunk := self._next()) is not None:
            out.append(chunk)
        return out

    def flush(self) -> list[str]:
        """End of reply: whatever is left."""
        rest, self.buf = self._tidy(self.buf), ""
        if rest:
            self.emitted += 1
            return [rest]
        return []

    # --------------------------------------------------------------- internals

    def _next(self) -> str | None:
        first = self.emitted == 0
        min_len = 1 if first else self._min_chars
        min_words = self.cfg.first_min_words if first else 0
        cut = None
        for i, c in enumerate(self.buf):
            # Look at the next character to know it is a real boundary ("3.5", "...").
            if i + 1 >= len(self.buf) or not (self.buf[i + 1].isspace() or c == "\n"):
                continue
            if (c in SENTENCE_END and not self._is_abbrev(i) and len(self.buf[: i + 1].strip()) >= min_len
                    and len(self.buf[: i + 1].split()) >= min_words):
                cut = i + 1
                break
            soft_min = self._first_min_chars if first else self.cfg.soft_min_chars
            if c in SOFT_END and len(self.buf[: i + 1].strip()) >= soft_min:
                cut = i + 1
                break
        cap = 0
        if self.emitted == 0:
            cap = self._first_cap
        elif self.emitted == 1 and self._second_cap:
            # chunk 2 is made while chunk 1 plays: keep it at most about 2x chunk 1
            cap = min(self._second_cap, max(self._first_cap, 2 * self._first_words))
        if cut is None and cap:
            words = list(_WORD_AND_SPACE.finditer(self.buf))  # complete words only (followed by a space)
            # first chunk: cut within cap-1..cap+2; later chunks: cap/2..cap.
            # One extra word is needed to judge the boundary.
            lo, hi = (max(self.cfg.first_min_words, cap - 1), cap + 2) if first else (max(3, cap // 2), cap)

            if len(words) > hi:
                k = phrase_break([w.group().strip() for w in words], lo, hi, target=cap)
                cut = words[k - 1].end()
        if cut is None and len(self.buf) > self._max_chars:
            cut = self._long_cut()
        if cut is None:
            return None
        chunk, self.buf = self._tidy(self.buf[:cut]), self.buf[cut:].lstrip()
        if not chunk:
            return self._next() if self.buf else None
        if self.emitted == 0:
            self._first_words = len(chunk.split())
        self.emitted += 1
        return chunk

    def _long_cut(self) -> int:
        """Cut a too-long sentence at a natural pause within max chars (comma, before "and")."""
        head = self.buf[: self._max_chars]
        words = list(_WORD_AND_SPACE.finditer(head))
        if len(words) < 4:
            space = head.rfind(" ")
            return space if space > 0 else self._max_chars
        k = phrase_break([w.group().strip() for w in words], max(3, len(words) // 2), len(words) - 1)
        return words[k - 1].end()

    def _is_abbrev(self, i: int) -> bool:
        if self.buf[i] != ".":
            return False
        word = self.buf[:i].split()[-1:] or [""]
        return word[0].lower().rstrip(".") in ABBREVIATIONS

    @staticmethod
    def _tidy(s: str) -> str:
        return _SPACES.sub(" ", s.replace("\n", " ")).strip()


def chunk_text(text: str, cfg: ChunkerCfg | None = None, step: int = 3, language: str = "en") -> list[str]:
    """Feed text in small deltas like a token stream (for chunk-test)."""
    ch = SpeechChunker(cfg, language)
    out: list[str] = []
    for i in range(0, len(text), step):
        out += ch.push(text[i : i + step])
    return out + ch.flush()


async def _llm_demo(prompt: str) -> None:
    from voice_agent.llm import OllamaClient

    cfg = load_config()
    llm = OllamaClient(cfg.llm)
    ch = SpeechChunker(cfg.chunker)
    t0 = time.perf_counter()
    try:
        async for d in llm.stream_chat(llm.messages(prompt)):
            for c in ch.push(d):
                print(f"  {(time.perf_counter() - t0) * 1000:6.0f} ms  [{ch.emitted}] {c}")
        for c in ch.flush():
            print(f"  {(time.perf_counter() - t0) * 1000:6.0f} ms  [{ch.emitted}] {c}  (flush)")
        print(f"LLM TTFT {llm.stats.ttft_ms:.0f} ms; first chunk time above = when TTS could start")
    finally:
        await llm.aclose()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Show how text is split for TTS")
    ap.add_argument("text", nargs="?")
    ap.add_argument("--llm", metavar="PROMPT", help="stream a real Ollama reply through the chunker")
    args = ap.parse_args(argv)
    if args.llm:
        asyncio.run(_llm_demo(args.llm))
        return 0
    text = args.text or "مرحباً، كيف حالك اليوم؟ الطقس جميل جداً. Hello, the price is 3.5 dollars! Dr. Smith agrees."
    for i, c in enumerate(chunk_text(text, load_config().chunker), 1):
        print(f"[{i}] {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
