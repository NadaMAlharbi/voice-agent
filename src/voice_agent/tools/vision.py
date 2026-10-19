"""Screen understanding with the same local LLM (Qwen3.5 can read images), so
"what's on my screen" and "click the Submit button" need no extra model.

  describe(image, question, lang) -> short spoken answer in the user's language
  locate(image, target)           -> (x, y) screen pixels of a described element

Screenshots are made smaller (long side <= max_side) for a faster answer;
locate() scales the point back to screen pixels.
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

DESCRIBE_PROMPT = {
    "en": ("This is a screenshot of the user's computer screen. {q} Answer in English in at most three short "
           "spoken sentences, no markdown, no lists. Mention the app in front and the most important visible "
           "content."),
    "ar": ("هذه لقطة لشاشة حاسوب المستخدم. {q} أجب بالعربية الفصحى في ثلاث جمل قصيرة على الأكثر مناسبة للنطق، "
           "بدون تنسيق أو قوائم. اذكر البرنامج الظاهر في المقدمة وأهم المحتوى الظاهر."),
}

LOCATE_PROMPT = ("Find this element in the screenshot: \"{t}\". Reply with JSON only: "
                 "{{\"found\": true, \"x\": X, \"y\": Y}} where X and Y are the element's center as relative "
                 "coordinates from 0 to 1000 (0,0 = top-left, 1000,1000 = bottom-right). "
                 "If it is not visible reply {{\"found\": false}}.")


def _encode(path: Path, max_side: int) -> tuple[str, tuple[int, int], tuple[int, int]]:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    orig = img.size
    scale = min(1.0, max_side / max(orig))
    if scale < 1:
        img = img.resize((round(orig[0] * scale), round(orig[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode(), orig, img.size


def parse_point(text: str, sent: tuple[int, int], orig: tuple[int, int]) -> tuple[int, int] | None:
    """Model reply -> screen pixels. Accepts 0-1000 relative (asked) or pixels of the sent image."""
    m = re.search(r"\{.*?\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    if not d.get("found", True) or "x" not in d or "y" not in d:
        nums = re.findall(r"\d+(?:\.\d+)?", text) if not d else []
        if len(nums) < 2 or d.get("found") is False:
            return None
        d = {"x": nums[0], "y": nums[1]}
    try:
        x, y = float(d["x"]), float(d["y"])
    except (TypeError, ValueError):
        return None
    if x <= 1000 and y <= 1000:                       # relative 0..1000
        return round(x / 1000 * orig[0]), round(y / 1000 * orig[1])
    sx, sy = orig[0] / sent[0], orig[1] / sent[1]     # pixels of the downscaled image
    return round(x * sx), round(y * sy)


@dataclass
class OllamaVision:
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen3.5:9b"
    max_side: int = 1280
    keep_alive: int | str = -1
    timeout_s: float = 60.0
    num_ctx: int = 8192        # must equal llm.num_ctx, or Ollama reloads the model for each vision call

    def _chat(self, prompt: str, image_b64: str, max_tokens: int) -> str:
        body = {"model": self.model, "stream": False, "think": False, "keep_alive": self.keep_alive,
                "options": {"temperature": 0.2, "num_predict": max_tokens, "num_ctx": self.num_ctx},
                "messages": [{"role": "user", "content": prompt, "images": [image_b64]}]}
        r = httpx.post(f"{self.host}/api/chat", json=body, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()

    def describe(self, image: Path, question: str = "", lang: str = "en") -> str:
        b64, _, _ = _encode(image, self.max_side)
        q = question.strip() or ("What is on the screen?" if lang == "en" else "ماذا يظهر على الشاشة؟")
        return self._chat(DESCRIBE_PROMPT.get(lang, DESCRIBE_PROMPT["en"]).format(q=q), b64, 220)

    def locate(self, image: Path, target: str) -> tuple[int, int] | None:
        b64, orig, sent = _encode(image, self.max_side)
        return parse_point(self._chat(LOCATE_PROMPT.format(t=target), b64, 60), sent, orig)
