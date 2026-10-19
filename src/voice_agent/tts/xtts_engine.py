"""Arabic TTS: XTTS-v2 (GPU). Built-in speakers, female and male. Streams audio while generating.

License: Coqui Public Model License (non-commercial use).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import numpy as np

from voice_agent.models import load_registry, model_path
from voice_agent.tts.base import Voice


MAX_CHARS = 160      # XTTS warns and may cut the audio above 166 characters for Arabic


def split_for_xtts(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Split text longer than `limit` at spaces (punctuation first). Short text stays one piece."""
    parts, rest = [], text.strip()
    while len(rest) > limit:
        head = rest[:limit]
        cut = max(head.rfind(p) for p in ("،", ",", "؛", ";", ":"))
        cut = cut + 1 if cut > limit // 3 else head.rfind(" ")
        if cut <= 0:
            cut = limit
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    return parts + ([rest] if rest else [])


class XttsEngine:
    name = "xtts"
    sample_rate = 24000
    streaming = True     # audio comes in pieces while the model is still working

    def __init__(self, cfg):
        self.cfg = cfg
        x = cfg.tts.xtts
        self.chunk_tokens = int(x.get("stream_chunk_tokens", 20))
        self.temperature = float(x.get("temperature", 0.65))
        self.speed = float(x.get("speed", 1.0))
        self.device = "cuda" if cfg.device == "cuda" else "cpu"
        self._cancel = threading.Event()
        self.model = None
        self.speakers: dict = {}
        self.last_stats: dict = {}

    def load(self, voices: dict[str, Voice]) -> None:
        # This function loads XTTS and the built-in speakers used by the voices.
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        spec = load_registry()[self.cfg.tts.xtts.get("model", "xtts-v2")]
        folder = model_path(spec, self.cfg.paths.models_dir)
        config = XttsConfig()
        config.load_json(str(folder / "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=str(folder), eval=True)
        self.model.to(self.device)
        known = self.model.speaker_manager.speakers
        for v in voices.values():
            if v.voice_id not in known:
                raise ValueError(f"XTTS has no speaker {v.voice_id!r}. Available: {sorted(known)}")
            latent, embedding = known[v.voice_id].values()
            self.speakers[v.voice_id] = (latent, embedding)
        for v in voices.values():               # warm-up: the first call is slower
            list(self.synthesize("مرحبا.", v))

    def device_info(self) -> str:
        return f"xtts-v2 ({self.device})"

    def cancel(self) -> None:
        self._cancel.set()

    def reset_cancel(self) -> None:
        self._cancel.clear()

    def synthesize(self, text: str, voice: Voice) -> Iterator[np.ndarray]:
        # This function streams one chunk: audio pieces are sent while the model is still generating.
        import torch

        latent, embedding = self.speakers[voice.voice_id]
        t0 = time.perf_counter()
        first_ms, total = None, 0
        with torch.inference_mode():
            for part in split_for_xtts(text):
                stream = self.model.inference_stream(
                    part, voice.language, latent, embedding, stream_chunk_size=self.chunk_tokens,
                    temperature=self.temperature, speed=self.speed, enable_text_splitting=False)
                for piece in stream:
                    if self._cancel.is_set():       # barge-in: stop generating
                        return
                    audio = piece.float().cpu().numpy().astype(np.float32)
                    if first_ms is None:
                        first_ms = round((time.perf_counter() - t0) * 1000)
                    total += len(audio)
                    yield audio
        self.last_stats = {"first_piece_ms": first_ms, "total_ms": round((time.perf_counter() - t0) * 1000),
                           "audio_s": round(total / self.sample_rate, 2), "voice": voice.voice_id}
