"""English TTS: Kokoro-82M (ONNX). Built-in American English voices, female and male."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import numpy as np

from voice_agent.models import load_registry, model_path
from voice_agent.tts.base import Voice

LANG_CODE = "en-us"      # American English pronunciation


def make_session(model_file, device: str):
    """ONNX Runtime session on the GPU (CUDA) when device is cuda, else CPU.

    We ask for CUDA ourselves. Left alone, kokoro-onnx also tries TensorRT first,
    and it silently runs on the CPU when the CUDA libraries are not found.
    Set ONNX_PROVIDER to force a provider.
    """
    import os

    import onnxruntime as rt

    providers = ["CPUExecutionProvider"]
    if os.getenv("ONNX_PROVIDER"):
        providers = [os.environ["ONNX_PROVIDER"]] + providers
    elif device == "cuda" and "CUDAExecutionProvider" in rt.get_available_providers():
        from voice_agent import cuda_dlls

        cuda_dlls.prepare()          # Windows: use the CUDA DLLs that come with torch
        providers = ["CUDAExecutionProvider"] + providers
    return rt.InferenceSession(str(model_file), providers=providers)


class KokoroEngine:
    name = "kokoro"
    sample_rate = 24000

    def __init__(self, cfg):
        self.cfg = cfg
        self.speed = float(cfg.tts.kokoro.get("speed", 1.0))
        self._cancel = threading.Event()
        self.model = None
        self.last_stats: dict = {}

    def load(self, voices: dict[str, Voice]) -> None:
        # This function loads the model and checks that every configured voice exists.
        from kokoro_onnx import Kokoro

        spec = load_registry()[self.cfg.tts.kokoro.get("model", "kokoro")]
        folder = model_path(spec, self.cfg.paths.models_dir)
        session = make_session(folder / "kokoro-v1.0.onnx", self.cfg.device)
        self.model = Kokoro.from_session(session, str(folder / "voices-v1.0.bin"))
        known = set(self.model.get_voices())
        missing = [v.voice_id for v in voices.values() if v.voice_id not in known]
        if missing:
            raise ValueError(f"Kokoro has no voice {missing}. Available: {sorted(known)}")
        for v in voices.values():               # warm-up: the first call is slower
            self.model.create("Hello.", voice=v.voice_id, speed=self.speed, lang=LANG_CODE)

    def device_info(self) -> str:
        # This function reports where the model really runs. kokoro-onnx picks the ONNX Runtime
        # provider itself (set the ONNX_PROVIDER environment variable to force one).
        sess = getattr(self.model, "sess", None)
        providers = sess.get_providers() if sess is not None else []
        device = providers[0].replace("ExecutionProvider", "").lower() if providers else "not loaded"
        note = " WARNING: on the CPU, run `uv run check-env`" if self.cfg.device == "cuda" and device == "cpu" else ""
        return f"kokoro-82M onnx ({device}){note}"

    def cancel(self) -> None:
        self._cancel.set()

    def reset_cancel(self) -> None:
        self._cancel.clear()

    def synthesize(self, text: str, voice: Voice) -> Iterator[np.ndarray]:
        # This function speaks one chunk. Kokoro is fast, so the whole chunk is made at once.
        if self._cancel.is_set():
            return
        t0 = time.perf_counter()
        audio, sr = self.model.create(text, voice=voice.voice_id, speed=self.speed, lang=LANG_CODE)
        audio = np.asarray(audio, dtype=np.float32)
        self.last_stats = {"total_ms": round((time.perf_counter() - t0) * 1000),
                           "audio_s": round(len(audio) / sr, 2), "voice": voice.voice_id}
        if not self._cancel.is_set():
            yield audio
