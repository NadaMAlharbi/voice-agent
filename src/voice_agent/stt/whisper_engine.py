"""faster-whisper large-v3-turbo on the GPU (CTranslate2)."""

from __future__ import annotations

import time

import numpy as np

from voice_agent import cuda_dlls
from voice_agent.models import enable_offline, load_registry, model_path
from voice_agent.stt.base import LANGS, SR, Transcript, speech_only


def choose_language(p_ar: float, p_en: float, last: str | None, speech_s: float) -> str:
    """Arabic or English from Whisper's language scores, with a bias to the last language.

    Short or unclear audio often gets the wrong language. A wrong forced language makes
    Whisper translate (Arabic speech -> English text). So the language only changes when
    the new one clearly wins.
    """
    best = "ar" if p_ar >= p_en else "en"
    if last not in LANGS or best == last:
        return best
    p_best, p_last = (p_ar, p_en) if best == "ar" else (p_en, p_ar)
    margin = 0.5 if speech_s < 1.0 else 0.25        # short audio needs a clearer win
    return best if p_best - p_last >= margin else last


class WhisperEngine:
    name = "whisper"

    def __init__(self, cfg):
        c = cfg.stt.whisper
        self.model_name = c.get("model", "whisper-large-v3-turbo")
        self.compute_type = c.get("compute_type", "float16")
        self.beam_size = int(c.get("beam_size", 1))
        self.vad_threshold = float(c.get("vad_threshold", 0.5))
        self.min_speech_ms = int(c.get("min_speech_ms", 250))
        self.no_speech_threshold = float(c.get("no_speech_threshold", 0.6))
        self.log_prob_threshold = float(c.get("log_prob_threshold", -1.0))
        self.compression_ratio_threshold = float(c.get("compression_ratio_threshold", 2.4))
        self.device = cfg.device
        self.gpu_index = cfg.gpu_index
        self.models_dir = cfg.paths.models_dir
        self.model = None
        self.last_language: str | None = None   # language of the last transcript (for choose_language)

    def load(self) -> None:
        enable_offline()
        cuda_dlls.prepare()
        from faster_whisper import WhisperModel

        path = model_path(load_registry()[self.model_name], self.models_dir)
        compute = self.compute_type if self.device == "cuda" else "int8"
        self.model = WhisperModel(
            str(path), device=self.device, device_index=self.gpu_index, compute_type=compute
        )

    def warmup(self) -> None:
        """Run the model itself (transcribe() would skip silence without touching it)."""
        noise = (np.random.default_rng(0).standard_normal(SR) * 0.01).astype(np.float32)
        self.model.detect_language(audio=noise)
        list(self.model.transcribe(noise, language="en", beam_size=1, without_timestamps=True)[0])
        speech_only(noise)  # loads the VAD model too

    def device_info(self) -> str:
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        return f"faster-whisper on {self.device} (compute={self.compute_type}, CUDA devices seen by CTranslate2: {n})"

    def detect(self, speech: np.ndarray) -> tuple[str, dict]:
        """Language ID over {ar, en} only. Returns (lang, diagnostics)."""
        top, top_p, all_probs = self.model.detect_language(audio=speech)
        probs = dict(all_probs or [])
        p_ar, p_en = probs.get("ar", 0.0), probs.get("en", 0.0)
        lang = choose_language(p_ar, p_en, self.last_language, len(speech) / SR)
        return lang, {"lid_top": top, "lid_top_p": round(float(top_p), 3),
                      "p_ar": round(p_ar, 3), "p_en": round(p_en, 3)}

    def _reject(self, seg) -> str | None:
        if seg.no_speech_prob > self.no_speech_threshold and seg.avg_logprob < self.log_prob_threshold:
            return "no_speech"
        if seg.compression_ratio > self.compression_ratio_threshold:
            return "repetition"
        return None

    def transcribe(self, audio: np.ndarray, language: str = "auto") -> Transcript:
        t0 = time.perf_counter()
        dur = len(audio) / SR
        speech, speech_s = speech_only(audio, self.vad_threshold, self.min_speech_ms)
        info: dict = {"speech_s": round(speech_s, 2)}
        if speech_s == 0.0:
            info["skip"] = "no_speech_vad"
            return Transcript("", language if language in LANGS else "en", (time.perf_counter() - t0) * 1000, dur, info)

        if language in LANGS:
            lang = language
        else:
            lang, lid = self.detect(speech)
            info.update(lid)

        segments, _ = self.model.transcribe(
            speech,
            language=lang,                     # forced: never decodes in another language
            task="transcribe",
            beam_size=self.beam_size,
            temperature=0.0,                  
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,              
            no_speech_threshold=self.no_speech_threshold,
            log_prob_threshold=self.log_prob_threshold,
            compression_ratio_threshold=self.compression_ratio_threshold,
        )
        kept, rejected = [], []
        for s in segments: 
            reason = self._reject(s)
            if reason:
                rejected.append({"text": s.text.strip(), "reason": reason, "no_speech_prob": round(s.no_speech_prob, 2),
                                 "avg_logprob": round(s.avg_logprob, 2), "cr": round(s.compression_ratio, 2)})
            else:
                kept.append(s.text.strip())
        if rejected:
            info["rejected"] = rejected
        text = " ".join(t for t in kept if t).strip()
        if text:
            self.last_language = lang
        if not text:
            info.setdefault("skip", "rejected_by_whisper_confidence" if rejected else "empty")
        return Transcript(text, lang, (time.perf_counter() - t0) * 1000, dur, info)
