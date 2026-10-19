"""tts-test: speak a test sentence with each voice, measure speed, save wavs, check the voices.

  uv run tts-test                         # every voice in config.yaml (English and Arabic, female and male)
  uv run tts-test --voices ar-female,ar-male
  uv run tts-test --asr                   # also transcribe each wav with Whisper (intelligibility)
  uv run tts-test --engine tone           # dry run, no models

Wavs are saved to outputs/tts/<voice>.wav. "pitch" is the median voice pitch: female voices
are usually above ~165 Hz, male voices below. The two voices of one language must sound different.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import numpy as np

from voice_agent.audio_io import save_wav
from voice_agent.chunker import chunk_text
from voice_agent.config import PROJECT_ROOT, load_config
from voice_agent.tts.base import all_voices, make_tts

SAMPLES = {
    "en": "Hello! I'm your local voice assistant. I run fully offline on this laptop, so how can I help you today?",
    "ar": "مرحباً! أنا مساعدك الصوتي المحلي. أعمل بالكامل دون اتصال بالإنترنت، فكيف يمكنني مساعدتك اليوم؟",
}


def pitch_hz(audio: np.ndarray, sr: int) -> float | None:
    # This function measures the median pitch of the voice (None if there is too little speech).
    import librosa

    x = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)[: 16000 * 15]
    f0, voiced, _ = librosa.pyin(x, fmin=60, fmax=400, sr=16000, frame_length=1024)
    f0 = f0[voiced & np.isfinite(f0)]
    return float(np.median(f0)) if len(f0) >= 20 else None


def speak(engine, text: str, voice, cfg) -> tuple[np.ndarray, float, float]:
    # This function speaks a text chunk by chunk, like the live pipeline.
    pieces, first_ms = [], None
    t0 = time.perf_counter()
    for chunk in chunk_text(text, cfg.chunker, language=voice.language):
        for audio in engine.synthesize(chunk, voice):
            if first_ms is None:
                first_ms = (time.perf_counter() - t0) * 1000
            pieces.append(audio)
    wav = np.concatenate(pieces) if pieces else np.zeros(1, np.float32)
    return wav, first_ms or 0.0, time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TTS test: every voice, speed, pitch, wavs")
    ap.add_argument("--engine", default=None, help="models | tone")
    ap.add_argument("--voices", default="all", help="comma list of voice names, or 'all'")
    ap.add_argument("--text", default=None, help="your own sentence (same for every voice)")
    ap.add_argument("--asr", action="store_true", help="transcribe each wav with Whisper and report WER")
    args = ap.parse_args(argv)

    cfg = load_config()
    voices = all_voices(cfg)
    if args.voices != "all":
        voices = {n: voices[n] for n in args.voices.split(",")}
    if args.engine == "tone":
        voices = {n: replace(v, engine="tone") for n, v in voices.items()}

    print("voice mapping (config.yaml -> model voice):")
    for v in voices.values():
        print(f"  {v.name:<20} {v.engine:<7} {v.voice_id}")

    engine = make_tts(cfg, args.engine)
    t0 = time.perf_counter()
    engine.load(voices)
    print(f"\nloaded in {time.perf_counter() - t0:.1f}s | {engine.device_info()}\n")

    stt = None
    if args.asr:
        from voice_agent.stt.base import make_engine

        stt = make_engine("whisper", cfg)
        stt.load()

    out_dir = PROJECT_ROOT / "outputs" / "tts"
    print(f"{'voice':<20}{'first audio':>12}{'total':>8}{'audio':>8}{'RTF':>6}{'pitch':>9}  file")
    pitches = {}
    for v in voices.values():
        text = args.text or SAMPLES[v.language]
        speak(engine, text, v, cfg)                       # warm-up run
        wav, first_ms, total = speak(engine, text, v, cfg)
        dur = len(wav) / engine.sample_rate
        hz = pitch_hz(wav, engine.sample_rate)
        pitches[v.name] = hz
        path = save_wav(out_dir / f"{v.name}.wav", wav, engine.sample_rate)
        line = (f"{v.name:<20}{first_ms:>9.0f} ms{total:>7.2f}s{dur:>7.1f}s{total / max(dur, 1e-6):>6.2f}"
                f"{(f'{hz:.0f} Hz') if hz else '-':>9}  {path.name}")
        if stt:
            import soxr

            from voice_agent.stt.wer import wer

            heard = stt.transcribe(soxr.resample(wav, engine.sample_rate, 16000).astype(np.float32), v.language).text
            line += f"  WER {wer(text, heard):.0%}  \"{heard.strip()}\""
        print(line)

    # Check: the female and male voice of a language must not sound the same.
    print()
    problems = 0
    for v in voices.values():
        if v.gender != "female":
            continue
        for m in voices.values():
            if m.gender == "male" and m.language == v.language:
                f, mm = pitches.get(v.name), pitches.get(m.name)
                ok = f is not None and mm is not None and f - mm >= 25
                problems += not ok
                print(f"{'ok  ' if ok else 'WARN'} {v.name} vs {m.name}: pitch "
                      f"{f or 0:.0f} Hz vs {mm or 0:.0f} Hz {'(female higher)' if ok else '(too close: listen to the wavs)'}")
    print(f"\nwavs in {out_dir}. RTF < 1 = faster than real time.")
    return 1 if problems and args.engine != "tone" else 0


if __name__ == "__main__":
    sys.exit(main())
