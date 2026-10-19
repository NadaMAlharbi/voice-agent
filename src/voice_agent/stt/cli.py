"""STT command-line tools.

  uv run stt-test recordings\\en_complete.wav recordings\\ar_complete.wav
  uv run stt-test --self-check        # silence and noise must give no text
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

from voice_agent.audio_io import load_wav
from voice_agent.config import load_config
from voice_agent.stt.base import SR, make_engine, warmup


def _expand(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in patterns:  # PowerShell does not expand globs for us
        files += [Path(f) for f in sorted(glob.glob(p))] or [Path(p)]
    return [f for f in files if f.suffix.lower() in (".wav", ".flac", ".ogg")]


def _load_engine(name: str, cfg):
    engine = make_engine(name, cfg)
    t0 = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - t0
    w = warmup(engine)
    print(f"[{name}] loaded in {load_s:.1f}s, warmup {w:.0f} ms | {engine.device_info()}")
    return engine


def noise_suite(seconds: float = 3.0) -> dict:
    """Non-speech inputs that must never produce text (what a live mic sends between words)."""
    import numpy as np

    n = int(seconds * SR)
    rng = np.random.default_rng(0)
    t = np.arange(n) / SR
    white = rng.standard_normal(n).astype(np.float32)
    pink = np.cumsum(white)
    pink = (pink - np.convolve(pink, np.ones(400) / 400, "same")).astype(np.float32)
    clicks = np.zeros(n, np.float32)
    clicks[rng.integers(0, n - 200, 12)[:, None] + np.arange(200)] = 0.3 * rng.standard_normal((12, 200))
    return {
        "silence": np.zeros(n, np.float32),
        "white noise quiet": 0.005 * white,
        "white noise loud": 0.05 * white,
        "pink noise (room/fan)": 0.1 * pink / (np.abs(pink).max() + 1e-9),
        "mains hum 50 Hz": (0.05 * np.sin(2 * np.pi * 50 * t)).astype(np.float32),
        "keyboard-like clicks": clicks,
    }


def test_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Transcribe files and show latency")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--engine", default=None, help="STT engine (default: config stt.engine)")
    ap.add_argument("--language", default=None, help="auto | ar | en (default: config)")
    ap.add_argument("--self-check", action="store_true",
                    help="also run silence/noise inputs that must produce NO text")
    args = ap.parse_args(argv)
    cfg = load_config()
    engine = _load_engine(args.engine or cfg.stt.engine, cfg)
    lang = args.language or cfg.stt.language

    for f in _expand(args.files):
        audio = load_wav(f, SR)
        t = engine.transcribe(audio, lang)
        print(f"\n{f.name}  [{t.language}]  {t.ms:.0f} ms for {t.audio_s:.1f}s audio (RTF {t.rtf:.3f})")
        print(f"  {t.text or '(no text)'}")
        print(f"  info: {t.info}")

    if not args.self_check:
        return 0
    print("\n== self-check: non-speech must give no text")
    bad = 0
    for name, audio in noise_suite().items():
        t = engine.transcribe(audio, lang)
        ok = not t.text
        bad += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<24} {t.ms:5.0f} ms  "
              f"{'(no text, ' + t.info.get('skip', '') + ')' if ok else 'HALLUCINATED: ' + t.text}")
    print(f"RESULT: {'OK' if not bad else f'{bad} hallucination(s)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(test_main())
