"""Small audio helpers: load/save wav, convert formats, split into frames."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def load_wav(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    """Read any wav/flac/ogg -> mono float32 in [-1, 1] at target_sr."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = soxr.resample(audio, sr, target_sr, quality="HQ").astype(np.float32)
    return audio


def save_wav(path: str | Path, audio: np.ndarray, sr: int = 16000) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


def float_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def pcm16_frames(pcm: bytes, sr: int = 16000, frame_ms: int = 20) -> Iterator[bytes]:
    """Yield fixed-size PCM16 frames, like a live mic stream. Last partial frame is zero-padded."""
    size = sr * frame_ms // 1000 * 2
    for i in range(0, len(pcm), size):
        chunk = pcm[i : i + size]
        yield chunk if len(chunk) == size else chunk + b"\x00" * (size - len(chunk))
