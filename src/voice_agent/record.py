"""Record a test clip from the default microphone (16 kHz mono PCM16).

  uv run record-clip en_question --seconds 5
  uv run record-clip --list-devices
"""

from __future__ import annotations

import argparse
import sys

from voice_agent.audio_io import save_wav
from voice_agent.config import PROJECT_ROOT

RECORDINGS = PROJECT_ROOT / "recordings"


def main(argv: list[str] | None = None) -> int:
    import sounddevice as sd

    ap = argparse.ArgumentParser(description="Record a mic clip to recordings/<name>.wav")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--device", default=None, help="input device index or name")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--rate", type=int, default=16000, help="sample rate (use 24000 for TTS voice clips)")
    ap.add_argument("--dir", default=None, help="output folder (default: recordings/)")
    args = ap.parse_args(argv)

    if args.list_devices:
        print(sd.query_devices())
        return 0
    if not args.name:
        ap.error("name is required")

    sr = args.rate
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    input("Press Enter, then speak...")
    print(f"Recording {args.seconds:.0f}s ...")
    audio = sd.rec(int(args.seconds * sr), samplerate=sr, channels=1, dtype="float32", device=device)
    sd.wait()
    out_dir = PROJECT_ROOT / args.dir if args.dir else RECORDINGS
    path = save_wav(out_dir / f"{args.name}.wav", audio[:, 0], sr)
    peak = float(abs(audio).max())
    print(f"Saved {path}  (peak level {peak:.2f}{', TOO QUIET' if peak < 0.05 else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
