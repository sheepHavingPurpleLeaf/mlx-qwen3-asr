"""Manual smoke test: load engine, record 3s, transcribe, print.

Run: source .venv/bin/activate && python -m examples.voice_input._smoketest
"""
from __future__ import annotations

import time

from examples.voice_input.audio_capture import AudioCapture
from examples.voice_input.engine import Engine


def main() -> None:
    eng = Engine()
    print("loading model…")
    print(f"  load:    {eng.load():.2f}s")
    print(f"  warmup:  {eng.warmup():.2f}s")

    cap = AudioCapture()
    cap.open()
    print("\nrecording 3 seconds — speak now.")
    cap.start()
    time.sleep(3.0)
    audio = cap.stop()
    cap.close()
    print(f"  captured {len(audio)} samples = {len(audio)/16000:.2f}s")

    text, t = eng.transcribe(audio)
    print(f"\nasr {t:.2f}s → {text!r}")


if __name__ == "__main__":
    main()
