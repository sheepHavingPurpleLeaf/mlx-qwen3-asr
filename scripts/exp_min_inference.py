"""Task 0 / Step 1 — minimal end-to-end inference with phase timing.

Output is captured to EXPERIMENTS_LOG.md by hand. Throwaway script.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

AUDIO = Path("唐雪梅 知性_02-07.wav")
MODEL_DIR = Path("models/Qwen3-ASR-0.6B")


def main() -> None:
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    import mlx_qwen3_asr as m

    timings["import_sec"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    sess = m.Session(model=str(MODEL_DIR))
    timings["session_init_and_load_sec"] = round(time.perf_counter() - t0, 3)

    # Warmup pass (first call may JIT / compile Metal kernels)
    t0 = time.perf_counter()
    res_warm = sess.transcribe(str(AUDIO))
    timings["transcribe_warm_sec"] = round(time.perf_counter() - t0, 3)

    # Steady-state pass
    t0 = time.perf_counter()
    res = sess.transcribe(str(AUDIO))
    timings["transcribe_steady_sec"] = round(time.perf_counter() - t0, 3)

    print("=== timings ===")
    print(json.dumps(timings, indent=2, ensure_ascii=False))
    print("=== result ===")
    print("text:", res.text)
    print("language:", getattr(res, "language", None))
    if hasattr(res, "tokens"):
        print("token_count:", len(res.tokens) if res.tokens else "n/a")
    print("audio_duration_sec: 6.04")
    rtf = timings["transcribe_steady_sec"] / 6.04
    print(f"rtf_steady: {rtf:.3f}")


if __name__ == "__main__":
    main()
