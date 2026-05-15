"""Task 2 — per-phase timing baseline for one transcribe call.

Splits `_transcribe_loaded_components` into 7 measurable phases, each bracketed
by `mx.eval` so timings reflect real GPU work (not async graph construction):

  1. mel              compute_features
  2. encoder          model.audio_tower
  3. prompt           tokenizer.build_prompt_tokens + position_ids
  4. prefill          model.prefill (first logits, populates KV cache)
  5. decode_loop      generate_with_info (autoregressive steps)
  6. detok            tokenizer.decode + parse_asr_output
  7. itn              optional Chinese inverse text normalization

Methodology:
- 1 warm pass (Metal kernel JIT, allocator warm) + N steady iterations
- Per-iteration: bracket each phase with mx.eval to force materialization
- For decode loop: also report TTFT (time to first generated token, includes
  prefill) and average per-step latency to separate prefill vs incremental cost
- Throwaway script — output captured to EXPERIMENTS_LOG.md by hand
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

import mlx_qwen3_asr as m
from mlx_qwen3_asr.audio import SAMPLE_RATE, compute_features, load_audio_np
from mlx_qwen3_asr.generate import (
    GenerationConfig,
    _build_decode_positions,
    _detect_repetition,
    _sample,
    resolve_max_new_tokens,
)
from mlx_qwen3_asr.tokenizer import (
    _TokenizerHolder,
    canonicalize_language,
    parse_asr_output,
)

AUDIO = Path("唐雪梅 知性_02-07.wav")
MODEL_DIR = Path("models/Qwen3-ASR-0.6B")
LANGUAGE = "Chinese"
N_STEADY = 5
DTYPE = mx.float16

PHASES = [
    "mel",
    "encoder",
    "prompt",
    "prefill",
    "decode_loop",
    "detok",
    "itn",
]


def _try_load_itn():
    try:
        from itn.chinese.inverse_normalizer import InverseNormalizer
        return InverseNormalizer()
    except Exception as exc:
        print(f"[itn] unavailable ({exc}); will skip ITN phase")
        return None


def _instrumented_transcribe(
    audio_np: np.ndarray,
    *,
    model,
    tokenizer,
    itn,
) -> dict:
    """One transcribe pass; returns per-phase ms + diagnostics."""
    forced_language = canonicalize_language(LANGUAGE)
    timings: dict[str, float] = {}

    # --- 1. mel ---
    t0 = time.perf_counter()
    mel, feature_lens = compute_features(audio_np)
    mx.eval(mel, feature_lens)
    timings["mel"] = (time.perf_counter() - t0) * 1e3

    # --- 2. encoder ---
    t0 = time.perf_counter()
    audio_features, _ = model.audio_tower(mel.astype(DTYPE), feature_lens)
    mx.eval(audio_features)
    timings["encoder"] = (time.perf_counter() - t0) * 1e3
    n_audio_tokens = int(audio_features.shape[1])

    # --- 3. prompt ---
    t0 = time.perf_counter()
    prompt_tokens = tokenizer.build_prompt_tokens(
        n_audio_tokens=n_audio_tokens,
        language=forced_language,
        context="",
    )
    input_ids = mx.array([prompt_tokens])
    seq_len = input_ids.shape[1]
    positions = mx.arange(seq_len)[None, :]
    position_ids = mx.stack([positions, positions, positions], axis=1)
    mx.eval(input_ids, position_ids)
    timings["prompt"] = (time.perf_counter() - t0) * 1e3

    # --- decode budget (matches transcribe.py adaptive cap) ---
    chunk_duration_sec = float(len(audio_np) / SAMPLE_RATE)
    max_new = resolve_max_new_tokens(None, audio_duration_sec=chunk_duration_sec)
    config = GenerationConfig(max_new_tokens=max_new, temperature=0.0)
    cache = model.create_cache(max_seq_len=int(seq_len + max_new))

    # --- 4. prefill ---
    t0 = time.perf_counter()
    logits = model.prefill(
        input_ids=input_ids,
        audio_features=audio_features,
        position_ids=position_ids,
        cache=cache,
    )
    mx.eval(logits)
    timings["prefill"] = (time.perf_counter() - t0) * 1e3

    # First token (counted under decode_loop, but timed for TTFT visibility).
    t0 = time.perf_counter()
    token = _sample(logits, config.temperature)
    generated = [token]
    first_step_ms = (time.perf_counter() - t0) * 1e3

    # --- 5. decode loop (steps 2..N) ---
    next_pos_3d = _build_decode_positions(
        seq_len=seq_len,
        max_new_tokens=config.max_new_tokens,
        dtype=position_ids.dtype,
    )
    per_step_ms: list[float] = []
    decode_t0 = time.perf_counter()
    for step in range(1, config.max_new_tokens):
        if token in config.eos_token_ids:
            break
        if _detect_repetition(generated):
            break
        s_t0 = time.perf_counter()
        next_ids = mx.array([[token]])
        next_position_ids = next_pos_3d[:, :, step - 1 : step]
        logits = model.step(
            input_ids=next_ids,
            position_ids=next_position_ids,
            cache=cache,
            validate_input_ids=False,
        )
        token = _sample(logits, config.temperature)
        generated.append(token)
        per_step_ms.append((time.perf_counter() - s_t0) * 1e3)
    timings["decode_loop"] = (time.perf_counter() - decode_t0) * 1e3 + first_step_ms

    # Strip trailing EOS like _finalize_generation_result.
    if generated and generated[-1] in config.eos_token_ids:
        generated = generated[:-1]

    # --- 6. detok + parse ---
    t0 = time.perf_counter()
    raw_text = tokenizer.decode(generated)
    lang, text = parse_asr_output(raw_text, user_language=forced_language)
    timings["detok"] = (time.perf_counter() - t0) * 1e3

    # --- 7. ITN (Chinese only, optional) ---
    if itn is not None:
        t0 = time.perf_counter()
        text = itn.normalize(text)
        timings["itn"] = (time.perf_counter() - t0) * 1e3
    else:
        timings["itn"] = 0.0

    return {
        "timings_ms": timings,
        "n_audio_tokens": n_audio_tokens,
        "prompt_len": seq_len,
        "generated_tokens": len(generated),
        "first_step_ms": first_step_ms,
        "per_step_ms": per_step_ms,
        "text": text,
        "language": lang,
    }


def _agg(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _print_table(rows: list[tuple[str, str]]) -> None:
    width_l = max(len(a) for a, _ in rows)
    for a, b in rows:
        print(f"  {a:<{width_l}}  {b}")


def main() -> None:
    print(f"=== exp_phase_timing — {AUDIO.name} ===")
    print(f"model={MODEL_DIR.name}  dtype={DTYPE}  language={LANGUAGE}  N_steady={N_STEADY}")
    print()

    sess = m.Session(model=str(MODEL_DIR))
    model = sess.model
    tokenizer = _TokenizerHolder.get(str(MODEL_DIR))
    itn = _try_load_itn()

    audio_np = load_audio_np(str(AUDIO), sr=SAMPLE_RATE)
    duration = len(audio_np) / SAMPLE_RATE
    print(f"audio: {len(audio_np)} samples @ {SAMPLE_RATE} Hz = {duration:.3f} s")

    # Warm pass — discarded.
    print("\n[warm pass] (discarded)")
    warm = _instrumented_transcribe(audio_np, model=model, tokenizer=tokenizer, itn=itn)
    warm_total = sum(warm["timings_ms"].values())
    print(f"  warm total: {warm_total:.1f} ms  text='{warm['text'][:60]}'")

    # Steady iterations.
    print(f"\n[steady x{N_STEADY}]")
    runs: list[dict] = []
    for i in range(N_STEADY):
        out = _instrumented_transcribe(audio_np, model=model, tokenizer=tokenizer, itn=itn)
        runs.append(out)
        total = sum(out["timings_ms"].values())
        print(
            f"  iter {i+1}: total={total:6.1f} ms  "
            f"gen={out['generated_tokens']}t  TTFT(prefill+1st)≈"
            f"{out['timings_ms']['prefill'] + out['first_step_ms']:5.1f} ms"
        )

    # Aggregate.
    print("\n=== per-phase (ms) — mean / min / max / stdev ===")
    totals = [sum(r["timings_ms"].values()) for r in runs]
    rows: list[tuple[str, str]] = []
    for phase in PHASES:
        vals = [r["timings_ms"][phase] for r in runs]
        agg = _agg(vals)
        pct = agg["mean"] / statistics.mean(totals) * 100.0
        rows.append((
            phase,
            f"{agg['mean']:7.2f}  "
            f"[{agg['min']:6.2f}, {agg['max']:6.2f}]  "
            f"σ={agg['stdev']:5.2f}  "
            f"({pct:5.1f}%)",
        ))
    rows.append((
        "TOTAL",
        f"{statistics.mean(totals):7.2f}  "
        f"[{min(totals):6.2f}, {max(totals):6.2f}]  "
        f"σ={statistics.stdev(totals) if len(totals)>1 else 0:5.2f}",
    ))
    _print_table(rows)

    # Decode-loop substructure.
    all_per_step = [s for r in runs for s in r["per_step_ms"]]
    first_steps = [r["first_step_ms"] for r in runs]
    print("\n=== decode-loop substructure ===")
    if all_per_step:
        print(
            f"  per-step (excl. 1st): mean={statistics.mean(all_per_step):.2f} ms  "
            f"median={statistics.median(all_per_step):.2f}  "
            f"min={min(all_per_step):.2f}  max={max(all_per_step):.2f}  "
            f"n={len(all_per_step)}"
        )
        tps = 1000.0 / statistics.mean(all_per_step)
        print(f"  ⇒ steady-state decode tokens/s ≈ {tps:.1f}")
    print(f"  1st sample (after prefill, included in decode_loop): mean={statistics.mean(first_steps):.2f} ms")

    # Diagnostic shapes.
    r0 = runs[0]
    print("\n=== shapes / counts ===")
    print(f"  n_audio_tokens={r0['n_audio_tokens']}  prompt_len={r0['prompt_len']}  generated_tokens={r0['generated_tokens']}")
    print(f"  text: '{r0['text']}'")
    print(f"  language: {r0['language']}")

    # RTF.
    mean_total_sec = statistics.mean(totals) / 1000.0
    rtf = mean_total_sec / duration
    print(f"\n  RTF (mean steady) = {mean_total_sec:.3f}s / {duration:.3f}s = {rtf:.3f}")
    print(f"  ITN status: {'enabled' if itn is not None else 'unavailable'}")


if __name__ == "__main__":
    main()
