"""Task 3 — decode-loop internal decomposition + step-vs-cache-length trend.

Two questions:
  A) What does one `model.step` spend its 26 ms on?
     Decompose into: py_setup / embed / rotary_cos_sin / transformer_stack / lm_head / sample_item
  B) Does step latency stay constant as KV cache grows?
     Run a forced 200-step decode (ignore EOS), record per-step natural ms,
     bucket by step index.

Methodology notes:
- Component breakdown: bracket each piece with mx.eval. This adds sync overhead
  the natural path doesn't have, so component sums DO NOT equal natural step time.
  We use these for RELATIVE proportions, not absolute calibration.
- Natural per-step timing: the loop already syncs at .item() (in _sample), so
  perf_counter around step+sample is a faithful per-step ms.
- We run instrumented step at two cache positions (early ≈1, late ≈200) to
  see whether component proportions drift with cache length.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import mlx_qwen3_asr as m
from mlx_qwen3_asr.audio import SAMPLE_RATE, compute_features, load_audio_np
from mlx_qwen3_asr.generate import (
    GenerationConfig,
    _build_decode_positions,
    _sample,
    resolve_max_new_tokens,
)
from mlx_qwen3_asr.tokenizer import _TokenizerHolder, canonicalize_language

AUDIO = Path("唐雪梅 知性_02-07.wav")
MODEL_DIR = Path("models/Qwen3-ASR-0.6B")
LANGUAGE = "Chinese"
DTYPE = mx.float16
N_FORCED_STEPS = 200          # ignore EOS, decode this many for trend
INSTRUMENTED_REPEATS = 5      # take this many instrumented samples per anchor
INSTRUMENTED_ANCHORS = [1, 50, 100, 200]   # cache positions where we decompose

COMPONENTS = [
    "py_setup",
    "embed",
    "rotary_cos_sin",
    "transformer_stack",
    "lm_head",
    "sample_item",
]


def _instrumented_step(model, token, step_idx, next_pos_3d, cache):
    """One step decomposed by mx.eval boundaries. Returns (new_token, timings_ms)."""
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    next_ids = mx.array([[token]])
    next_position_ids = next_pos_3d[:, :, step_idx - 1 : step_idx]
    mx.eval(next_ids, next_position_ids)
    timings["py_setup"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    embeds = model._embed_tokens(next_ids, validate_input_ids=False)
    mx.eval(embeds)
    timings["embed"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    cos, sin = model.model.rotary_emb(next_position_ids, dtype=embeds.dtype)
    mx.eval(cos, sin)
    timings["rotary_cos_sin"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    h = embeds
    for i, layer in enumerate(model.model.layers):
        h = layer(h, cos, sin, mask=None, cache=cache, layer_idx=i)
    h = model.model.norm(h)
    mx.eval(h)
    timings["transformer_stack"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    logits = model.lm_head(h)
    mx.eval(logits)
    timings["lm_head"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    new_token = _sample(logits, 0.0)  # argmax + .item() forces sync
    timings["sample_item"] = (time.perf_counter() - t0) * 1e3

    return new_token, timings


def _natural_step(model, token, step_idx, next_pos_3d, cache):
    """One step the natural way (no extra mx.eval). Returns (new_token, ms)."""
    t0 = time.perf_counter()
    next_ids = mx.array([[token]])
    next_position_ids = next_pos_3d[:, :, step_idx - 1 : step_idx]
    logits = model.step(
        input_ids=next_ids,
        position_ids=next_position_ids,
        cache=cache,
        validate_input_ids=False,
    )
    new_token = _sample(logits, 0.0)
    return new_token, (time.perf_counter() - t0) * 1e3


def _setup_decode_state(model, tokenizer, audio_np):
    """Return (token_after_prefill, next_pos_3d, cache, seq_len, max_new)."""
    forced_language = canonicalize_language(LANGUAGE)
    mel, feature_lens = compute_features(audio_np)
    audio_features, _ = model.audio_tower(mel.astype(DTYPE), feature_lens)
    n_audio_tokens = int(audio_features.shape[1])

    prompt_tokens = tokenizer.build_prompt_tokens(
        n_audio_tokens=n_audio_tokens, language=forced_language, context=""
    )
    input_ids = mx.array([prompt_tokens])
    seq_len = input_ids.shape[1]
    positions = mx.arange(seq_len)[None, :]
    position_ids = mx.stack([positions, positions, positions], axis=1)

    max_new = max(N_FORCED_STEPS + 16, resolve_max_new_tokens(None, audio_duration_sec=len(audio_np) / 16000))
    cache = model.create_cache(max_seq_len=int(seq_len + max_new))

    logits = model.prefill(
        input_ids=input_ids,
        audio_features=audio_features,
        position_ids=position_ids,
        cache=cache,
    )
    token = _sample(logits, 0.0)

    next_pos_3d = _build_decode_positions(
        seq_len=seq_len, max_new_tokens=max_new, dtype=position_ids.dtype
    )
    return token, next_pos_3d, cache, seq_len, max_new


def _agg(values):
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    print(f"=== exp_decode_internal — {AUDIO.name} ===")
    print(f"model={MODEL_DIR.name}  dtype={DTYPE}  N_forced={N_FORCED_STEPS}  anchors={INSTRUMENTED_ANCHORS}")
    print()

    sess = m.Session(model=str(MODEL_DIR))
    model = sess.model
    tokenizer = _TokenizerHolder.get(str(MODEL_DIR))
    audio_np = load_audio_np(str(AUDIO), sr=SAMPLE_RATE)

    # Warm pass: Metal kernels JIT.
    print("[warm pass]")
    token, next_pos_3d, cache, seq_len, max_new = _setup_decode_state(model, tokenizer, audio_np)
    for step in range(1, 16):
        token, _ = _natural_step(model, token, step, next_pos_3d, cache)
    print(f"  warm done, prompt_len={seq_len}\n")

    # ---- Component breakdown at each anchor ----
    # Strategy: fresh decode setup per anchor, run N natural steps to bring cache
    # to anchor-1, then take INSTRUMENTED_REPEATS instrumented samples in a row.
    print("=== component breakdown (mean ms, stdev) ===")
    header = f"  {'anchor (step)':<14}  " + "  ".join(f"{c:>17}" for c in COMPONENTS) + "    sum"
    print(header)
    breakdown_rows = []
    for anchor in INSTRUMENTED_ANCHORS:
        token, next_pos_3d, cache, seq_len, max_new = _setup_decode_state(model, tokenizer, audio_np)
        # natural decode up to step (anchor-1) so cache offset = seq_len + (anchor-1)
        for step in range(1, anchor):
            token, _ = _natural_step(model, token, step, next_pos_3d, cache)
        # take INSTRUMENTED_REPEATS instrumented steps starting at `anchor`
        per_component = {c: [] for c in COMPONENTS}
        cur_step = anchor
        for _ in range(INSTRUMENTED_REPEATS):
            token, timings = _instrumented_step(model, token, cur_step, next_pos_3d, cache)
            for c, v in timings.items():
                per_component[c].append(v)
            cur_step += 1
        means = {c: _agg(per_component[c])["mean"] for c in COMPONENTS}
        stdevs = {c: _agg(per_component[c])["stdev"] for c in COMPONENTS}
        cells = "  ".join(f"{means[c]:8.3f}±{stdevs[c]:5.2f}" for c in COMPONENTS)
        total_sum = sum(means.values())
        cache_offset = seq_len + (anchor - 1)
        print(f"  step {anchor:<3} (cache={cache_offset:<3})  {cells}    {total_sum:6.2f}")
        breakdown_rows.append((anchor, cache_offset, means, stdevs, total_sum))

    # Show % share at first anchor (most representative of natural step).
    first = breakdown_rows[0]
    print(f"\n=== % share at step {first[0]} (cache_offset={first[1]}) ===")
    means = first[2]; total = first[4]
    for c in COMPONENTS:
        pct = means[c] / total * 100.0 if total > 0 else 0.0
        print(f"  {c:<20}  {means[c]:7.3f} ms  ({pct:5.1f}%)")
    print(f"  {'SUM (instrumented)':<20}  {total:7.3f} ms  (note: includes mx.eval sync overhead)")

    # ---- Trend: forced 200-step natural decode ----
    print(f"\n=== natural per-step latency vs step index ({N_FORCED_STEPS} steps, EOS ignored) ===")
    token, next_pos_3d, cache, seq_len, max_new = _setup_decode_state(model, tokenizer, audio_np)
    per_step_ms = []
    for step in range(1, N_FORCED_STEPS + 1):
        token, ms = _natural_step(model, token, step, next_pos_3d, cache)
        per_step_ms.append(ms)

    # Buckets (step ranges).
    buckets = [(1, 10), (11, 50), (51, 100), (101, 150), (151, 200)]
    print(f"  {'step range':<12}  {'cache_offset range':<22}  {'mean ms':>8}  {'min':>6}  {'max':>6}  {'stdev':>6}  n")
    for lo, hi in buckets:
        slice_ms = per_step_ms[lo - 1 : hi]
        if not slice_ms:
            continue
        agg = _agg(slice_ms)
        cache_lo = seq_len + lo - 1
        cache_hi = seq_len + hi - 1
        print(
            f"  [{lo:>3}, {hi:>3}]   "
            f"[{cache_lo:>4}, {cache_hi:>4}]          "
            f"{agg['mean']:8.3f}  {agg['min']:6.2f}  {agg['max']:6.2f}  {agg['stdev']:6.2f}  {len(slice_ms)}"
        )

    # Slope estimate: linear regression step_idx → ms.
    n = len(per_step_ms)
    xs = list(range(1, n + 1))
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(per_step_ms)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, per_step_ms))
    den = sum((x - mean_x) ** 2 for x in xs)
    slope = num / den if den else 0.0
    print(f"\n  linear slope: {slope:+.4f} ms per step  (over {n} steps)")
    print(f"  mean overall: {mean_y:.3f} ms  ⇒ {1000.0/mean_y:.1f} tok/s")
    print(f"  first 5: {[f'{v:.2f}' for v in per_step_ms[:5]]}")
    print(f"  last 5:  {[f'{v:.2f}' for v in per_step_ms[-5:]]}")

    # Drift across anchors (sanity check that breakdown does or doesn't change with cache).
    print("\n=== component drift across cache positions (mean ms) ===")
    print(f"  {'component':<20}  " + "  ".join(f"step{a:>4}" for a, _, _, _, _ in breakdown_rows))
    for c in COMPONENTS:
        cells = "  ".join(f"{row[2][c]:8.3f}" for row in breakdown_rows)
        print(f"  {c:<20}  {cells}")


if __name__ == "__main__":
    main()
