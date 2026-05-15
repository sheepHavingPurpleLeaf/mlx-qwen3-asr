"""Task 4 — try three decode-loop optimizations against the baseline.

Variants (toggleable, run all):
  1. baseline                       — vanilla model.step
  2. +mx.compile                    — wrap step body with mx.compile
  3. +fused SwiGLU                  — merge gate_proj+up_proj into one Linear
  4. +lm_head int4                  — quantize ONLY lm_head to int4 (full vocab)
  5. all                            — compile + fused SwiGLU + lm_head int4

Plus a sub-experiment: profile lm_head GEMV at varying output dims to confirm
whether the 7 ms is bandwidth-bound (would scale linearly with N) or has a
launch-overhead floor.

Methodology:
- Reload model fresh per variant via `load_model()` (bypasses cache); SwiGLU and
  lm_head patches are in-place.
- Each variant: 1 warm + 5 steady runs of full transcribe (mel → encoder →
  prefill → decode loop → detok). Report mean per-step ms, total ms, RTF, AND
  text equivalence vs baseline (Hamming-style: same string?).
- Decoding stops on natural EOS so token counts may differ across variants if a
  variant changes argmax. Text mismatch is the safety alarm.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_qwen3_asr.audio import SAMPLE_RATE, compute_features, load_audio_np
from mlx_qwen3_asr.decoder import SwiGLU
from mlx_qwen3_asr.generate import (
    GenerationConfig,
    _build_decode_positions,
    _detect_repetition,
    _sample,
    resolve_max_new_tokens,
)
from mlx_qwen3_asr.load_models import load_model
from mlx_qwen3_asr.tokenizer import (
    _TokenizerHolder,
    canonicalize_language,
    parse_asr_output,
)

AUDIO = Path("唐雪梅 知性_02-07.wav")
MODEL_DIR = Path("models/Qwen3-ASR-0.6B")
LANGUAGE = "Chinese"
DTYPE = mx.float16
N_STEADY = 5
LM_HEAD_PROFILE_NS = [151936, 75968, 37984, 16384, 8192, 4096, 1024]


# ---------- Optimization 1: mx.compile wrapper ----------
def make_compiled_step(model):
    """Return a step function wrapped by mx.compile (best effort).

    KVCache mutates state, so we do not pass it as a compile-tracked input.
    Instead we compile only the pure-functional core (embed + transformer +
    lm_head) under the assumption MLX's tracer treats cache writes as side
    effects. If MLX rejects this, we fall back to uncompiled.
    """
    @mx.compile
    def _step(input_ids, position_ids):
        return model.step(
            input_ids=input_ids,
            position_ids=position_ids,
            cache=cache_box[0],
            validate_input_ids=False,
        )

    cache_box = [None]  # mutable holder; closed over by _step

    def step_fn(input_ids, position_ids, cache):
        cache_box[0] = cache
        return _step(input_ids, position_ids)

    return step_fn


# ---------- Optimization 2: fused SwiGLU ----------
class FusedSwiGLU(nn.Module):
    """SwiGLU with gate_proj and up_proj fused into a single (hidden -> 2*inter) Linear."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up_proj(x)
        gate, up = mx.split(gate_up, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)


def patch_fused_swiglu(model) -> int:
    """Replace every TextDecoderLayer.mlp with a FusedSwiGLU. Returns count patched."""
    text_decoder = model.model
    n_patched = 0
    for layer in text_decoder.layers:
        old: SwiGLU = layer.mlp
        hidden = old.gate_proj.weight.shape[1]
        inter = old.gate_proj.weight.shape[0]
        fused = FusedSwiGLU(hidden, inter)
        # Stack gate and up along output dim: shape (2*inter, hidden).
        merged_w = mx.concatenate(
            [old.gate_proj.weight, old.up_proj.weight], axis=0
        ).astype(old.gate_proj.weight.dtype)
        fused.gate_up_proj.weight = merged_w
        fused.down_proj.weight = old.down_proj.weight
        layer.mlp = fused
        n_patched += 1
    mx.eval(model.parameters())
    return n_patched


# ---------- Optimization 3: quantize lm_head ----------
def quantize_lm_head(model, bits: int = 4, group_size: int = 64) -> None:
    """Replace lm_head with a QuantizedLinear (preserves vocab; affects argmax slightly)."""
    old = model.lm_head
    qlin = nn.QuantizedLinear.from_linear(old, group_size=group_size, bits=bits)
    model.lm_head = qlin
    mx.eval(model.parameters())


# ---------- Common decode harness ----------
def run_transcribe(model, tokenizer, audio_np, *, step_fn=None) -> dict:
    """One full transcribe, returns ms breakdown + per-step list + text + tokens."""
    forced_language = canonicalize_language(LANGUAGE)
    t_total = time.perf_counter()

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

    max_new = resolve_max_new_tokens(None, audio_duration_sec=len(audio_np) / 16000)
    cfg = GenerationConfig(max_new_tokens=max_new, temperature=0.0)
    cache = model.create_cache(max_seq_len=int(seq_len + max_new))

    t_prefill = time.perf_counter()
    logits = model.prefill(
        input_ids=input_ids,
        audio_features=audio_features,
        position_ids=position_ids,
        cache=cache,
    )
    token = _sample(logits, 0.0)
    prefill_ms = (time.perf_counter() - t_prefill) * 1e3
    generated = [token]

    next_pos_3d = _build_decode_positions(
        seq_len=seq_len, max_new_tokens=cfg.max_new_tokens, dtype=position_ids.dtype
    )
    per_step_ms: list[float] = []
    for step in range(1, cfg.max_new_tokens):
        if token in cfg.eos_token_ids or _detect_repetition(generated):
            break
        s_t0 = time.perf_counter()
        next_ids = mx.array([[token]])
        next_position_ids = next_pos_3d[:, :, step - 1 : step]
        if step_fn is None:
            logits = model.step(
                input_ids=next_ids,
                position_ids=next_position_ids,
                cache=cache,
                validate_input_ids=False,
            )
        else:
            logits = step_fn(next_ids, next_position_ids, cache)
        token = _sample(logits, 0.0)
        generated.append(token)
        per_step_ms.append((time.perf_counter() - s_t0) * 1e3)

    if generated and generated[-1] in cfg.eos_token_ids:
        generated = generated[:-1]

    raw_text = tokenizer.decode(generated)
    _, text = parse_asr_output(raw_text, user_language=forced_language)

    total_ms = (time.perf_counter() - t_total) * 1e3
    return {
        "total_ms": total_ms,
        "prefill_ms": prefill_ms,
        "per_step_ms": per_step_ms,
        "tokens": generated,
        "text": text,
    }


def measure_variant(name: str, audio_np, *, model, tokenizer, step_fn=None) -> dict:
    print(f"\n[variant] {name}")
    # warm
    warm = run_transcribe(model, tokenizer, audio_np, step_fn=step_fn)
    print(f"  warm: total={warm['total_ms']:.1f}ms gen={len(warm['tokens'])}t")
    runs = []
    for i in range(N_STEADY):
        r = run_transcribe(model, tokenizer, audio_np, step_fn=step_fn)
        runs.append(r)
    totals = [r["total_ms"] for r in runs]
    all_steps = [s for r in runs for s in r["per_step_ms"]]
    text = runs[0]["text"]
    return {
        "name": name,
        "total_ms_mean": statistics.mean(totals),
        "total_ms_min": min(totals),
        "step_ms_mean": statistics.mean(all_steps) if all_steps else 0.0,
        "step_ms_median": statistics.median(all_steps) if all_steps else 0.0,
        "step_ms_min": min(all_steps) if all_steps else 0.0,
        "step_ms_max": max(all_steps) if all_steps else 0.0,
        "n_tokens": len(runs[0]["tokens"]),
        "text": text,
        "tokens": runs[0]["tokens"],
    }


# ---------- lm_head GEMV profile ----------
def profile_lm_head(model) -> dict:
    """Profile lm_head Linear with output dim truncated to varying N.

    Confirms whether the 7 ms is memory-bound (linear with N) or has fixed
    launch overhead.
    """
    print("\n=== lm_head GEMV profile (vary output dim) ===")
    lm = model.lm_head
    W = lm.weight  # (vocab, hidden)
    hidden = W.shape[1]
    print(f"  W shape: {tuple(W.shape)} dtype={W.dtype}")

    h = mx.random.normal((1, 1, hidden)).astype(DTYPE)
    mx.eval(h)

    results = []
    for N in LM_HEAD_PROFILE_NS:
        if N > W.shape[0]:
            continue
        W_sub = W[:N, :].astype(DTYPE)
        mx.eval(W_sub)

        # warm
        for _ in range(2):
            y = h @ W_sub.T
            mx.eval(y)
        # measure
        n_iter = 50
        t0 = time.perf_counter()
        for _ in range(n_iter):
            y = h @ W_sub.T
            mx.eval(y)
        ms = (time.perf_counter() - t0) * 1e3 / n_iter
        # bytes read = N * hidden * 2 (fp16 weight) + tiny activation/output
        bytes_read = N * hidden * 2
        bw_gb_s = (bytes_read / 1e9) / (ms / 1e3)
        results.append((N, ms, bw_gb_s))
        print(f"  N={N:>6}  ms={ms:6.3f}  weight={bytes_read/1e6:7.1f} MB  effective bw={bw_gb_s:6.1f} GB/s")
    return {"results": results}


def text_equiv_marker(target: str, candidate: str) -> str:
    return "✓" if target == candidate else "✗"


def main() -> None:
    print(f"=== exp_decode_optim — {AUDIO.name} ===")
    print(f"model={MODEL_DIR.name}  dtype={DTYPE}  N_steady={N_STEADY}")

    tokenizer = _TokenizerHolder.get(str(MODEL_DIR))
    audio_np = load_audio_np(str(AUDIO), sr=SAMPLE_RATE)
    print(f"audio: {len(audio_np)} samples = {len(audio_np)/16000:.3f}s")

    variants = []

    # --- 1. baseline (also used for lm_head profile) ---
    model, _ = load_model(str(MODEL_DIR), dtype=DTYPE)
    base = measure_variant("baseline", audio_np, model=model, tokenizer=tokenizer)
    variants.append(base)
    baseline_text = base["text"]

    # lm_head profile on baseline model (before any patching).
    lm_profile = profile_lm_head(model)

    # --- 2. +mx.compile only ---
    model, _ = load_model(str(MODEL_DIR), dtype=DTYPE)
    try:
        step_fn = make_compiled_step(model)
        v = measure_variant("+mx.compile", audio_np, model=model, tokenizer=tokenizer, step_fn=step_fn)
        variants.append(v)
    except Exception as e:
        print(f"  mx.compile path failed: {e}")
        variants.append({
            "name": "+mx.compile",
            "total_ms_mean": float("nan"),
            "step_ms_mean": float("nan"),
            "step_ms_median": float("nan"),
            "step_ms_min": float("nan"),
            "step_ms_max": float("nan"),
            "n_tokens": 0,
            "text": f"FAILED: {e}",
            "tokens": [],
            "total_ms_min": float("nan"),
        })

    # --- 3. +fused SwiGLU only ---
    model, _ = load_model(str(MODEL_DIR), dtype=DTYPE)
    n = patch_fused_swiglu(model)
    print(f"  patched {n} SwiGLU layers")
    v = measure_variant("+fused SwiGLU", audio_np, model=model, tokenizer=tokenizer)
    variants.append(v)

    # --- 4. +lm_head int4 only ---
    model, _ = load_model(str(MODEL_DIR), dtype=DTYPE)
    quantize_lm_head(model, bits=4)
    v = measure_variant("+lm_head int4", audio_np, model=model, tokenizer=tokenizer)
    variants.append(v)

    # --- 5. all (compile + swiglu + int4) ---
    model, _ = load_model(str(MODEL_DIR), dtype=DTYPE)
    patch_fused_swiglu(model)
    quantize_lm_head(model, bits=4)
    try:
        step_fn = make_compiled_step(model)
        v = measure_variant("all (compile+swiglu+int4)", audio_np, model=model, tokenizer=tokenizer, step_fn=step_fn)
        variants.append(v)
    except Exception as e:
        print(f"  combined compile path failed: {e}")
        variants.append({
            "name": "all (compile failed; swiglu+int4 only)",
            "total_ms_mean": float("nan"),
            "step_ms_mean": float("nan"),
            "step_ms_median": float("nan"),
            "step_ms_min": float("nan"),
            "step_ms_max": float("nan"),
            "n_tokens": 0,
            "text": f"FAILED: {e}",
            "tokens": [],
            "total_ms_min": float("nan"),
        })
        # Re-measure without compile so we still have the swiglu+int4 datapoint.
        v = measure_variant("swiglu+int4 (no compile)", audio_np, model=model, tokenizer=tokenizer)
        variants.append(v)

    # ---- Summary ----
    print("\n\n=== SUMMARY ===")
    print(f"{'variant':<32}  {'total_ms':>9}  {'step_ms':>10}  {'min/max':>14}  {'tok':>4}  text==base?")
    for v in variants:
        marker = text_equiv_marker(baseline_text, v["text"])
        print(
            f"  {v['name']:<30}  {v['total_ms_mean']:>9.1f}  "
            f"{v['step_ms_mean']:>5.2f} (med {v['step_ms_median']:>4.1f})  "
            f"{v['step_ms_min']:>5.1f}/{v['step_ms_max']:>5.1f}  "
            f"{v['n_tokens']:>4}  {marker}"
        )

    print("\n=== text mismatch detail (if any) ===")
    for v in variants:
        if v["text"] != baseline_text:
            print(f"  {v['name']}:")
            print(f"    base = {baseline_text!r}")
            print(f"    var  = {v['text']!r}")

    print("\n=== lm_head GEMV scaling ===")
    for N, ms, bw in lm_profile["results"]:
        print(f"  N={N:>6}  {ms:6.3f} ms  {bw:6.1f} GB/s")
    if len(lm_profile["results"]) >= 2:
        N1, ms1, _ = lm_profile["results"][0]
        N2, ms2, _ = lm_profile["results"][-1]
        ratio_n = N1 / N2
        ratio_ms = ms1 / max(ms2, 1e-9)
        print(f"  ⇒ N ratio {ratio_n:.1f}× → ms ratio {ratio_ms:.1f}× ({'memory-bound' if ratio_ms > ratio_n * 0.5 else 'launch-bound'})")


if __name__ == "__main__":
    main()
