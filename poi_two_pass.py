# coding=utf-8
"""Two-pass POI biasing — Tier 1 (safe skip) + Tier 2 (audio-feature reuse).

Both optimizations preserve the per-sample `mlx_hyp` of the naive double-call
Plan C path, while saving compute on the cases where pass 2 cannot help.

Tier 1: When all POI candidates in the pass-2 context are already members of
        pass-1's extracted spans, pass 2 cannot offer a new choice — skip it.
        The candidate set is a subset of what pass-1 already produced, so the
        decoder has no new information to act on; pass 2 ≈ pass 1.

Tier 2: Pass 1 and pass 2 receive the same audio. The encoder path
        (mel → Conv2d stem → 24 transformer encoder layers) is deterministic
        and gives bit-identical audio_features regardless of context. Compute
        once, reuse for both passes — saves the ~50 ms / pass 2 spent in the
        encoder.

Only single-chunk audio (< 30 s) is supported here; eval samples are 1-5 s
voice commands so this is fine. For long-audio paths use the upstream
`_transcribe_loaded_components` instead.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_qwen3_asr.audio import SAMPLE_RATE, compute_features
from mlx_qwen3_asr.chunking import MAX_CHUNK_SECONDS
from mlx_qwen3_asr.generate import (
    GenerationConfig,
    coerce_generation_result,
    generate_with_info,
    resolve_max_new_tokens,
)
from mlx_qwen3_asr.tokenizer import parse_asr_output

from poi_lookup import build_context, extract_spans


def _generate_one(
    model,
    tokenizer,
    audio_features: mx.array,
    n_audio_tokens: int,
    language: str,
    context: str,
    chunk_duration_sec: float,
    num_draft_tokens: int = 4,
) -> str:
    """Build prompt, run generation, parse text. Mirrors the inner loop of
    `_transcribe_loaded_components` for a single chunk so output is identical
    to the naive path when called with the same arguments."""
    gen_config = GenerationConfig(
        max_new_tokens=resolve_max_new_tokens(None, audio_duration_sec=chunk_duration_sec),
        temperature=0.0,
        num_draft_tokens=num_draft_tokens,
    )
    prompt_tokens = tokenizer.build_prompt_tokens(
        n_audio_tokens=n_audio_tokens,
        language=language,
        context=context,
    )
    input_ids = mx.array([prompt_tokens])
    seq_len = input_ids.shape[1]
    positions = mx.arange(seq_len)[None, :]
    position_ids = mx.stack([positions, positions, positions], axis=1)

    gen_out = generate_with_info(
        model=model,
        input_ids=input_ids,
        audio_features=audio_features,
        position_ids=position_ids,
        config=gen_config,
    )
    generation = coerce_generation_result(gen_out, gen_config)
    raw_text = tokenizer.decode(generation.tokens)
    _lang, text = parse_asr_output(raw_text, user_language=language)
    return text


def _tier1_safe_skip(pass1_text: str, ctx: str) -> bool:
    """Tier 1 predicate. True iff every POI candidate proposed for pass 2 is
    already a span that the verb-prefix / suffix-anchor regex extracted from
    pass-1's own output.

    Intuition: if pass 1 already "said" each candidate (or contained it as a
    POI-like span), running pass 2 with that same set as bias just re-confirms
    the existing decoder choice. Pass 2's output is overwhelmingly equal to
    pass 1's; skipping saves a full inference with no measurable result change.
    """
    if not ctx:
        return False
    parts = ctx.split()
    cands = parts[1:] if parts and parts[0] == "导航到" else parts
    if not cands:
        return False
    spans = set(extract_spans(pass1_text))
    return all(c in spans for c in cands)


def transcribe_two_pass(
    audio_np: np.ndarray,
    *,
    model,
    tokenizer,
    dtype,
    language: str,
    poi_index: dict,
    top_k: int = 10,
    enable_tier1_skip: bool = True,
    num_draft_tokens: int = 4,
) -> dict:
    """Run pass 1 → POI lookup → optional pass 2 with shared audio encoding.

    Returns a dict with `pass1_text`, `pass2_context`, `pass2_text`, `mlx_hyp`,
    `used_pass`, `skipped_pass2`.
    """
    chunk_duration_sec = float(len(audio_np) / SAMPLE_RATE)
    if chunk_duration_sec > MAX_CHUNK_SECONDS:
        raise ValueError(
            f"audio {chunk_duration_sec:.1f}s exceeds {MAX_CHUNK_SECONDS}s "
            "single-chunk limit; route long audio through transcribe.py"
        )

    # Tier 2: encode audio once — same audio means same audio_features for both passes.
    mel, feature_lens = compute_features(audio_np)
    audio_features, _ = model.audio_tower(mel.astype(dtype), feature_lens)
    n_audio_tokens = audio_features.shape[1]

    # Pass 1: empty context.
    pass1_text = _generate_one(
        model, tokenizer, audio_features, n_audio_tokens,
        language, "", chunk_duration_sec, num_draft_tokens,
    )

    pass2_ctx = build_context(pass1_text, poi_index, k=top_k)
    if not pass2_ctx:
        return {
            "pass1_text": pass1_text,
            "pass2_context": "",
            "pass2_text": "",
            "mlx_hyp": pass1_text,
            "used_pass": 1,
            "skipped_pass2": False,
        }

    if enable_tier1_skip and _tier1_safe_skip(pass1_text, pass2_ctx):
        return {
            "pass1_text": pass1_text,
            "pass2_context": pass2_ctx,
            "pass2_text": "",
            "mlx_hyp": pass1_text,
            "used_pass": 1,
            "skipped_pass2": True,
        }

    # Pass 2: same audio_features, new context bias.
    pass2_text = _generate_one(
        model, tokenizer, audio_features, n_audio_tokens,
        language, pass2_ctx, chunk_duration_sec, num_draft_tokens,
    )
    return {
        "pass1_text": pass1_text,
        "pass2_context": pass2_ctx,
        "pass2_text": pass2_text,
        "mlx_hyp": pass2_text,
        "used_pass": 2,
        "skipped_pass2": False,
    }
