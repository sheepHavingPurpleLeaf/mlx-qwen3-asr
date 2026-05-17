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
    FINISH_REASON_LENGTH,
    GenerationConfig,
    coerce_generation_result,
    generate_with_info,
    resolve_max_new_tokens,
)
from mlx_qwen3_asr.generate import (
    _build_decode_positions,
    _detect_repetition,
    _finalize_generation_result,
    _sample,
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
    *,
    return_tokens: bool = False,
):
    """Build prompt, run generation, parse text. Mirrors the inner loop of
    `_transcribe_loaded_components` for a single chunk so output is identical
    to the naive path when called with the same arguments.

    When ``return_tokens=True``, returns ``(text, tokens)`` instead of just text;
    tokens are the raw assistant-output token ids (EOS stripped to match the
    upstream `GenerationResult.tokens` contract).
    """
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
    if return_tokens:
        return text, list(generation.tokens)
    return text


def _generate_pass2_speculative(
    model,
    tokenizer,
    audio_features: mx.array,
    n_audio_tokens: int,
    language: str,
    context: str,
    chunk_duration_sec: float,
    draft_tokens: list[int],
    num_draft_tokens: int = 4,
) -> str:
    """Tier-3 speculative pass 2: verify pass-1 tokens in one batched forward
    against pass-2's biased model, then continue autoregressively from the
    first divergence.

    For greedy (temperature=0) decoding this is **bit-identical** to running
    pass 2 standalone, because every accepted token equals pass-2's argmax at
    that position and every forked token is sampled from pass-2's logits the
    same way standalone decoding would. The saving is one batched forward
    pass over K draft tokens instead of K sequential single-token forwards.
    """
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

    max_seq_len = int(seq_len + gen_config.max_new_tokens)
    cache = model.create_cache(max_seq_len=max_seq_len)

    # Pass 2 prefill: same audio_features, biased context.
    logits = model.prefill(
        input_ids=input_ids,
        audio_features=audio_features,
        position_ids=position_ids,
        cache=cache,
    )
    pred_first = _sample(logits, gen_config.temperature)

    # Trim draft of any trailing EOS that pass 1 emitted (we don't want to
    # feed EOS into step_many; if pass 2 wants EOS it will sample it itself).
    draft = list(draft_tokens)

    if not draft or pred_first != draft[0]:
        # Diverge at position 0 — no batched verify possible. Fall through to
        # standard autoregressive generation starting from pred_first.
        generated: list[int] = [pred_first]
    else:
        # First token agrees. Verify draft[1..K-1] in a single batched forward.
        K = len(draft)
        if K == 1:
            generated = [draft[0]]
        else:
            # Feed draft[0..K-1] into step_many. Each verify position i predicts
            # the token AFTER draft[i] — so we need draft[1..K-1] to match the
            # argmax of verify_logits[0..K-2].
            verify_ids = mx.array([draft])
            # Build MRoPE positions for the K new tokens: positions seq_len..seq_len+K-1.
            verify_positions = _build_decode_positions(
                seq_len=seq_len,
                max_new_tokens=K + 1,  # _build_decode_positions returns max_new_tokens-1 positions
                dtype=position_ids.dtype,
            )[:, :, :K]

            verify_logits = model.step_many(
                input_ids=verify_ids,
                position_ids=verify_positions,
                cache=cache,
                validate_input_ids=False,
            )
            # Force evaluation so subsequent .item() reads are cheap.
            mx.eval(verify_logits)
            verify_argmax = mx.argmax(verify_logits, axis=-1)  # shape (1, K)

            diverge_at: int | None = None
            for i in range(K - 1):
                if int(verify_argmax[0, i].item()) != draft[i + 1]:
                    diverge_at = i + 1
                    break

            if diverge_at is None:
                # All draft tokens accepted. The next token (beyond pass-1's
                # last) is pass-2's argmax at the final verify slot.
                generated = list(draft) + [int(verify_argmax[0, K - 1].item())]
            else:
                # Diverge at draft[diverge_at]. Accept draft[0..diverge_at-1],
                # replace with pass-2's pred at the divergence slot, and trim
                # cache to discard the K - diverge_at speculatively added entries.
                forked = int(verify_argmax[0, diverge_at - 1].item())
                cache.trim(K - diverge_at)
                generated = list(draft[:diverge_at]) + [forked]

    # Autoregressive tail (standard): generated[-1] is the most-recent decided
    # token that is NOT yet in cache. We feed it via step() to advance cache,
    # take the new logits, sample, append, repeat — exactly like
    # generate_with_info's loop.
    next_pos_3d = _build_decode_positions(
        seq_len=seq_len,
        max_new_tokens=gen_config.max_new_tokens,
        dtype=position_ids.dtype,
    )

    while len(generated) < gen_config.max_new_tokens:
        token = generated[-1]
        if token in gen_config.eos_token_ids:
            break
        if _detect_repetition(generated):
            break
        next_ids = mx.array([[token]])
        # Cache currently has seq_len + (len(generated) - 1) entries. The slot
        # for `token` is at absolute position seq_len + len(generated) - 1.
        # _build_decode_positions[step-1] gives position seq_len + step - 1
        # when called with seq_len. So we use index = len(generated) - 1.
        idx = len(generated) - 1
        next_position_ids = next_pos_3d[:, :, idx : idx + 1]
        logits = model.step(
            input_ids=next_ids,
            position_ids=next_position_ids,
            cache=cache,
            validate_input_ids=False,
        )
        token = _sample(logits, gen_config.temperature)
        generated.append(token)

    result = _finalize_generation_result(generated, gen_config)
    raw_text = tokenizer.decode(result.tokens)
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
    enable_tier3_speculative: bool = False,
    num_draft_tokens: int = 4,
) -> dict:
    """Run pass 1 → POI lookup → optional pass 2 with shared audio encoding.

    With ``enable_tier3_speculative=True``, pass 2 uses pass-1 tokens as draft
    in a single batched verify pass, falling back to standard autoregressive
    only at the first divergence. For greedy decoding this is bit-identical
    to standalone pass 2 and saves the per-token Python/dispatch overhead on
    every position where pass 1 already produced the right token.

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

    # Pass 1: empty context. When Tier 3 is on we also need pass-1 tokens.
    if enable_tier3_speculative:
        pass1_text, pass1_tokens = _generate_one(
            model, tokenizer, audio_features, n_audio_tokens,
            language, "", chunk_duration_sec, num_draft_tokens,
            return_tokens=True,
        )
    else:
        pass1_text = _generate_one(
            model, tokenizer, audio_features, n_audio_tokens,
            language, "", chunk_duration_sec, num_draft_tokens,
        )
        pass1_tokens = []

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
    if enable_tier3_speculative and pass1_tokens:
        pass2_text = _generate_pass2_speculative(
            model, tokenizer, audio_features, n_audio_tokens,
            language, pass2_ctx, chunk_duration_sec, pass1_tokens, num_draft_tokens,
        )
    else:
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
