# coding=utf-8
"""Unified two-pass pipeline: encode audio + run pass-1 ONCE, then try each
retriever in priority order. Replaces the earlier setup where every retriever
(autoPilot, POI) re-encoded audio and re-ran pass-1 independently.

Pipeline:

    encode(audio)               <- once
    pass-1 (empty ctx)          <- once
        ↓
    try autoPilot retriever:    trigger? → pass-2 (autoPilot ctx) → accept?
    try carControl retriever:   trigger? → pass-2 (carControl ctx) → accept?
    try POI retriever:          spans found? → pass-2 (POI top-K ctx)
                                (POI uses Tier 1 safe-skip when context ⊂ pass-1 spans)
        ↓
    return whichever retriever accepted; otherwise pass-1

Priority order autoPilot → carControl → POI matches the failure-mode coverage
those domains target. Wall-time is dominated by encoder + decoder pass-1; the
optional pass-2's are gated by their triggers so they only run when their
domain's signature appears in pass-1.

Reuses Tier 2 audio-feature sharing and Tier 3 speculative pass-2 from the
existing POI two-pass implementation.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_qwen3_asr.audio import SAMPLE_RATE, compute_features
from mlx_qwen3_asr.chunking import MAX_CHUNK_SECONDS

import autopilot_trigger
import carcontrol_trigger
from poi_lookup import build_context as _build_poi_context
from poi_two_pass import (
    _generate_one,
    _generate_pass2_speculative,
    _tier1_safe_skip,
)


def transcribe_unified(
    audio_np: np.ndarray,
    *,
    model,
    tokenizer,
    dtype,
    language: str,
    poi_index: dict | None = None,
    poi_top_k: int = 10,
    enable_autopilot: bool = True,
    enable_carcontrol: bool = True,
    enable_tier1_skip: bool = True,
    enable_tier3_speculative: bool = False,
    num_draft_tokens: int = 4,
) -> dict:
    """Run the unified retrieve-and-rerun pipeline. Single-chunk only (<30s).

    Returns a dict:

      pass1_text       — what pass 1 (empty ctx) produced
      pass2_text       — pass 2 output, or "" if no retriever fired
      mlx_hyp          — final text (= pass2_text when accepted, else pass1_text)
      used_retriever   — 'autopilot' / 'carcontrol' / 'poi' / '' (none)
      pass2_context    — the context string fed to pass 2 (for debugging)
      autopilot_fired  — autoPilot trigger evaluated and matched pass-1
      carcontrol_fired — carControl trigger evaluated and matched pass-1
      poi_fired        — POI build_context returned a non-empty string
      tier1_skipped    — POI's Tier 1 safe-skip determined pass-2 unnecessary
    """
    chunk_duration_sec = float(len(audio_np) / SAMPLE_RATE)
    if chunk_duration_sec > MAX_CHUNK_SECONDS:
        raise ValueError(
            f"audio {chunk_duration_sec:.1f}s exceeds {MAX_CHUNK_SECONDS}s; "
            "long audio must route through transcribe.py"
        )

    # ---- Audio encoding (once) ----
    mel, feature_lens = compute_features(audio_np)
    audio_features, _ = model.audio_tower(mel.astype(dtype), feature_lens)
    n_audio_tokens = audio_features.shape[1]

    # ---- Pass 1 (once, empty context) ----
    # Always keep tokens when Tier 3 is on — pass 2 may want them as draft.
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

    # Common return shape — populated as we go.
    out = {
        "pass1_text": pass1_text,
        "pass2_text": "",
        "mlx_hyp": pass1_text,
        "used_retriever": "",
        "pass2_context": "",
        "autopilot_fired": False,
        "carcontrol_fired": False,
        "poi_fired": False,
        "tier1_skipped": False,
    }

    def _run_pass2(ctx: str) -> str:
        """Pass 2 with the given context, reusing audio_features."""
        if enable_tier3_speculative and pass1_tokens:
            return _generate_pass2_speculative(
                model, tokenizer, audio_features, n_audio_tokens,
                language, ctx, chunk_duration_sec, pass1_tokens, num_draft_tokens,
            )
        return _generate_one(
            model, tokenizer, audio_features, n_audio_tokens,
            language, ctx, chunk_duration_sec, num_draft_tokens,
        )

    # ---- autoPilot retriever ----
    if enable_autopilot and autopilot_trigger.trigger_fires(pass1_text):
        out["autopilot_fired"] = True
        pass2_text = _run_pass2(autopilot_trigger.CONTEXT)
        if autopilot_trigger.accepts(pass1_text, pass2_text):
            out.update({
                "pass2_text": pass2_text,
                "mlx_hyp": pass2_text,
                "used_retriever": "autopilot",
                "pass2_context": autopilot_trigger.CONTEXT,
            })
            return out

    # ---- carControl retriever ----
    if enable_carcontrol and carcontrol_trigger.trigger_fires(pass1_text):
        out["carcontrol_fired"] = True
        pass2_text = _run_pass2(carcontrol_trigger.CONTEXT)
        if carcontrol_trigger.accepts(pass1_text, pass2_text):
            out.update({
                "pass2_text": pass2_text,
                "mlx_hyp": pass2_text,
                "used_retriever": "carcontrol",
                "pass2_context": carcontrol_trigger.CONTEXT,
            })
            return out

    # ---- POI retriever (unchanged behavior from poi_two_pass) ----
    if poi_index is not None:
        poi_ctx = _build_poi_context(pass1_text, poi_index, k=poi_top_k)
        if poi_ctx:
            out["poi_fired"] = True
            if enable_tier1_skip and _tier1_safe_skip(pass1_text, poi_ctx):
                out.update({
                    "pass2_context": poi_ctx,
                    "tier1_skipped": True,
                    # mlx_hyp stays pass1_text
                })
                return out
            pass2_text = _run_pass2(poi_ctx)
            out.update({
                "pass2_text": pass2_text,
                "mlx_hyp": pass2_text,
                "used_retriever": "poi",
                "pass2_context": poi_ctx,
            })
            return out

    return out
