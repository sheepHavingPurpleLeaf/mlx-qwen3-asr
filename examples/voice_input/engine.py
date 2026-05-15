"""ASR engine wrapper: long-lived Session + warmup."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

import mlx_qwen3_asr as m

log = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "Qwen3-ASR-0.6B"


def _is_chinese(lang: str | None) -> bool:
    if not lang:
        return False
    s = lang.lower()
    return "chin" in s or s.startswith("zh") or s == "mandarin"


class Engine:
    """Wraps a Session, holds warm state, exposes a simple transcribe call.

    Set ``profile=True`` to log a per-phase breakdown
    (mel / encoder / prompt / prefill / decode / detok / itn) on every
    transcribe call. Off by default — the instrumented path duplicates
    pipeline glue and adds minor per-call sync overhead, so it is intended for
    dogfood diagnostics, not production use.

    ``quantize_lm_head=True`` (default) replaces lm_head with an int4
    QuantizedLinear after model load. Validated on the 2476-sample short
    Chinese voice-command set: corpus CER 7.464% → 7.476% (+0.012 pp; 96%
    of samples produce identical text), wall-time 1.23× faster. Disable to
    A/B compare or for non-Chinese / long-form domains where this regression
    has not been validated.
    """

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        language: str = "Chinese",
        itn: bool = True,
        profile: bool = False,
        quantize_lm_head: bool = True,
    ):
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}. "
                "Download Qwen/Qwen3-ASR-0.6B and place it there."
            )
        self.model_dir = model_dir
        self.language = language
        self.itn = itn
        self.profile = profile
        self.quantize_lm_head = quantize_lm_head
        self._session: Optional[m.Session] = None
        self._itn_normalizer: Optional[Any] = None

    def load(self) -> float:
        t0 = time.perf_counter()
        self._session = m.Session(model=str(self.model_dir))
        if self.quantize_lm_head:
            import mlx.core as mx
            import mlx.nn as nn
            model = self._session.model
            model.lm_head = nn.QuantizedLinear.from_linear(
                model.lm_head, group_size=64, bits=4
            )
            mx.eval(model.parameters())
            log.info("lm_head quantized to int4 (group_size=64)")
        if self.itn and _is_chinese(self.language):
            try:
                from itn.chinese.inverse_normalizer import InverseNormalizer
                self._itn_normalizer = InverseNormalizer()
            except Exception as e:
                log.warning(
                    "ITN unavailable (%s); transcripts keep spoken-form numbers. "
                    "Install: brew install openfst && pip install pynini WeTextProcessing importlib_resources",
                    e,
                )
                self._itn_normalizer = None
        return time.perf_counter() - t0

    def warmup(self) -> float:
        if self._session is None:
            self.load()
        # 1 second of near-silence → run one full transcribe so Metal kernels JIT.
        silence = np.zeros(16000, dtype=np.float32)
        # Add tiny noise so the encoder is exercised on non-degenerate input.
        silence[::1000] = 1e-3
        t0 = time.perf_counter()
        try:
            self._session.transcribe(silence, language=self.language)
        except Exception as e:
            log.warning("warmup transcribe raised: %s — kernels likely still warm", e)
        if self._itn_normalizer is not None:
            try:
                self._itn_normalizer.normalize("一点三四")
            except Exception as e:
                log.warning("ITN warmup raised: %s", e)
        return time.perf_counter() - t0

    def transcribe(self, audio_np: np.ndarray, language: Optional[str] = None) -> tuple[str, float]:
        if self._session is None:
            self.load()
        lang = language if language is not None else self.language
        if self.profile:
            return self._transcribe_profiled(audio_np, lang)
        t0 = time.perf_counter()
        kwargs = {}
        if lang and lang.lower() != "auto":
            kwargs["language"] = lang
        result = self._session.transcribe(audio_np, **kwargs)
        text = result.text
        if self._itn_normalizer is not None and _is_chinese(lang):
            try:
                text = self._itn_normalizer.normalize(text)
            except Exception as e:
                log.warning("ITN normalize failed (%s); using raw transcript", e)
        return text, time.perf_counter() - t0

    def _transcribe_profiled(
        self, audio_np: np.ndarray, lang: Optional[str]
    ) -> tuple[str, float]:
        """Phase-instrumented transcribe; logs per-phase ms once per call."""
        import mlx.core as mx
        from mlx_qwen3_asr.audio import compute_features
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

        sess = self._session
        assert sess is not None
        model = sess.model
        tokenizer = _TokenizerHolder.get(str(self.model_dir))
        forced_language = canonicalize_language(lang) if (lang and lang.lower() != "auto") else None
        dtype = mx.float16
        timings: dict[str, float] = {}
        t_call = time.perf_counter()

        t0 = time.perf_counter()
        mel, feature_lens = compute_features(audio_np)
        mx.eval(mel, feature_lens)
        timings["mel"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        audio_features, _ = model.audio_tower(mel.astype(dtype), feature_lens)
        mx.eval(audio_features)
        timings["encoder"] = (time.perf_counter() - t0) * 1e3
        n_audio_tokens = int(audio_features.shape[1])

        t0 = time.perf_counter()
        prompt_tokens = tokenizer.build_prompt_tokens(
            n_audio_tokens=n_audio_tokens, language=forced_language, context=""
        )
        input_ids = mx.array([prompt_tokens])
        seq_len = input_ids.shape[1]
        positions = mx.arange(seq_len)[None, :]
        position_ids = mx.stack([positions, positions, positions], axis=1)
        mx.eval(input_ids, position_ids)
        timings["prompt"] = (time.perf_counter() - t0) * 1e3

        max_new = resolve_max_new_tokens(None, audio_duration_sec=len(audio_np) / 16000)
        cfg = GenerationConfig(max_new_tokens=max_new, temperature=0.0)
        cache = model.create_cache(max_seq_len=int(seq_len + max_new))

        t0 = time.perf_counter()
        logits = model.prefill(
            input_ids=input_ids,
            audio_features=audio_features,
            position_ids=position_ids,
            cache=cache,
        )
        mx.eval(logits)
        timings["prefill"] = (time.perf_counter() - t0) * 1e3

        token = _sample(logits, cfg.temperature)
        generated = [token]
        next_pos_3d = _build_decode_positions(
            seq_len=seq_len, max_new_tokens=cfg.max_new_tokens, dtype=position_ids.dtype
        )
        t0 = time.perf_counter()
        for step in range(1, cfg.max_new_tokens):
            if token in cfg.eos_token_ids or _detect_repetition(generated):
                break
            next_ids = mx.array([[token]])
            logits = model.step(
                input_ids=next_ids,
                position_ids=next_pos_3d[:, :, step - 1 : step],
                cache=cache,
                validate_input_ids=False,
            )
            token = _sample(logits, cfg.temperature)
            generated.append(token)
        timings["decode_loop"] = (time.perf_counter() - t0) * 1e3

        if generated and generated[-1] in cfg.eos_token_ids:
            generated = generated[:-1]

        t0 = time.perf_counter()
        raw_text = tokenizer.decode(generated)
        _, text = parse_asr_output(raw_text, user_language=forced_language)
        timings["detok"] = (time.perf_counter() - t0) * 1e3

        timings["itn"] = 0.0
        if self._itn_normalizer is not None and _is_chinese(lang):
            t0 = time.perf_counter()
            try:
                text = self._itn_normalizer.normalize(text)
            except Exception as e:
                log.warning("ITN normalize failed (%s); using raw transcript", e)
            timings["itn"] = (time.perf_counter() - t0) * 1e3

        elapsed = time.perf_counter() - t_call
        breakdown = "  ".join(f"{k}={v:.1f}" for k, v in timings.items())
        log.info(
            "[profile] gen=%dt prompt=%d audio_tok=%d  %s  total=%.1fms",
            len(generated),
            seq_len,
            n_audio_tokens,
            breakdown,
            elapsed * 1e3,
        )
        return text, elapsed
