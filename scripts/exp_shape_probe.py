"""Task 0 / Step 2 — instrument the pipeline to dump tensor shapes."""
from __future__ import annotations

from pathlib import Path
import time

import mlx.core as mx
import numpy as np

import mlx_qwen3_asr as m
from mlx_qwen3_asr.audio import compute_features, load_audio_np, SAMPLE_RATE
from mlx_qwen3_asr.tokenizer import _TokenizerHolder, canonicalize_language

AUDIO = Path("唐雪梅 知性_02-07.wav")
MODEL_DIR = Path("models/Qwen3-ASR-0.6B")


def main() -> None:
    sess = m.Session(model=str(MODEL_DIR))
    model = sess.model
    tokenizer = _TokenizerHolder.get(str(MODEL_DIR))

    # 1. Audio load
    audio_np = load_audio_np(str(AUDIO), sr=SAMPLE_RATE)
    print(f"[1] audio_np: shape={audio_np.shape} dtype={audio_np.dtype} duration={len(audio_np)/SAMPLE_RATE:.3f}s")

    # 2. Mel features
    t0 = time.perf_counter()
    mel, feature_lens = compute_features(audio_np)
    mx.eval(mel, feature_lens)
    print(
        f"[2] mel: shape={tuple(mel.shape)} dtype={mel.dtype} "
        f"feature_lens={feature_lens.tolist()} "
        f"(t={(time.perf_counter()-t0)*1e3:.1f} ms)"
    )

    # 3. Audio encoder
    t0 = time.perf_counter()
    audio_features, output_lens = model.audio_tower(mel.astype(mx.float16), feature_lens)
    mx.eval(audio_features, output_lens)
    print(
        f"[3] audio_features: shape={tuple(audio_features.shape)} dtype={audio_features.dtype} "
        f"output_lens={output_lens.tolist()} "
        f"(t={(time.perf_counter()-t0)*1e3:.1f} ms)"
    )

    # 4. Prompt construction
    n_audio_tokens = audio_features.shape[1]
    forced_language = canonicalize_language("Chinese")
    prompt_tokens = tokenizer.build_prompt_tokens(
        n_audio_tokens=n_audio_tokens,
        language=forced_language,
        context="",
    )
    audio_pad = 151676
    n_pads = sum(1 for t in prompt_tokens if t == audio_pad)
    print(
        f"[4] prompt_tokens len={len(prompt_tokens)}  "
        f"audio_pad placeholders={n_pads} (== n_audio_tokens? {n_pads == n_audio_tokens})  "
        f"audio_start present={151669 in prompt_tokens}, audio_end present={151670 in prompt_tokens}"
    )

    # 5. position_ids
    input_ids = mx.array([prompt_tokens])
    seq_len = input_ids.shape[1]
    positions = mx.arange(seq_len)[None, :]
    position_ids = mx.stack([positions, positions, positions], axis=1)
    print(f"[5] input_ids: shape={tuple(input_ids.shape)}  position_ids: shape={tuple(position_ids.shape)}  (B, 3, L)")

    # 6. One forward call to confirm logits shape
    t0 = time.perf_counter()
    logits = model(
        input_ids=input_ids,
        input_features=mel.astype(mx.float16),
        feature_lens=feature_lens,
        position_ids=position_ids,
    )
    mx.eval(logits)
    print(
        f"[6] logits: shape={tuple(logits.shape)} dtype={logits.dtype} "
        f"(prefill+forward t={(time.perf_counter()-t0)*1e3:.1f} ms)"
    )

    # 7. Run full transcribe through Session for token count + decoded text
    t0 = time.perf_counter()
    res = sess.transcribe(str(AUDIO), return_chunks=True)
    print(
        f"[7] transcribe text='{res.text}'  language={res.language}  "
        f"(t={(time.perf_counter()-t0)*1e3:.1f} ms)"
    )
    if res.chunks:
        c = res.chunks[0]
        print(f"    chunk0: generated_tokens={c.get('generated_tokens')} max_new_tokens={c.get('max_new_tokens')} finish_reason={c.get('finish_reason')}")


if __name__ == "__main__":
    main()
