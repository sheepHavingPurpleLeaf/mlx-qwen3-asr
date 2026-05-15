"""Task 5 — CER baseline on test_cases/ for the lm_head-int4 comparison.

Test set: 2476 short Chinese voice-command utterances (smart-home / IoT domain).
References: spoken form, no Arabic numerals, no punctuation.
Expected runtime: ~6-9 min on 0.6B fp16, M-series Mac.

Procedure:
- Load model once, force language="Chinese", no ITN, no timestamps.
- Transcribe each sample, normalize hyp (strip punctuation + whitespace).
- Char-level Levenshtein → CER per sample.
- Aggregate: corpus CER = total_edits / total_ref_chars.
- Save per-sample JSON (uid, ref, hyp, edits, ref_len, sec, rtf) for diff
  against the int4 variant.

Two CLI flags:
  --variant {baseline, lm_head_int4}    (default baseline)
  --limit N                              (process first N samples; default all)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import mlx.core as mx

from mlx_qwen3_asr.audio import SAMPLE_RATE, load_audio_np
from mlx_qwen3_asr.load_models import load_model
from mlx_qwen3_asr.transcribe import _transcribe_loaded_components
from mlx_qwen3_asr.tokenizer import _TokenizerHolder

REPO = Path(__file__).resolve().parents[1]
TEST_DIR = REPO / "test_cases"
MODEL_DIR = REPO / "models" / "Qwen3-ASR-0.6B"
LANGUAGE = "Chinese"
DTYPE = mx.float16

# Strip ASCII + Chinese punctuation, collapse whitespace.
_PUNCT_RE = re.compile(
    r"[\s,.;:!?\-\"'()\[\]{}<>/\\|`~@#$%^&*=_+。，、；：！？「」『』（）《》【】〈〉…—·~“”‘’]"
)


def normalize(s: str) -> str:
    return _PUNCT_RE.sub("", s).strip()


def levenshtein(a: str, b: str) -> int:
    """Char-level edit distance. O(len(a)*len(b))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,           # deletion
                cur[j - 1] + 1,        # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = cur
    return prev[-1]


def load_test_set(test_dir: Path) -> list[tuple[str, Path, str]]:
    """Returns list of (uid, wav_path, ref_text)."""
    scp = {}
    for line in (test_dir / "wav.scp").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        uid, rel = line.split(maxsplit=1)
        scp[uid] = test_dir / rel
    refs = {}
    for line in (test_dir / "text").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        uid = parts[0]
        ref = parts[1] if len(parts) > 1 else ""
        refs[uid] = ref
    pairs = []
    for uid, path in scp.items():
        if uid in refs:
            pairs.append((uid, path, refs[uid]))
    pairs.sort()  # deterministic order
    return pairs


def quantize_lm_head(model, bits: int = 4, group_size: int = 64) -> None:
    import mlx.nn as nn
    qlin = nn.QuantizedLinear.from_linear(model.lm_head, group_size=group_size, bits=bits)
    model.lm_head = qlin
    mx.eval(model.parameters())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline", choices=["baseline", "lm_head_int4"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default: scripts/results/cer_<variant>.json)")
    args = ap.parse_args()

    out_path = args.out or REPO / "scripts" / "results" / f"cer_{args.variant}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== exp_cer_baseline — variant={args.variant} ===")
    print(f"model={MODEL_DIR.name}  dtype={DTYPE}  language={LANGUAGE}  ITN=off  timestamps=off")

    pairs = load_test_set(TEST_DIR)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"loaded {len(pairs)} samples from {TEST_DIR}")
    print(f"output: {out_path}")

    # Load model + tokenizer.
    print("\n[loading model]")
    t0 = time.perf_counter()
    model, _ = load_model(str(MODEL_DIR), dtype=DTYPE)
    if args.variant == "lm_head_int4":
        quantize_lm_head(model, bits=4)
        print("  lm_head quantized to int4 (group_size=64)")
    tokenizer = _TokenizerHolder.get(str(MODEL_DIR))
    print(f"  loaded in {time.perf_counter() - t0:.2f}s")

    # Warmup with one short sample so Metal kernels JIT.
    print("[warmup]")
    audio0 = load_audio_np(str(pairs[0][1]), sr=SAMPLE_RATE)
    _ = _transcribe_loaded_components(
        audio_np=audio0, model_obj=model, tokenizer=tokenizer, dtype=DTYPE,
        draft_model_obj=None, context="", language=LANGUAGE, aligner=None,
        return_timestamps=False, diarization_config=None, return_chunks=False,
        max_new_tokens=None, num_draft_tokens=4, verbose=False, on_progress=None,
    )

    # Run.
    print(f"\n[transcribing {len(pairs)} samples]")
    results = []
    total_edits = 0
    total_ref_chars = 0
    total_audio_sec = 0.0
    total_wall_sec = 0.0
    t_start = time.perf_counter()
    progress_every = max(50, len(pairs) // 40)

    for i, (uid, wav_path, ref) in enumerate(pairs):
        try:
            audio = load_audio_np(str(wav_path), sr=SAMPLE_RATE)
            audio_sec = len(audio) / SAMPLE_RATE
            t0 = time.perf_counter()
            result = _transcribe_loaded_components(
                audio_np=audio, model_obj=model, tokenizer=tokenizer, dtype=DTYPE,
                draft_model_obj=None, context="", language=LANGUAGE, aligner=None,
                return_timestamps=False, diarization_config=None, return_chunks=False,
                max_new_tokens=None, num_draft_tokens=4, verbose=False, on_progress=None,
            )
            wall = time.perf_counter() - t0
            hyp = result.text
        except Exception as e:
            wall = 0.0
            audio_sec = 0.0
            hyp = ""
            print(f"  [error] {uid}: {e}")

        ref_n = normalize(ref)
        hyp_n = normalize(hyp)
        edits = levenshtein(ref_n, hyp_n)
        ref_len = len(ref_n)
        sample_cer = edits / max(ref_len, 1)

        total_edits += edits
        total_ref_chars += ref_len
        total_audio_sec += audio_sec
        total_wall_sec += wall

        results.append({
            "uid": uid,
            "ref": ref,
            "hyp": hyp,
            "ref_norm": ref_n,
            "hyp_norm": hyp_n,
            "edits": edits,
            "ref_len": ref_len,
            "cer": sample_cer,
            "audio_sec": round(audio_sec, 3),
            "wall_sec": round(wall, 3),
        })

        if (i + 1) % progress_every == 0 or (i + 1) == len(pairs):
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            eta = (len(pairs) - (i + 1)) / max(rate, 1e-6)
            corpus_cer = total_edits / max(total_ref_chars, 1)
            print(
                f"  [{i+1:>5}/{len(pairs)}] "
                f"running CER={corpus_cer*100:.3f}%  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                f"({rate:.1f} samp/s)"
            )

    # Aggregate.
    corpus_cer = total_edits / max(total_ref_chars, 1)
    rtf = total_wall_sec / max(total_audio_sec, 1e-9)
    sample_cers = [r["cer"] for r in results if r["ref_len"] > 0]
    sample_cers_sorted = sorted(sample_cers)
    n_perfect = sum(1 for c in sample_cers if c == 0)
    n_with_error = len(sample_cers) - n_perfect

    summary = {
        "variant": args.variant,
        "n_samples": len(results),
        "total_audio_sec": round(total_audio_sec, 3),
        "total_wall_sec": round(total_wall_sec, 3),
        "rtf": round(rtf, 4),
        "total_edits": total_edits,
        "total_ref_chars": total_ref_chars,
        "corpus_cer": round(corpus_cer, 6),
        "n_perfect": n_perfect,
        "n_with_error": n_with_error,
        "sample_cer_mean": round(statistics.mean(sample_cers), 6) if sample_cers else 0,
        "sample_cer_median": round(statistics.median(sample_cers), 6) if sample_cers else 0,
        "sample_cer_p90": round(sample_cers_sorted[int(0.9 * len(sample_cers_sorted))], 6) if sample_cers else 0,
        "sample_cer_p99": round(sample_cers_sorted[int(0.99 * len(sample_cers_sorted))], 6) if sample_cers else 0,
    }

    # Save full results.
    payload = {"summary": summary, "samples": results}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved → {out_path}")

    # Print summary.
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:<20}  {v}")

    # Top-10 worst.
    worst = sorted(results, key=lambda r: -r["cer"])[:10]
    print("\n=== top-10 worst CER ===")
    for r in worst:
        print(f"  CER={r['cer']*100:>6.1f}%  ref={r['ref_norm']!r}")
        print(f"                  hyp={r['hyp_norm']!r}")


if __name__ == "__main__":
    main()
