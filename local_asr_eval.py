# coding=utf-8
"""Local MLX ASR evaluation on AI数据音频标注_20260511_1000.

本地版本对应 streaming_vllm_client.py — 用 mlx_qwen3_asr 包替换云端 WebSocket
调用, 离线遍历测试集, 算 WER / ACC 指标.

**双参考列** —— 两个系统输出形式不同, 各自对齐不同的标注列:
  - MLX (ITN=off, 输出口语形式)  →  vs 标注文本
  - 讯飞 (ITN on, 输出书面形式)  →  vs 标注ITN文本
这样数字/字母类话术 ("二十三度" vs "23度") 才公平.

WER 是 char-level edit rate (中文 ASR 习惯写 WER, 数值等同 CER).
ACC 是 utterance-level exact match (normalize 后 hyp == ref 的占比).

输出:
  scripts/results/voicecmd_local/results_<ts>.csv     每条样本一行 (诊断用)
  scripts/results/voicecmd_local/results_<ts>.json    含 summary + samples
  scripts/results/voicecmd_local/summary_<ts>.csv     按 domain 汇总 (主表格)

模式:
  默认: 加载模型 + 跑推理 + 打分.
  --rescore JSON: 只读现有 JSON 的 mlx_hyp/xunfei_hyp 重新打分 (不跑推理).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import mlx.core as mx

from mlx_qwen3_asr.audio import SAMPLE_RATE, load_audio_np
from mlx_qwen3_asr.load_models import load_model
from mlx_qwen3_asr.tokenizer import _TokenizerHolder
from mlx_qwen3_asr.transcribe import _transcribe_loaded_components

REPO = Path(__file__).resolve().parent
DATASET = REPO / "AI数据音频标注_20260511_1000"
MODEL_DIR = REPO / "models" / "Qwen3-ASR-0.6B"
LANGUAGE = "Chinese"
DTYPE = mx.float16

_PUNCT_RE = re.compile(
    r"[\s,.;:!?\-\"'()\[\]{}<>/\\|`~@#$%^&*=_+。，、；：！？「」『』（）《》【】〈〉…—·~“”‘’]"
)
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def normalize(s: str) -> str:
    return _PUNCT_RE.sub("", s or "").strip()


def levenshtein(a: str, b: str) -> int:
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
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _col_idx(ref: str) -> int:
    """Excel cell ref ("B12") → 0-based column index (1)."""
    letters = ""
    for ch in ref:
        if ch.isalpha():
            letters += ch
        else:
            break
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch.upper()) - ord("A") + 1)
    return n - 1


def read_xlsx(path: Path) -> list[dict[str, str]]:
    """Stdlib xlsx reader — sheet1 only, first row as headers, sparse-cell aware."""
    with zipfile.ZipFile(path) as z:
        strings: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            tree = ET.parse(z.open("xl/sharedStrings.xml"))
            for si in tree.getroot().findall(f"{_NS}si"):
                strings.append("".join(t.text or "" for t in si.iter(f"{_NS}t")))
        tree = ET.parse(z.open("xl/worksheets/sheet1.xml"))
        rows_xml = tree.getroot().find(f"{_NS}sheetData").findall(f"{_NS}row")

    rows: list[list[str]] = []
    for r in rows_xml:
        cells = r.findall(f"{_NS}c")
        if not cells:
            continue
        max_idx = max(_col_idx(c.get("r", "A1")) for c in cells)
        row = [""] * (max_idx + 1)
        for c in cells:
            idx = _col_idx(c.get("r", "A1"))
            t = c.get("t")
            v = c.find(f"{_NS}v")
            text = ""
            if v is not None and v.text is not None:
                text = strings[int(v.text)] if t == "s" else v.text
            elif t == "inlineStr":
                is_ = c.find(f"{_NS}is")
                if is_ is not None:
                    text = "".join(tn.text or "" for tn in is_.iter(f"{_NS}t"))
            row[idx] = text
        rows.append(row)

    if not rows:
        return []
    headers = rows[0]
    out: list[dict[str, str]] = []
    for r in rows[1:]:
        padded = r + [""] * (len(headers) - len(r))
        out.append({h: padded[i] for i, h in enumerate(headers)})
    return out


def rescore_from(json_path: Path, dataset: Path, out_dir: Path) -> None:
    """Re-score an existing run with the correct per-system reference column.

    Reads `mlx_hyp` / `xunfei_hyp` from the old JSON and the two ref columns
    from annotation.xlsx, recomputes WER/ACC, writes a new summary CSV (and a
    fresh JSON). Does not touch the model or audio.
    """
    print(f"=== rescore from {json_path.name} ===")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    old_samples = payload["samples"]

    ann_rows = read_xlsx(dataset / "annotation.xlsx")
    ref_map: dict[str, tuple[str, str]] = {}
    for r in ann_rows:
        rid = (r.get("request_id") or "").strip()
        plain = (r.get("标注文本") or "").strip()
        itn = (r.get("标注ITN文本") or "").strip() or plain
        if rid and plain:
            ref_map[rid] = (plain, itn)

    rescored: list[dict] = []
    missing = 0
    for s in old_samples:
        rid = s["request_id"]
        if rid not in ref_map:
            missing += 1
            continue
        ref_plain, ref_itn = ref_map[rid]
        rp_n = normalize(ref_plain)
        ri_n = normalize(ref_itn)
        hyp_n = normalize(s.get("mlx_hyp", ""))
        xf_n = normalize(s.get("xunfei_hyp", ""))

        mlx_edits = levenshtein(rp_n, hyp_n)
        if s.get("xunfei_hyp"):
            xf_edits = levenshtein(ri_n, xf_n)
        else:
            xf_edits = None

        rescored.append(
            {
                "request_id": rid,
                "domain": s["domain"],
                "ref_plain": ref_plain,
                "ref_itn": ref_itn,
                "mlx_hyp": s.get("mlx_hyp", ""),
                "xunfei_hyp": s.get("xunfei_hyp", ""),
                "ref_plain_len": len(rp_n),
                "ref_itn_len": len(ri_n),
                "mlx_edits": mlx_edits,
                "mlx_cer": round(mlx_edits / max(len(rp_n), 1), 6),
                "xunfei_edits": xf_edits,
                "xunfei_cer": round(xf_edits / max(len(ri_n), 1), 6) if xf_edits is not None else None,
            }
        )
    print(f"  rescored {len(rescored)} samples (missing in annotation: {missing})")

    summary_rows, _ = _aggregate_and_emit(rescored, out_dir, suffix="rescore")


def _aggregate_and_emit(results: list[dict], out_dir: Path, suffix: str = "") -> tuple[list[dict], dict]:
    """Bucket per-sample → Total + per-domain rows; write summary CSV + JSON; print table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = f"_{suffix}" if suffix else ""
    summary_csv = out_dir / f"summary{tag}_{ts}.csv"
    json_out = out_dir / f"results{tag}_{ts}.json"

    def _empty() -> dict:
        return {
            "n": 0,
            "mlx_edits": 0, "mlx_ref_chars": 0, "mlx_correct": 0,
            "xf_n": 0, "xf_edits": 0, "xf_ref_chars": 0, "xf_correct": 0,
        }

    total_b = _empty()
    domain_b: dict[str, dict] = defaultdict(_empty)
    for r in results:
        for b in (total_b, domain_b[r["domain"]]):
            b["n"] += 1
            b["mlx_edits"] += r["mlx_edits"]
            b["mlx_ref_chars"] += r["ref_plain_len"]
            if r["mlx_edits"] == 0:
                b["mlx_correct"] += 1
            if r["xunfei_edits"] is not None:
                b["xf_n"] += 1
                b["xf_edits"] += r["xunfei_edits"]
                b["xf_ref_chars"] += r["ref_itn_len"]
                if r["xunfei_edits"] == 0:
                    b["xf_correct"] += 1

    def _row(name: str, b: dict) -> dict:
        return {
            "domain": name, "n": b["n"],
            "xunfei_wer": round(b["xf_edits"] / b["xf_ref_chars"] * 100, 2) if b["xf_ref_chars"] else None,
            "xunfei_acc": round(b["xf_correct"] / b["xf_n"] * 100, 2) if b["xf_n"] else None,
            "mlx_wer": round(b["mlx_edits"] / b["mlx_ref_chars"] * 100, 2) if b["mlx_ref_chars"] else None,
            "mlx_acc": round(b["mlx_correct"] / b["n"] * 100, 2) if b["n"] else None,
        }

    summary_rows = [_row("Total", total_b)]
    for dom, b in sorted(domain_b.items(), key=lambda kv: -kv[1]["n"]):
        summary_rows.append(_row(dom, b))

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "数量", "讯飞 wer", "讯飞 acc", "qwen3-asr-0.6b baseline wer", "qwen3-asr-0.6b baseline acc"])
        for r in summary_rows:
            w.writerow([
                r["domain"], r["n"],
                "" if r["xunfei_wer"] is None else r["xunfei_wer"],
                "" if r["xunfei_acc"] is None else r["xunfei_acc"],
                "" if r["mlx_wer"] is None else r["mlx_wer"],
                "" if r["mlx_acc"] is None else r["mlx_acc"],
            ])

    json_out.write_text(
        json.dumps({"summary": {"rows": summary_rows, "n_samples": len(results)}, "samples": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 84)
    print(f"=== Summary ({len(results)} samples) ===\n")
    h2, h3, h4, h5, h6 = "数量", "讯飞 wer", "讯飞 acc", "qwen3-asr-0.6b wer", "qwen3-asr-0.6b acc"
    print(f"  {'':<18} {h2:>6}  {h3:>9}  {h4:>9}  {h5:>20}  {h6:>20}")
    for r in summary_rows:
        def _fmt(v):
            return f"{v:>9.2f}" if v is not None else "      N/A"
        print(
            f"  {r['domain']:<18} {r['n']:>6}  "
            f"{_fmt(r['xunfei_wer'])}  {_fmt(r['xunfei_acc'])}  "
            f"{_fmt(r['mlx_wer']):>20}  {_fmt(r['mlx_acc']):>20}"
        )
    print(f"\n  summary CSV: {summary_csv}")
    print(f"  JSON:        {json_out}")
    return summary_rows, {"csv": str(summary_csv), "json": str(json_out)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Local MLX ASR eval on AI数据音频标注_20260511_1000")
    ap.add_argument("--limit", type=int, default=0, help="0 = all samples")
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "scripts" / "results" / "voicecmd_local",
    )
    ap.add_argument(
        "--rescore",
        type=Path,
        default=None,
        help="Rescore an existing results JSON without running inference",
    )
    ap.add_argument(
        "--context-file",
        type=Path,
        default=None,
        help="JSON file mapping domain name → context string (hotword bias)",
    )
    ap.add_argument(
        "--two-pass-poi",
        type=Path,
        default=None,
        help="Path to pinyin index pickle for two-pass POI biasing",
    )
    ap.add_argument(
        "--two-pass-domains",
        type=str,
        default="navi",
        help="Comma-separated list of domains to apply two-pass POI biasing (default: navi)",
    )
    ap.add_argument(
        "--top-k-pois",
        type=int,
        default=10,
        help="Number of POI candidates to inject into pass-2 context (default: 10)",
    )
    ap.add_argument(
        "--no-tier1-skip",
        action="store_true",
        help="Disable Tier 1 safe pass-2 skip optimization (only Tier 2 audio reuse remains)",
    )
    ap.add_argument(
        "--tier3-speculative",
        action="store_true",
        help="Enable Tier 3: speculative pass 2 using pass-1 tokens as draft "
             "(batched verify + autoregressive fork; bit-identical to naive pass 2 "
             "for greedy decoding)",
    )
    args = ap.parse_args()

    if args.rescore is not None:
        rescore_from(args.rescore, args.dataset, args.out_dir)
        return

    context_by_domain: dict[str, str] = {}
    if args.context_file is not None:
        context_by_domain = json.loads(args.context_file.read_text(encoding="utf-8"))
        print(f"  per-domain context loaded for: {list(context_by_domain.keys())}")
        for d, ctx in context_by_domain.items():
            print(f"    {d}: {len(ctx)} chars")

    poi_index = None
    two_pass_domains: set[str] = set()
    transcribe_two_pass = None
    if args.two_pass_poi is not None:
        from poi_lookup import load_index
        from poi_two_pass import transcribe_two_pass as _two_pass_fn

        transcribe_two_pass = _two_pass_fn
        print(f"\n[loading POI pinyin index from {args.two_pass_poi}]")
        t0 = time.perf_counter()
        poi_index = load_index(args.two_pass_poi)
        print(
            f"  loaded {len(poi_index):,} pinyin keys in "
            f"{time.perf_counter() - t0:.2f}s"
        )
        two_pass_domains = {d.strip() for d in args.two_pass_domains.split(",") if d.strip()}
        print(
            f"  two-pass enabled for domains: {sorted(two_pass_domains)}  "
            f"top_k={args.top_k_pois}  tier1_skip={not args.no_tier1_skip}  "
            f"tier3_speculative={args.tier3_speculative}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    csv_path = args.out_dir / f"results_{ts}.csv"
    json_path = args.out_dir / f"results_{ts}.json"
    summary_csv_path = args.out_dir / f"summary_{ts}.csv"

    audio_dir = args.dataset / "audio"
    annotation_xlsx = args.dataset / "annotation.xlsx"

    print("=== local_asr_eval ===")
    print(f"model={args.model_dir.name}  dtype={DTYPE}  language={LANGUAGE}  ITN=off")
    print(f"dataset={args.dataset.name}")

    print("\n[loading annotation]")
    rows = read_xlsx(annotation_xlsx)
    print(f"  annotation rows: {len(rows)}")

    samples: list[dict] = []
    skipped_no_ref = 0
    skipped_no_audio = 0
    for row in rows:
        rid = (row.get("request_id") or "").strip()
        ref_plain = (row.get("标注文本") or "").strip()
        ref_itn = (row.get("标注ITN文本") or "").strip() or ref_plain
        if not rid or not ref_plain:
            skipped_no_ref += 1
            continue
        wav = audio_dir / f"asr_{rid}.wav"
        if not wav.exists():
            skipped_no_audio += 1
            continue
        samples.append(
            {
                "request_id": rid,
                "wav": wav,
                "ref_plain": ref_plain,
                "ref_itn": ref_itn,
                "domain": (row.get("标注domain") or "").strip() or "unknown",
                "xunfei": (row.get("讯飞识别文本") or "").strip(),
            }
        )
    print(
        f"  usable samples: {len(samples)}  "
        f"(skipped: no-ref={skipped_no_ref}, no-audio={skipped_no_audio})"
    )

    if args.limit:
        samples = samples[: args.limit]
        print(f"  limited to first {len(samples)}")

    print(f"\n[loading model from {args.model_dir}]")
    t0 = time.perf_counter()
    model, _ = load_model(str(args.model_dir), dtype=DTYPE)
    tokenizer = _TokenizerHolder.get(str(args.model_dir))
    print(f"  loaded in {time.perf_counter() - t0:.2f}s")

    print("[warmup]")
    a0 = load_audio_np(str(samples[0]["wav"]), sr=SAMPLE_RATE)
    _ = _transcribe_loaded_components(
        audio_np=a0, model_obj=model, tokenizer=tokenizer, dtype=DTYPE,
        draft_model_obj=None, context="", language=LANGUAGE, aligner=None,
        return_timestamps=False, diarization_config=None, return_chunks=False,
        max_new_tokens=None, num_draft_tokens=4, verbose=False, on_progress=None,
    )

    print(f"\n[transcribing {len(samples)} samples]")
    results: list[dict] = []
    progress_every = max(20, len(samples) // 20)
    t_start = time.perf_counter()
    total_mlx_edits = total_mlx_ref = 0
    total_xf_edits = total_xf_ref = 0

    for i, s in enumerate(samples):
        ctx = context_by_domain.get(s["domain"], "")
        pass1_text = ""
        pass2_context = ""
        used_pass = 1
        skipped_pass2 = False
        try:
            audio = load_audio_np(str(s["wav"]), sr=SAMPLE_RATE)
            audio_sec = len(audio) / SAMPLE_RATE

            if (
                poi_index is not None
                and s["domain"] in two_pass_domains
                and ctx == ""
            ):
                # Optimized two-pass path: single audio encode (Tier 2) +
                # safe pass-2 skip (Tier 1). Output is preserved vs naive
                # plan-C-v2 path; only compute is reduced.
                result = transcribe_two_pass(
                    audio,
                    model=model,
                    tokenizer=tokenizer,
                    dtype=DTYPE,
                    language=LANGUAGE,
                    poi_index=poi_index,
                    top_k=args.top_k_pois,
                    enable_tier1_skip=not args.no_tier1_skip,
                    enable_tier3_speculative=args.tier3_speculative,
                    num_draft_tokens=4,
                )
                pass1_text = result["pass1_text"]
                pass2_context = result["pass2_context"]
                hyp = result["mlx_hyp"]
                used_pass = result["used_pass"]
                skipped_pass2 = result["skipped_pass2"]
            else:
                # Single-pass (default for non-navi domains, and when a
                # per-domain context-file overrides pass-1 input).
                res1 = _transcribe_loaded_components(
                    audio_np=audio, model_obj=model, tokenizer=tokenizer, dtype=DTYPE,
                    draft_model_obj=None, context=ctx, language=LANGUAGE, aligner=None,
                    return_timestamps=False, diarization_config=None, return_chunks=False,
                    max_new_tokens=None, num_draft_tokens=4, verbose=False, on_progress=None,
                )
                pass1_text = res1.text
                hyp = pass1_text
            err = ""
        except Exception as e:
            audio_sec = 0.0
            hyp = ""
            err = str(e)
            print(f"  [error] {s['request_id']}: {e}")

        rp_n = normalize(s["ref_plain"])
        ri_n = normalize(s["ref_itn"])
        hyp_n = normalize(hyp)
        xf_n = normalize(s["xunfei"])

        mlx_edits = levenshtein(rp_n, hyp_n)
        mlx_cer = mlx_edits / max(len(rp_n), 1)
        if s["xunfei"]:
            xf_edits = levenshtein(ri_n, xf_n)
            xf_cer = xf_edits / max(len(ri_n), 1)
        else:
            xf_edits = None
            xf_cer = None

        total_mlx_edits += mlx_edits
        total_mlx_ref += len(rp_n)
        if xf_edits is not None:
            total_xf_edits += xf_edits
            total_xf_ref += len(ri_n)

        results.append(
            {
                "request_id": s["request_id"],
                "domain": s["domain"],
                "audio_sec": round(audio_sec, 3),
                "ref_plain": s["ref_plain"],
                "ref_itn": s["ref_itn"],
                "mlx_hyp": hyp,
                "pass1_text": pass1_text,
                "pass2_context": pass2_context,
                "used_pass": used_pass,
                "skipped_pass2": skipped_pass2,
                "xunfei_hyp": s["xunfei"],
                "ref_plain_norm": rp_n,
                "ref_itn_norm": ri_n,
                "mlx_hyp_norm": hyp_n,
                "xunfei_hyp_norm": xf_n,
                "ref_plain_len": len(rp_n),
                "ref_itn_len": len(ri_n),
                "mlx_edits": mlx_edits,
                "mlx_cer": round(mlx_cer, 6),
                "xunfei_edits": xf_edits,
                "xunfei_cer": round(xf_cer, 6) if xf_cer is not None else None,
                "error": err,
            }
        )

        if (i + 1) % progress_every == 0 or (i + 1) == len(samples):
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            eta = (len(samples) - (i + 1)) / max(rate, 1e-6)
            print(
                f"  [{i+1:>4}/{len(samples)}]  "
                f"MLX WER={total_mlx_edits / max(total_mlx_ref, 1) * 100:.3f}%  "
                f"XF  WER={total_xf_edits / max(total_xf_ref, 1) * 100:.3f}%  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  ({rate:.1f}/s)"
            )

    elapsed = time.perf_counter() - t_start

    # Bucket: Total + per-domain. Each bucket tracks chars (for WER) and
    # utterance match counts (for ACC). XunFei is counted only on rows that
    # actually have a 讯飞识别文本 value.
    def _empty() -> dict:
        return {
            "n": 0,
            "mlx_edits": 0,
            "mlx_ref_chars": 0,
            "mlx_correct": 0,
            "xf_n": 0,
            "xf_edits": 0,
            "xf_ref_chars": 0,
            "xf_correct": 0,
        }

    total_b = _empty()
    domain_b: dict[str, dict] = defaultdict(_empty)

    for r in results:
        for b in (total_b, domain_b[r["domain"]]):
            b["n"] += 1
            b["mlx_edits"] += r["mlx_edits"]
            b["mlx_ref_chars"] += r["ref_plain_len"]
            if r["mlx_edits"] == 0:
                b["mlx_correct"] += 1
            if r["xunfei_edits"] is not None:
                b["xf_n"] += 1
                b["xf_edits"] += r["xunfei_edits"]
                b["xf_ref_chars"] += r["ref_itn_len"]
                if r["xunfei_edits"] == 0:
                    b["xf_correct"] += 1

    def _row(name: str, b: dict) -> dict:
        return {
            "domain": name,
            "n": b["n"],
            "xunfei_wer": round(b["xf_edits"] / b["xf_ref_chars"] * 100, 2) if b["xf_ref_chars"] else None,
            "xunfei_acc": round(b["xf_correct"] / b["xf_n"] * 100, 2) if b["xf_n"] else None,
            "mlx_wer": round(b["mlx_edits"] / b["mlx_ref_chars"] * 100, 2) if b["mlx_ref_chars"] else None,
            "mlx_acc": round(b["mlx_correct"] / b["n"] * 100, 2) if b["n"] else None,
        }

    summary_rows = [_row("Total", total_b)]
    for dom, b in sorted(domain_b.items(), key=lambda kv: -kv[1]["n"]):
        summary_rows.append(_row(dom, b))

    summary = {
        "n_samples": len(results),
        "elapsed_sec": round(elapsed, 2),
        "model_dir": str(args.model_dir),
        "language": LANGUAGE,
        "rows": summary_rows,
    }

    # Per-sample diagnostic CSV.
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "request_id", "domain", "audio_sec",
                "ref_plain", "ref_itn",
                "mlx_hyp", "xunfei_hyp",
                "ref_plain_len", "ref_itn_len",
                "mlx_edits", "mlx_cer", "xunfei_edits", "xunfei_cer",
                "error",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r["request_id"], r["domain"], r["audio_sec"],
                    r["ref_plain"], r["ref_itn"],
                    r["mlx_hyp"], r["xunfei_hyp"],
                    r["ref_plain_len"], r["ref_itn_len"],
                    r["mlx_edits"], r["mlx_cer"], r["xunfei_edits"], r["xunfei_cer"],
                    r["error"],
                ]
            )

    # Domain summary CSV — matches user's target table layout.
    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "数量", "讯飞 wer", "讯飞 acc", "qwen3-asr-0.6b baseline wer", "qwen3-asr-0.6b baseline acc"])
        for r in summary_rows:
            w.writerow([
                r["domain"], r["n"],
                "" if r["xunfei_wer"] is None else r["xunfei_wer"],
                "" if r["xunfei_acc"] is None else r["xunfei_acc"],
                "" if r["mlx_wer"] is None else r["mlx_wer"],
                "" if r["mlx_acc"] is None else r["mlx_acc"],
            ])

    json_path.write_text(
        json.dumps({"summary": summary, "samples": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Console table — matches the screenshot the user shared.
    print("\n" + "=" * 84)
    print(f"=== Summary ({len(results)} samples, {elapsed:.1f}s) ===\n")
    h1, h2, h3, h4, h5, h6 = "", "数量", "讯飞 wer", "讯飞 acc", "qwen3-asr-0.6b wer", "qwen3-asr-0.6b acc"
    print(f"  {h1:<18} {h2:>6}  {h3:>9}  {h4:>9}  {h5:>20}  {h6:>20}")
    for r in summary_rows:
        def _fmt(v):
            return f"{v:>9.2f}" if v is not None else "      N/A"
        print(
            f"  {r['domain']:<18} {r['n']:>6}  "
            f"{_fmt(r['xunfei_wer'])}  {_fmt(r['xunfei_acc'])}  "
            f"{_fmt(r['mlx_wer']):>20}  {_fmt(r['mlx_acc']):>20}"
        )

    print(f"\n  summary CSV:  {summary_csv_path}")
    print(f"  per-sample CSV: {csv_path}")
    print(f"  JSON:           {json_path}")


if __name__ == "__main__":
    main()
