"""Diff baseline vs lm_head_int4 CER results from exp_cer_baseline.py outputs.

Reports:
- Side-by-side summary (corpus CER, n_perfect, RTF)
- Verdict shifts: perfect→wrong, wrong→perfect, wrong-differently, identical
- Samples where int4 strictly degraded (regression list)
- Samples where int4 strictly improved (rare, sanity)
- Per-sample CER delta histogram
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "scripts" / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / f"cer_{name}.json").read_text(encoding="utf-8"))


def main() -> None:
    base = load("baseline")
    q = load("lm_head_int4")
    bs = base["summary"]; qs = q["summary"]

    print("=== summary ===")
    width = 30
    print(f"  {'metric':<{width}}  {'baseline':>14}  {'lm_head_int4':>14}  {'delta':>14}")
    for k in [
        "n_samples", "total_audio_sec", "total_wall_sec", "rtf",
        "total_edits", "total_ref_chars", "corpus_cer",
        "n_perfect", "n_with_error",
        "sample_cer_mean", "sample_cer_median", "sample_cer_p90", "sample_cer_p99",
    ]:
        bv = bs.get(k); qv = qs.get(k)
        if isinstance(bv, (int, float)) and isinstance(qv, (int, float)):
            delta = qv - bv
            print(f"  {k:<{width}}  {bv:>14}  {qv:>14}  {delta:>+14.4f}" if isinstance(bv, float) else
                  f"  {k:<{width}}  {bv:>14}  {qv:>14}  {delta:>+14}")

    # CER ratio.
    if bs["corpus_cer"] > 0:
        rel = (qs["corpus_cer"] - bs["corpus_cer"]) / bs["corpus_cer"] * 100
        print(f"\n  corpus CER relative change: {rel:+.2f}%")
    abs_delta_pp = (qs["corpus_cer"] - bs["corpus_cer"]) * 100
    print(f"  corpus CER absolute change: {abs_delta_pp:+.3f} percentage points")

    # Speedup.
    if qs["total_wall_sec"] > 0 and bs["total_wall_sec"] > 0:
        speedup = bs["total_wall_sec"] / qs["total_wall_sec"]
        print(f"  wall-time speedup (full pipeline): {speedup:.3f}×  ({bs['total_wall_sec']:.1f}s → {qs['total_wall_sec']:.1f}s)")
        print(f"  RTF: {bs['rtf']:.4f} → {qs['rtf']:.4f}")

    # Per-sample diff.
    print("\n=== verdict shifts ===")
    by_uid_b = {s["uid"]: s for s in base["samples"]}
    by_uid_q = {s["uid"]: s for s in q["samples"]}
    common = sorted(set(by_uid_b) & set(by_uid_q))

    same_text = 0
    perfect_to_wrong = []   # baseline correct, int4 wrong
    wrong_to_perfect = []   # baseline wrong, int4 correct
    wrong_differently = []  # both wrong but text differs
    int4_worse_cer = []     # both wrong, int4 has more edits
    int4_better_cer = []    # both wrong, int4 has fewer edits
    deltas = []

    for uid in common:
        b = by_uid_b[uid]; v = by_uid_q[uid]
        if b["hyp_norm"] == v["hyp_norm"]:
            same_text += 1
            continue
        b_ok = b["edits"] == 0
        v_ok = v["edits"] == 0
        if b_ok and not v_ok:
            perfect_to_wrong.append((uid, b, v))
        elif not b_ok and v_ok:
            wrong_to_perfect.append((uid, b, v))
        else:
            wrong_differently.append((uid, b, v))
            if v["edits"] > b["edits"]:
                int4_worse_cer.append((uid, b, v))
            elif v["edits"] < b["edits"]:
                int4_better_cer.append((uid, b, v))
        deltas.append(v["edits"] - b["edits"])

    print(f"  identical hyp text:         {same_text:>5} / {len(common)} ({same_text/len(common)*100:.1f}%)")
    print(f"  perfect → wrong (regression): {len(perfect_to_wrong):>5}")
    print(f"  wrong → perfect (improvement):{len(wrong_to_perfect):>5}")
    print(f"  wrong differently:            {len(wrong_differently):>5}")
    print(f"    of which int4 has MORE edits: {len(int4_worse_cer):>5}")
    print(f"    of which int4 has FEWER edits:{len(int4_better_cer):>5}")
    if deltas:
        net = sum(deltas)
        print(f"  net edit-count delta (int4 - base): {net:+d}  (positive = int4 worse)")
        print(f"  per-sample edit delta: mean={statistics.mean(deltas):+.3f}  "
              f"min={min(deltas):+d}  max={max(deltas):+d}")

    def show_examples(label, items, n=10):
        print(f"\n=== {label} (showing up to {n}) ===")
        for uid, b, v in items[:n]:
            print(f"  ref:  {b['ref_norm']!r}")
            print(f"  base: {b['hyp_norm']!r}  (edits={b['edits']})")
            print(f"  int4: {v['hyp_norm']!r}  (edits={v['edits']})")
            print()

    # Show regressions first (safety alarm).
    show_examples("REGRESSIONS (perfect→wrong)", perfect_to_wrong, n=15)
    show_examples("IMPROVEMENTS (wrong→perfect)", wrong_to_perfect, n=10)
    show_examples("int4 strictly worse (both wrong, more edits)", int4_worse_cer, n=10)


if __name__ == "__main__":
    main()
