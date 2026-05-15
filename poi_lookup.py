# coding=utf-8
"""POI pinyin index + lookup for two-pass POI biasing.

Build phase (one-time, ~30-60s):
    .venv/bin/python poi_lookup.py --build 2026_poi.txt -o pinyin_index.pkl

Runtime usage:
    from poi_lookup import load_index, build_context
    idx = load_index('pinyin_index.pkl')
    ctx = build_context(pass1_text, idx, k=10)
"""
from __future__ import annotations

import argparse
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

import pypinyin


# --- pinyin normalization ---

def to_pinyin_key(text: str) -> str:
    """Chinese text → space-separated no-tone pinyin string."""
    return " ".join(pypinyin.lazy_pinyin(text, style=pypinyin.Style.NORMAL))


# --- index build / persist ---

def build_index(
    poi_path: Path,
    min_score: int = 10,
    low_score_max_len: int | None = None,
) -> dict[str, list[tuple[str, int]]]:
    """Read 2026_poi.txt → dict[pinyin_key → list[(poi_name, score)]] (sorted by score desc).

    Args:
        min_score: drop POI entries with score below this.
        low_score_max_len: if set, score=5 entries are kept only when name length
            ≤ this value (used to drop the heavy 12/15/18/21/24-char templated
            spike bands that dominate score=5 mass).
    """
    index: dict[str, list[tuple[str, int]]] = defaultdict(list)
    seen_pair: set[tuple[str, str]] = set()
    total = 0
    kept = 0
    skipped_low_score_long = 0
    with poi_path.open(encoding="utf-8") as f:
        for line in f:
            total += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            name, _alias, score_s = parts
            try:
                score = int(score_s)
            except ValueError:
                continue
            if score < min_score:
                continue
            name = name.strip()
            if not name:
                continue
            if (
                low_score_max_len is not None
                and score < 10
                and len(name) > low_score_max_len
            ):
                skipped_low_score_long += 1
                continue
            key = to_pinyin_key(name)
            if (key, name) in seen_pair:
                continue
            seen_pair.add((key, name))
            index[key].append((name, score))
            kept += 1
    for key in index:
        index[key].sort(key=lambda x: (-x[1], x[0]))
    print(f"[build_index] total lines: {total:,}  kept: {kept:,}  (min_score={min_score}, low_score_max_len={low_score_max_len})")
    if skipped_low_score_long:
        print(f"[build_index] skipped (score<10 and name too long): {skipped_low_score_long:,}")
    print(f"[build_index] unique pinyin keys: {len(index):,}")
    return dict(index)


def save_index(index: dict, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path) -> dict[str, list[tuple[str, int]]]:
    with open(path, "rb") as f:
        return pickle.load(f)


# --- POI span extraction from pass-1 transcript ---

# Strategy 1: action-verb anchored (most reliable).
_VERB_SPAN_RE = re.compile(
    r"(?:导航(?:到|去)?|去|到|搜索(?:附近的)?)([一-鿿0-9a-zA-Z]{2,30})"
)

# Strategy 2: POI-suffix anchored — many POIs end in a recognisable suffix.
_POI_SUFFIXES = (
    "广场", "酒店", "宾馆", "饭店", "大厦", "大楼", "商场", "商城", "公园",
    "医院", "学校", "大学", "学院", "中学", "小学", "幼儿园",
    "市场", "超市", "便利店", "商店", "餐厅",
    "车站", "火车站", "汽车站", "高铁站", "机场", "码头", "港口", "服务区",
    "公司", "工厂", "公馆", "小区", "花园", "家园", "新村", "社区",
    "镇", "县", "市", "区", "村", "街道", "胡同",
    "路", "街", "巷", "桥", "口", "门",
    "馆", "院", "楼", "园", "寺", "山", "河", "湖", "海", "塔", "城", "店",
)
_SUFFIX_GROUP = "|".join(_POI_SUFFIXES)
_SUFFIX_SPAN_RE = re.compile(
    rf"([一-鿿]{{2,15}}(?:{_SUFFIX_GROUP}))"
)

# Strip verbose enders from the right edge.
_TRAIL_TRIM = re.compile(r"(附近|周围|了|吗|呢|啊|哦|啦|嘛)$")

# Strip leading action verbs that crept in via the suffix regex (e.g. "去临河市X").
_LEADING_VERB_TRIM = re.compile(r"^(导航(?:到|去)?|去|到|搜索(?:附近的)?)")


def extract_spans(text: str) -> list[str]:
    """Find POI-candidate spans in a transcript using multi-strategy heuristics.

    Strategy order:
      1. verb-prefixed span: "导航到X" → X
      2. POI-suffix-anchored span: "X广场" / "Y酒店"

    Returns unique spans, longest first.
    """
    text = text.strip().rstrip("。，！？.,!?　 ")
    spans: list[str] = []
    seen: set[str] = set()

    def _add(span: str) -> None:
        span = span.strip()
        # Strip leading verbs (the suffix regex sometimes captures them).
        while True:
            new_span = _LEADING_VERB_TRIM.sub("", span)
            if new_span == span:
                break
            span = new_span
        while True:
            new_span = _TRAIL_TRIM.sub("", span)
            if new_span == span:
                break
            span = new_span
        if len(span) >= 2 and span not in seen:
            seen.add(span)
            spans.append(span)

    for m in _VERB_SPAN_RE.finditer(text):
        _add(m.group(1))
    for m in _SUFFIX_SPAN_RE.finditer(text):
        _add(m.group(1))

    spans.sort(key=lambda s: -len(s))
    return spans


# --- lookup: pinyin key → top-K POI candidates ---

def find_candidates(span: str, index: dict, k: int = 10) -> list[str]:
    """Exact pinyin-key lookup with left-trim suffix fallback.

    When the full span has no matching pinyin key (common when ASR includes a
    city prefix like "临河市..." but the POI db only stores "..."), progressively
    trim characters off the left and retry. The first non-empty bucket wins.
    Stops at min length 2.
    """
    out: list[str] = []
    seen: set[str] = set()

    cursor = span
    while len(cursor) >= 2:
        key = to_pinyin_key(cursor)
        bucket = index.get(key, [])
        if bucket:
            for name, _score in bucket:
                if name not in seen:
                    seen.add(name)
                    out.append(name)
                    if len(out) >= k:
                        return out
            return out  # first non-empty bucket wins
        cursor = cursor[1:]
    return out


def build_context(pass1_text: str, index: dict, k: int = 10) -> str:
    """Build a pass-2 context string from pass-1 transcript + POI index.

    Returns empty string when no POI span detected or no candidates found
    (caller should then skip pass 2 and keep pass 1 output).

    Format: "导航到 <poi1> <poi2> ..." — leading verb anchor borrowed from
    任务 7.5 finding (verb templates also bias the decoder, not just POI names).
    """
    spans = extract_spans(pass1_text)
    if not spans:
        return ""
    pois: list[str] = []
    seen: set[str] = set()
    for span in spans:
        for c in find_candidates(span, index, k=k):
            if c not in seen:
                seen.add(c)
                pois.append(c)
    if not pois:
        return ""
    return "导航到 " + " ".join(pois[:k])


# --- CLI: build index ---

def _main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Build pinyin index from a tab-separated POI file")
    b.add_argument("poi_file", type=Path)
    b.add_argument("-o", "--out", type=Path, default=Path("pinyin_index.pkl"))
    b.add_argument("--min-score", type=int, default=10)
    b.add_argument(
        "--low-score-max-len",
        type=int,
        default=None,
        help="Cap name length for score<10 entries (drops templated long names)",
    )
    args = ap.parse_args()

    if args.cmd == "build":
        t0 = time.perf_counter()
        idx = build_index(
            args.poi_file,
            min_score=args.min_score,
            low_score_max_len=args.low_score_max_len,
        )
        t_build = time.perf_counter() - t0
        save_index(idx, args.out)
        t_total = time.perf_counter() - t0
        sz = args.out.stat().st_size / (1024 * 1024)
        print(f"[build] index built in {t_build:.1f}s, saved to {args.out} ({sz:.1f} MB)")
        print(f"[build] total elapsed {t_total:.1f}s")

        # quick sanity checks
        print("\n[sanity check]")
        for q in ["育红东街", "玉虹东街", "齐齐哈尔", "北京西站", "盒马鲜生", "河马先生"]:
            cands = find_candidates(q, idx, k=5)
            print(f"  {q!r} → {cands}")


if __name__ == "__main__":
    _main()
