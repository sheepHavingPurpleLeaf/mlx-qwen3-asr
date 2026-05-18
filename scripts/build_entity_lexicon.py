# coding=utf-8
"""Extract product-entity lexicon from 长安科技语音垂类语义协议 v1.5.

Parses every NLU sheet for slot-value patterns of the form
    ``EN_NAME：CN_canonical[/alias1/alias2[,alias3]]``
and emits a flat JSON list where each entry carries domain / intent / slot
context plus the canonical Chinese surface form and any aliases.

The same `{canonical, aliases[]}` structure also covers English abbreviations
(ACC, IACC, NCA, LCC...) parsed out of the 智驾 sheet's example-utterance
column where the protocol does NOT formalize them as slot values.

Usage:
    .venv/bin/python scripts/build_entity_lexicon.py \\
        --xlsx 长安科技语音垂类语义协议v1.5.xlsx \\
        --out  data/changan_entities.json

Deferred optimizations (require upstream signal we don't have today):
    - Multi-turn ASR biasing using the 上下文 sheet (891 carControl follow-up
      rows). Needs the previous turn's intent+slot vocabulary to be passed in
      at ASR call time. Defer until the runtime exposes a "prev_turn" hook.
    - Phone-domain FST / contact-list biasing. The phone_num slot is digit
      sequences and contact_name is the user's address book — both must come
      from the device. Defer until upstream wires them through.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from python_calamine import CalamineWorkbook


# Sheet → domain code used in eval anchors.
SHEET_DOMAINS: dict[str, str] = {
    "电话": "phone",
    "多媒体": "media",
    "问答记忆和交互": "memorizeUserInfo",
    "车辆信息": "vehicleInfo",
    "应用和页面设置": "appPageControl",
    "导航": "navi",
    "智驾": "autoPilot",
    "通用域": "generalControl",
    "天气": "weather",
    "小安看世界": "xiaoAnWorldview",
    "车控": "carControl",
    "智能家居": "smartHome",
    "联接精灵": "linkSprite",
    "app控车": "appCarControl",
    "路书": "roadBook",
}

# Slots that are intentionally open-vocabulary — skip extraction, surface as TODO.
OPEN_VOCAB_SLOTS: set[str] = {
    "phone_num", "phone_number", "number",
    "contact_name", "contact_item",
    "poi", "center",
    "custom_name",
    "song_name", "track_name", "artist", "album", "resource_name",
}

# Generic action verbs that already exist in every Chinese ASR vocabulary —
# excluded from biasing because they add noise without helping.
GENERIC_SKIP_CANONICAL: set[str] = {
    "打开", "关闭", "开启", "退出", "返回", "搜索", "查询", "查找",
    "暂停", "继续", "播放", "停止", "结束",
    "上", "下", "左", "右", "中", "前", "后",
}

# Pattern: leading normalization code, then ：or:, then CN payload.
# Code can be either ALL-CAPS English (QQ_MUSIC) or — in change-log-added rows
# — the Chinese canonical itself (酷我音乐：酷我,酷我音乐盒). Both shapes occur.
_NORM_LINE = re.compile(r"^([A-Z][A-Z0-9_]*|[一-鿿0-9A-Za-z+]{2,})\s*[：:]\s*(.+?)\s*$")

# Filter: template placeholders use literal "x"/"X"/"…" — drop them.
_TEMPLATE_RE = re.compile(r"[xX×…]+")


def _split_aliases(payload: str) -> list[str]:
    """Split a CN payload on / , ， 、 | — keeping order, dropping empties.

    The first item is treated as canonical, remainder as aliases.
    """
    parts = re.split(r"[\\/,，、|]", payload)
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        p = p.strip()
        if not p or p in seen:
            continue
        # Drop trailing parenthetical comments like "(AIBOX使用)" / "（最高级）".
        p = re.sub(r"[（(].*?[）)]\s*$", "", p).strip()
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _extract_from_slot_value_cell(cell: str) -> list[tuple[str, str, list[str]]]:
    """Parse a Slot Value cell into [(norm_code, canonical, aliases), ...].

    A cell may contain multiple normalization lines separated by newlines.
    Lines that don't match the EN：CN pattern (e.g. comments like
    "1%-99%（AIBOX使用）") are silently skipped.
    """
    out: list[tuple[str, str, list[str]]] = []
    for line in str(cell).split("\n"):
        m = _NORM_LINE.match(line.strip())
        if not m:
            continue
        code, payload = m.group(1), m.group(2)
        # Drop template-placeholder lines like "评分x以上" / "营业到晚上x".
        if _TEMPLATE_RE.search(payload):
            continue
        aliases = _split_aliases(payload)
        # Also drop aliases that are themselves templates.
        aliases = [a for a in aliases if not _TEMPLATE_RE.search(a)]
        if not aliases:
            continue
        canonical, *rest = aliases
        # If code matches the canonical itself (CN：CN/CN/CN form), the code
        # is redundant — keep only the surface forms, set norm to canonical.
        if code == canonical:
            code = canonical
        out.append((code, canonical, rest))
    return out


def extract_from_sheets(xlsx: Path) -> list[dict]:
    wb = CalamineWorkbook.from_path(str(xlsx))
    entries: list[dict] = []
    # Stable id derived from (domain, slot, code, canonical) to allow dedup.
    seen_keys: set[tuple[str, str, str, str]] = set()

    for sheet_name, domain in SHEET_DOMAINS.items():
        if sheet_name not in wb.sheet_names:
            continue
        ws = wb.get_sheet_by_name(sheet_name)
        data = ws.to_python()

        # Forward-fill intent / slot across merged cells (xlsx merge cells
        # render as blank in calamine — copy down the most-recent non-empty
        # value within the same sheet so we don't lose context).
        last_intent = ""
        last_slot = ""
        for row in data[1:]:
            intent = (row[5] if len(row) > 5 else "") or ""
            slot = (row[6] if len(row) > 6 else "") or ""
            slot_value = (row[8] if len(row) > 8 else "") or ""

            if intent:
                last_intent = intent
            if slot:
                last_slot = slot
            cur_intent = intent or last_intent
            cur_slot = slot or last_slot

            if cur_slot in OPEN_VOCAB_SLOTS:
                continue
            if not slot_value:
                continue

            for code, canonical, aliases in _extract_from_slot_value_cell(slot_value):
                if canonical in GENERIC_SKIP_CANONICAL:
                    continue

                key = (domain, cur_slot, code, canonical)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                entries.append({
                    "canonical": canonical,
                    "aliases": aliases,
                    "norm": code,
                    "domain": domain,
                    "intent": cur_intent,
                    "slot": cur_slot,
                    "sheet": sheet_name,
                })

    # The 应用和页面设置 sheet enumerates third-party app/page names in the
    # 示例泛化说法 column (col 3) inline as "app_name=A/B/C/..." illustrations,
    # NOT as formalized slot-value rows. These are exactly the open-vocabulary
    # proper nouns the SFT model is most likely to mis-recognize — extract
    # them as standalone canonicals (no English norm, no aliases).
    _SLOT_LIST_RE = re.compile(r"\[?(app_name|page_name)=([^\]【】\n]+)\]?")
    ws = wb.get_sheet_by_name("应用和页面设置")
    for row in ws.to_python()[1:]:
        if len(row) < 4 or not row[3]:
            continue
        for m in _SLOT_LIST_RE.finditer(str(row[3])):
            slot = m.group(1)
            for name in m.group(2).split("/"):
                name = name.strip()
                if not name or len(name) < 2 or _TEMPLATE_RE.search(name):
                    continue
                key = ("appPageControl", slot, name, name)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                entries.append({
                    "canonical": name,
                    "aliases": [],
                    "norm": name,
                    "domain": "appPageControl",
                    "intent": "CONTROL_APP" if slot == "app_name" else "CONTROL_PAGE",
                    "slot": slot,
                    "sheet": "应用和页面设置",
                })

    # 智驾 sheet does NOT encode the ACC/IACC/NCA/LCC abbreviations as slot
    # values — they live in the 示例说法 column. Hard-code the small set since
    # there are only ~6 and the protocol enumerates them in plain text.
    autopilot_abbrev_aliases = [
        # (canonical_cn, [aliases including english abbrev], intent, slot)
        ("自适应巡航", ["ACC"], "OP_ACC", "_synonym"),
        ("智能巡航", ["IACC", "LCC"], "OP_IACC", "_synonym"),
        ("领航辅助", ["NCA", "智驾领航辅助"], "OP_NCA", "_synonym"),
        ("自动驾驶", ["智能驾驶", "辅助驾驶"], "OP_AUTO_DRIVE", "_synonym"),
        ("自动泊车", ["APA", "智能泊车", "智能停车", "停车助手"], "OP_AUTO_PARKING", "_synonym"),
    ]
    for canonical, aliases, intent, slot in autopilot_abbrev_aliases:
        key = ("autoPilot", slot, intent, canonical)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        entries.append({
            "canonical": canonical,
            "aliases": aliases,
            "norm": intent,
            "domain": "autoPilot",
            "intent": intent,
            "slot": slot,
            "sheet": "智驾",
        })

    return entries


def _build_surface_index(entries: list[dict]) -> dict[str, dict[str, list[str]]]:
    """Per-domain unique surface form → list of canonicals it can map to.

    Used by biasing: "give me every Chinese surface form the user might say
    in domain X, deduped." Aliases collapse onto their canonical here.
    """
    out: dict[str, dict[str, set[str]]] = {}
    for e in entries:
        bucket = out.setdefault(e["domain"], {})
        for surface in [e["canonical"], *e["aliases"]]:
            bucket.setdefault(surface, set()).add(e["canonical"])
    return {d: {s: sorted(cs) for s, cs in b.items()} for d, b in out.items()}


def write_outputs(entries: list[dict], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v1.5",
        "source": "长安科技语音垂类语义协议v1.5.xlsx",
        "entries": entries,
        "surface_by_domain": _build_surface_index(entries),
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_summary(entries: list[dict]) -> None:
    from collections import Counter
    by_domain = Counter(e["domain"] for e in entries)
    print(f"Total entries (provenance rows): {len(entries)}")
    print("By domain:")
    for d, n in by_domain.most_common():
        print(f"  {d:>20s}  {n}")

    surface_idx = _build_surface_index(entries)
    print("\nUnique surface forms by domain (deduped):")
    for d in by_domain:
        print(f"  {d:>20s}  {len(surface_idx.get(d, {})):>4d}")

    n_aliased = sum(1 for e in entries if e["aliases"])
    total_surfaces = sum(len(b) for b in surface_idx.values())
    print(f"\nEntries with ≥1 alias: {n_aliased} / {len(entries)}")
    print(f"Total unique surface forms across all domains: {total_surfaces}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=Path("长安科技语音垂类语义协议v1.5.xlsx"))
    ap.add_argument("--out", type=Path, default=Path("data/changan_entities.json"))
    args = ap.parse_args()
    entries = extract_from_sheets(args.xlsx)
    write_outputs(entries, args.out)
    print_summary(entries)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    _main()
