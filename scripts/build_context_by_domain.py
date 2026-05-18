# coding=utf-8
"""Build a per-domain context JSON from data/changan_entities.json.

The eval script `local_asr_eval.py` accepts `--context-file <json>` whose
shape is ``{domain_name: context_string}``. When a sample's annotated domain
matches a key, that context string is prepended to the decoder prompt as a
static hotword bias — same mechanism POI two-pass uses, but with no pass-1
extraction step (single-pass static bias).

Strategy: only fill in domains where the gain is likely positive and the
side-effects are bounded.

  - appPageControl: rich set of proprietary app/page/scenario names many of
    which are post-pretrain v1.5 additions the SFT model has likely not seen.
  - autoPilot: short, closed list of cruise/parking abbreviations + Chinese
    expansions (ACC ↔ 自适应巡航 etc.) — extremely targeted.

Domains intentionally left out:
  - navi: two-pass POI biasing (dynamic) already handles it; adding a static
    context here would DISABLE the two-pass path (eval skips two-pass when
    ctx != "").
  - generalControl / unknown / lifeService: open-vocab / short-imperative
    utterances where long static bias dilutes the signal.
  - media / carControl: large entity sets risk dilution; revisit after the
    first round shows the approach is sound.
"""
from __future__ import annotations

import json
from pathlib import Path

# Entities to drop globally — generic directional / structural words the
# decoder already knows. Including them adds zero signal and consumes tokens.
GENERIC_DROP: set[str] = {
    "主驾", "主驾驶", "副驾", "副驾驶", "后排", "前排", "中控",
    "左侧", "右侧", "左前", "右前", "左后", "右后",
    "中间", "前", "后", "左", "右",
    "所有", "全部", "全屋",
    # action verbs (defensive — should be filtered upstream too)
    "打开", "关闭", "暂停", "继续", "播放", "停止", "结束",
}


def _select_for_domain(
    surface_by_domain: dict[str, dict[str, list[str]]],
    domain: str,
    slots: set[str] | None,
    entries: list[dict],
    max_entities: int | None = None,
) -> list[str]:
    """Collect deduped surface forms for `domain`, optionally restricted to
    a specific subset of slot names. Preserves insertion order from the
    entries list so domain-canonical forms appear before aliases.
    """
    if slots is None:
        # Take everything for this domain.
        return [s for s in surface_by_domain.get(domain, {}) if s not in GENERIC_DROP][:max_entities]

    out: list[str] = []
    seen: set[str] = set()
    for e in entries:
        if e["domain"] != domain:
            continue
        if e["slot"] not in slots:
            continue
        for surface in [e["canonical"], *e["aliases"]]:
            if surface in GENERIC_DROP or surface in seen:
                continue
            seen.add(surface)
            out.append(surface)
    if max_entities is not None:
        out = out[:max_entities]
    return out


def build_context_map(entities_json: Path) -> dict[str, str]:
    data = json.loads(entities_json.read_text(encoding="utf-8"))
    entries = data["entries"]
    surface_by_domain = data["surface_by_domain"]

    out: dict[str, str] = {}

    # appPageControl was tried with a 198-word full-domain bias and regressed
    # WER from 3.79 → 22.32 on the 1797-sample eval (see commit history).
    # Root cause: prompt leakage — the decoder echoed the long context list
    # as its hypothesis. Disabled until a tighter retrieval-style approach
    # (top-K dynamic, like POI two-pass) is implemented.

    # carControl was tried with a 12-word curated list focused on proprietary
    # modes + homophone-prone direction words. Result: WER 4.45 → 5.42
    # (+0.97pp regression on 685 samples). Root cause: prompt leakage on
    # SHORT audio commands (1-2s) — the decoder copied the entire context
    # verbatim. 3 catastrophic leakages contributed 85 edits, wiping out
    # the 25 genuine improvements. Disabled — carControl needs dynamic
    # retrieval (pass-1 span detection like POI two-pass), not static bias.

    # ---- autoPilot: cruise/parking abbreviations + Chinese forms. Verb 打开. ----
    autopilot_words = _select_for_domain(
        surface_by_domain, "autoPilot",
        slots={"_synonym"},  # the hard-coded EN/CN synonym group
        entries=entries,
    )
    if autopilot_words:
        out["autoPilot"] = "打开 " + " ".join(autopilot_words)

    return out


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    src = repo / "data" / "changan_entities.json"
    dst = repo / "data" / "changan_context_by_domain.json"

    ctx_map = build_context_map(src)

    dst.write_text(json.dumps(ctx_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {dst}\n")
    for d, ctx in ctx_map.items():
        words = ctx.split()
        print(f"  {d}: {len(words)} words ({len(ctx)} chars)")
        print(f"    {ctx[:160]}{'...' if len(ctx) > 160 else ''}")


if __name__ == "__main__":
    main()
