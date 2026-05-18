# coding=utf-8
"""carControl retriever: trigger / accept / bias context.

Same shape as :mod:`autopilot_trigger`. Tuned for the carControl error
patterns observed in the sft_planC_v4 anchor (17 entity-addressable errors
out of 94 total carControl errors).

Failure modes the trigger covers:

  - 副驾按摩 → 附加按摩          (副驾 ↔ 附加, phonetic fu-jia)
  - 全车二十六度 → 前车二十六度  (全车 ↔ 前车)
  - 中控屏居中 → 中控屏追踪      (居中 ↔ 追踪)
  - 取消零重力 → 取消零公里      (零重力 ↔ 零公里 / 零风力)
  - 打开速冷 → 打开速览          (速冷 ↔ 速览)
  - 我说现在运动模式 → 用多么... (运动模式 ↔ 用多么)

Words deliberately NOT included as triggers/canonicals:

  - 制冷 / 智能 — 智能 is too common (智能模式, 智能驾驶, 智能识别...);
                  trigger would over-fire massively.
  - 除霜 / 车窗 — 车窗 is a legitimate carControl term; FP risk.
  - 中控 / 东风 — 东风 might appear as a proper noun.
  - 上电 / 下电 — 下电 has acceptable FP rate but limited improvement signal.

The 17 addressable errors collapse into ~10 high-confidence wins under this
trigger; the rest were skipped due to FP risk per the failure modes above.
"""
from __future__ import annotations


# Bias prompt — kept short (10 entities + leading verb). Long static prompts
# leak into output on short audio (the failed appPageControl 198-word
# experiment showed this is a real failure mode); gating via trigger keeps
# this list off non-carControl utterances.
CONTEXT = (
    "打开 副驾 全车 居中 中控 零重力 速冷 上电 除霜 运动模式 哨塔模式"
)

# Triggers — primarily MISHEAR words (low FP risk, high signal that pass-1
# heard wrong). We also include the canonicals themselves so the trigger
# fires when pass-1 already happens to contain the right word — the accept
# gate harmlessly rejects in that case but the cost is a wasted pass-2 we
# could have avoided. Keeping canonicals out of trigger reduces fire rate.
_TRIGGER_CN = (
    # mishear of 副驾 (fu-jia ~ fu-zhu-jia-shi vs fu-jia-shi)
    "附加",
    # mishear of 全车 (qu-an-che vs qian-che — diff but ASR confuses)
    "前车",
    # mishear of 居中
    "追踪",
    # mishear of 零重力
    "零公里", "零风力",
    # mishear of 速冷
    "速览",
    # mishear of 运动模式
    "用多么",
)

# Canonicals — pass-2 must introduce one of these (and pass-1 must NOT have
# had it) for accept. These ARE legitimate carControl terms — common in
# pass-1 outputs of carControl samples — so the new-only delta is what
# protects against accidentally accepting pass-1==pass-2 cases.
_CANONICAL_CN = (
    "副驾", "全车", "居中", "中控",
    "零重力", "速冷", "上电", "除霜",
    "运动模式", "哨塔模式",
)


def trigger_fires(text: str) -> bool:
    """True iff `text` contains a carControl mishear trigger (broad)."""
    if not text:
        return False
    return any(lit in text for lit in _TRIGGER_CN)


def _canonicals_in(text: str) -> set[str]:
    if not text:
        return set()
    return {lit for lit in _CANONICAL_CN if lit in text}


def accepts(pass1_text: str, pass2_text: str) -> bool:
    """Strict accept: pass-2 introduces a canonical pass-1 didn't have,
    AND the new-canonical count is small (defends against prompt leakage
    that would dump every bias word into the output), AND pass-2 isn't
    pathologically longer than pass-1 (another leakage signature).
    """
    new_canon = _canonicals_in(pass2_text) - _canonicals_in(pass1_text)
    if not new_canon:
        return False
    # Leakage usually injects many bias words at once.
    if len(new_canon) >= 3:
        return False
    # Length-blowup guard. A real carControl fix is local (副驾↔附加 swaps a
    # 2-char span); a leaked prompt blows the length by 10+ chars.
    if len(pass2_text) > len(pass1_text) + 6:
        return False
    return True
