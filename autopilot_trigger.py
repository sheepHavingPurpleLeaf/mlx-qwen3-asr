# coding=utf-8
"""autoPilot retriever: trigger / accept / bias context.

Stateless helpers consumed by :mod:`unified_two_pass`. No pipeline logic here;
the pipeline owns audio encoding and pass-1, this module only owns the
domain-specific rules.

Design (see also :mod:`carcontrol_trigger` for the same shape applied to
carControl):

  Trigger is BROAD — fires on direct autoPilot terms AND on common ASR
  mishears (副驾驶 for 辅助驾驶, 导航 for 领航, 循环 for 巡航). False positives
  are essentially free because the accept gate kills them.

  Accept is STRICT — pass-2 must introduce an autoPilot CANONICAL that
  pass-1 did NOT contain. This rejects prompt-leakage cases where pass-2
  just prepends "打开" to text that already had the canonical, or echoes
  the bias prompt wholesale.
"""
from __future__ import annotations

import re


CONTEXT = (
    "打开 自适应巡航 ACC 智能巡航 IACC LCC 领航辅助 NCA 智驾领航辅助 "
    "自动驾驶 智能驾驶 辅助驾驶 自动泊车 APA 智能泊车 智能停车 停车助手"
)

# Broad triggers — literal autoPilot terms + common mishears.
_TRIGGER_CN = (
    "巡航", "领航", "泊车", "智驾",
    "辅助驾驶", "自动驾驶", "智能驾驶",
    "车道辅助", "车道保持", "车道巡航",
    "副驾驶",      # mishear of 辅助驾驶
    "导航",        # mishear of 领航 (fires on real navi too — accept gate filters)
    "循环",        # mishear of 巡航
    "车道循环",    # explicit form of the 巡航 ↔ 循环 mishear
)
_TRIGGER_EN = ("ACC", "IACC", "NCA", "LCC", "APA")

# Strict canonicals — pass-2 output must contain one of these AND pass-1 must
# not. (Keep narrower than triggers; mishears like 副驾驶 are not canonicals.)
_CANONICAL_CN = ("巡航", "领航", "泊车", "智驾", "辅助驾驶")
_CANONICAL_EN = ("ACC", "IACC", "NCA", "LCC", "APA")

_TRIGGER_EN_RE = re.compile(r"\b(?:" + "|".join(_TRIGGER_EN) + r")\b", re.IGNORECASE)
_CANONICAL_EN_RE = re.compile(r"\b(?:" + "|".join(_CANONICAL_EN) + r")\b", re.IGNORECASE)


def trigger_fires(text: str) -> bool:
    """True iff `text` contains any autoPilot trigger (broad)."""
    if not text:
        return False
    for lit in _TRIGGER_CN:
        if lit in text:
            return True
    return bool(_TRIGGER_EN_RE.search(text))


def _canonicals_in(text: str) -> set[str]:
    if not text:
        return set()
    found = {lit for lit in _CANONICAL_CN if lit in text}
    found.update(m.upper() for m in _CANONICAL_EN_RE.findall(text))
    return found


def accepts(pass1_text: str, pass2_text: str) -> bool:
    """True iff pass-2 introduces a canonical that pass-1 didn't have."""
    return bool(_canonicals_in(pass2_text) - _canonicals_in(pass1_text))
