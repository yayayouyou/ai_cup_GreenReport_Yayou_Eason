"""Rule-based post-processing (Option A: rules read `data`, test-realistic).

These are deterministic, spec-derived rules applied AFTER the model predicts.
They read only `data` (available at test time) — never gold promise_string /
evidence_string — so they transfer directly to the test set.

Each rule returns a concrete label only on a HIGH-CONFIDENCE branch, and returns
None on its low-confidence fallback so the model's prediction is kept. This makes
the rule a precision-oriented override rather than a blunt replacement.

Reference year for the competition = 2024 (ESG report release year).
"""

import re

REFERENCE_YEAR = 2024

# Past / completed action — implies the commitment is already in effect.
PAST_DONE_KW = [
    "已建立", "已導入", "已取得", "已通過", "已完成", "已成立", "已認證",
    "已實施", "已執行", "已推動", "已開始", "已達成", "已簽署", "已加入",
    "已發行", "已發布", "已上線", "已揭露", "已參與", "已制訂", "已制定",
    "已實行", "已採用", "已修訂", "已新增", "已調整", "已優化",
]
# Recurring / operational cadence — also implies "already" in effect.
OPERATIONAL_KW = [
    "每年", "年度", "定期", "常態", "常設", "例行", "逐年", "逐月",
    "每季", "每月", "不定期", "即時", "常年", "本年度", "本期", "當期",
]
# Long-horizon vision language.
LONG_TERM_KW = [
    "長期", "中長期", "未來 5 年", "未來五年", "未來十年",
    "永續經營", "永續發展願景", "長遠",
]
# Net-zero / science-based targets — conventionally long-horizon (>5y).
NET_ZERO_KW = [
    "Net Zero", "net zero", "淨零", "SBTi", "Science Based Targets",
    "科學基礎", "碳中和", "溫室氣體淨零",
]

_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_NUM_RE = re.compile(r"\d+\.\d+|\d{3,}|\d+\s*%|\d+\s*％")


def has_specific_numbers(text: str) -> bool:
    """Concrete quantitative content: decimals, 3+ digit figures, or percentages."""
    return bool(_NUM_RE.search(text or ""))


def evidence_rule(data: str) -> str | None:
    """Predict evidence_status from `data` alone.

    Specific numbers are a high-precision (~93% on gold promise=Yes) signal that
    concrete supporting evidence exists. Returns 'Yes' on that branch; None
    otherwise (keep the model's prediction). Only meaningful when promise=Yes —
    the caller must gate on predicted promise.
    """
    if has_specific_numbers(data):
        return "Yes"
    return None


def timeline_rule(data: str, year_only: bool = True) -> str | None:
    """Predict verification_timeline from `data` alone.

    On `data` (Option A), only the explicit-future-year branch is reliable
    (~72% precision, beats the model). The keyword branches (net-zero, long-term,
    past-done, operational) fire on unrelated boilerplate in the full paragraph
    and are net-negative — they need the clean evidence/promise span (Option B)
    to be trustworthy. So `year_only` defaults to True.

    Returns a concrete label only on a confident branch; None otherwise (the
    model's prediction is kept).
    """
    text = data or ""
    years = [int(y) for y in _YEAR_RE.findall(text)]
    future_years = [y for y in years if y > REFERENCE_YEAR]

    if future_years:
        delta = max(future_years) - REFERENCE_YEAR
        if delta <= 2:
            return "within_2_years"
        if delta <= 5:
            return "between_2_and_5_years"
        return "more_than_5_years"

    if year_only:
        return None  # keyword branches are unreliable on full `data`

    has_past_done = any(kw in text for kw in PAST_DONE_KW)
    if any(kw in text for kw in NET_ZERO_KW) and not has_past_done:
        return "more_than_5_years"
    if any(kw in text for kw in LONG_TERM_KW) and not has_past_done:
        return "more_than_5_years"
    if has_past_done:
        return "already"
    if any(kw in text for kw in OPERATIONAL_KW):
        return "already"

    return None  # low confidence -> keep model prediction
