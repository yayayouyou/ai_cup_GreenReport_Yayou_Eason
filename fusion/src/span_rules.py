"""Span-based post-processing rules, ported verbatim from GreenReport
(pipeline/evaluation/predict_test_realigned.py).

These run on the Claude-extracted spans (promise_string / evidence_string), NOT on
the noisy full `data` paragraph — that clean input is what makes them reliable
(our data-only versions in src/rules.py had to disable most branches).

predict_timeline_rule / predict_quality_rule expect a dict r with keys:
  promise_status, evidence_status, promise_string, evidence_string, data
"""
import re

REFERENCE_YEAR = 2024
PAST_DONE_KW = ['已建立','已導入','已取得','已通過','已完成','已成立','已認證','已實施','已執行','已推動','已開始','已達成','已簽署','已加入','已發行','已發布','已上線','已揭露','已參與','已制訂','已制定','已實行','已採用','已修訂','已新增','已調整','已優化']
OPERATIONAL_KW = ['每年','年度','定期','常態','常設','例行','逐年','逐月','每季','每月','不定期','即時','常年','長期維持','本年度','本期','當期']
PAST_VERB_KW = ['導入','建立','取得','通過','完成','成立','推出','獲得','簽署','發行','發布','上線','揭露','參與','制訂','制定','修訂','採用']
LONG_TERM_KW = ['長期','中長期','未來 5 年','未來五年','未來十年','永續經營','永續發展願景','長遠','持續推動長期']
NET_ZERO_KW = ['Net Zero','net zero','淨零','SBTi','Science Based Targets','科學基礎','碳中和','溫室氣體淨零']
NOT_CLEAR_KW = ['積極','持續','努力','加強','提升','優化','強化']


def has_specific_numbers(t):
    return bool(re.search(r'\d+\.\d+|\d{3,}|\d+\s*%|\d+\s*％', t or ''))


def predict_timeline_rule(r, fallback=True):
    if r.get('promise_status') == 'No':
        return 'N/A'
    ps = r.get('promise_string', '') or ''
    es = r.get('evidence_string', '') or ''
    text = ps + ' ' + es
    if not (ps or es):
        text = r.get('data', '')
    years = [int(y) for y in re.findall(r'\b(20\d{2})\b', text)]
    fy = [y for y in years if y > REFERENCE_YEAR]
    py = [y for y in years if y <= REFERENCE_YEAR]
    if fy:
        d = max(fy) - REFERENCE_YEAR
        return 'within_2_years' if d <= 2 else 'between_2_and_5_years' if d <= 5 else 'more_than_5_years'
    if any(k in text for k in NET_ZERO_KW) and not any(k in text for k in PAST_DONE_KW):
        return 'more_than_5_years'
    if any(k in text for k in LONG_TERM_KW) and not any(k in text for k in PAST_DONE_KW):
        return 'more_than_5_years'
    if any(k in text for k in PAST_DONE_KW):
        return 'already'
    if any(k in text for k in OPERATIONAL_KW):
        return 'already'
    if py and any(k in text for k in PAST_VERB_KW):
        return 'already'
    return 'between_2_and_5_years' if fallback else None  # fallback clobbers; off = only confident branches


def predict_quality_rule(r, conf_only=False):
    if r.get('promise_status') == 'No':
        return 'N/A'
    if r.get('evidence_status') in ('No', 'N/A'):
        return 'N/A'
    es = r.get('evidence_string', '') or ''
    if not es:
        return None
    if '｜' in es:
        segs = [s.strip() for s in es.split('｜') if s.strip()]
        if any(len(s) < 10 for s in segs):
            return 'Not Clear'
    if has_specific_numbers(es):       # high precision: concrete numbers -> Clear
        return 'Clear'
    if conf_only:
        return None                    # skip the noisier length / vague-keyword branches
    if len(es) < 30:
        return 'Not Clear'
    if sum(1 for k in NOT_CLEAR_KW if k in es) >= 2:
        return 'Not Clear'
    return None
