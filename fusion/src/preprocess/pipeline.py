"""
Preprocessing pipeline — reads step list from config and applies in order.
Each step is a function in this module: step_<name>(records, **kwargs).
"""


def run_pipeline(records: list[dict], cfg: dict) -> list[dict]:
    import copy
    records = copy.deepcopy(records)
    for step_block in cfg.get("steps", []):
        step_name, kwargs = next(iter(step_block.items()))
        fn = globals().get(f"step_{step_name}")
        if fn is None:
            raise ValueError(f"Unknown preprocess step: {step_name}")
        records = fn(records, **(kwargs or {}))
    return records


# ── Steps ─────────────────────────────────────────────────────────────────────

def step_normalize_esg_type(records: list[dict], sort_labels: bool = True) -> list[dict]:
    for r in records:
        parts = [p.strip() for p in r.get("esg_type", "").split(";") if p.strip()]
        if sort_labels:
            parts = sorted(parts)
        r["esg_type"] = ";".join(parts)
    return records


def step_fill_empty_esg_type(records: list[dict], fill_value: str = "Unknown") -> list[dict]:
    for r in records:
        if not r.get("esg_type"):
            r["esg_type"] = fill_value
    return records


def step_normalize_timeline(records: list[dict], rename: dict | None = None) -> list[dict]:
    if not rename:
        return records
    for r in records:
        vt = r.get("verification_timeline", "")
        r["verification_timeline"] = rename.get(vt, vt)
    return records


def step_clean_text(records: list[dict], field: str = "data") -> list[dict]:
    import re
    for r in records:
        text = r.get(field, "")
        text = re.sub(r"\*{1,2}([^*]*?)\*{1,2}", r"\1", text)  # **bold** / *italic*
        text = re.sub(r"#{1,6}\s*", "", text)                   # ## heading markers
        text = re.sub(r"[ \t]+", " ", text)                     # collapse spaces/tabs
        text = re.sub(r"\n+", " ", text)                        # newlines → space
        r[field] = text.strip()
    return records


def step_add_esg_prefix(records: list[dict], template: str = "[{esg_type}]") -> list[dict]:
    for r in records:
        esg = r.get("esg_type") or "Unknown"
        prefix = template.format(esg_type=esg)
        r["data"] = f"{prefix} {r['data']}"
    return records


def step_inject_cues(records: list[dict], field: str = "data") -> list[dict]:
    """Feature injection: insert salient marker tokens before discriminative cues.

    [NUM]   before concrete numbers (decimals / 3+ digits / % / 萬億噸) -> Clear / evidence=Yes
    [DOC]   before named standards/documents (ISO/SBTi/GRI/政策/守則...)  -> Clear / evidence=Yes
    [VAGUE] before vague/aspirational verbs (致力/持續/積極...)           -> Not Clear

    Does NOT delete any text — only makes the cues explicit so the model (esp. in
    the low-data regime) can attend to them. Markers should be added to the
    tokenizer as atomic tokens (see training.extra_tokens) so they don't fragment.
    """
    import re
    DOC = ["ISO", "SBTi", "GRI", "TCFD", "TNFD", "CDP", "政策", "守則", "辦法",
           "準則", "委員會", "規範", "框架", "指引", "認證", "管理系統", "標準"]
    VAGUE = ["致力", "持續", "積極", "努力", "規劃", "期望", "逐步", "強化",
             "優化", "推動", "期許", "邁向", "朝向", "力求"]
    num_re = re.compile(r"\d+\.\d+|\d{3,}|\d+\s*[%％]|\d+\s*(?:萬|億|公噸|噸)")
    doc_re = re.compile("(" + "|".join(map(re.escape, DOC)) + ")")
    vague_re = re.compile("(" + "|".join(map(re.escape, VAGUE)) + ")")
    for r in records:
        t = r.get(field, "")
        t = num_re.sub(lambda m: " [NUM] " + m.group(0), t)
        t = doc_re.sub(r" [DOC] \1", t)
        t = vague_re.sub(r" [VAGUE] \1", t)
        r[field] = re.sub(r"\s+", " ", t).strip()
    return records


def step_normalize_fullwidth(records: list[dict], field: str = "data") -> list[dict]:
    # 只正規化全形數字與拉丁字母，保留中文標點不動
    # （NFKC 會把 ，→, 等中文標點也轉掉，對繁中 BERT 反而有害）
    _FW_DIGIT = str.maketrans(
        "０１２３４５６７８９",
        "0123456789",
    )
    _FW_UPPER = str.maketrans(
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    )
    _FW_LOWER = str.maketrans(
        "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
        "abcdefghijklmnopqrstuvwxyz",
    )
    for r in records:
        text = r.get(field, "")
        text = text.translate(_FW_DIGIT).translate(_FW_UPPER).translate(_FW_LOWER)
        r[field] = text
    return records
