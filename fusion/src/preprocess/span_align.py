"""
Span alignment: locate promise_string / evidence_string inside data.

Both `data` and the target string are fullwidth-normalized first (same rule as
preprocess.pipeline.step_normalize_fullwidth) so the model never sees raw
全/半形差異 — span positions refer to the normalized text.

Run as a script for diagnostics:
    PYTHONPATH=. .venv/bin/python src/preprocess/span_align.py
"""

import json
import sys
from pathlib import Path


_FW_DIGIT = str.maketrans("０１２３４５６７８９", "0123456789")
_FW_UPPER = str.maketrans(
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
)
_FW_LOWER = str.maketrans(
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
    "abcdefghijklmnopqrstuvwxyz",
)


def normalize_fullwidth(text: str) -> str:
    return text.translate(_FW_DIGIT).translate(_FW_UPPER).translate(_FW_LOWER)


_MULTI_SPAN_DELIM = "｜"   # full-width vertical bar — used by annotators to join multiple spans


def find_spans(data: str, target: str) -> list[tuple[int, int]]:
    """Return list of (start, end) for each sub-span of `target` found in `data`.
    Both inputs are fullwidth-normalized first. Multi-span targets are split by ｜.
    Returns [] if target is empty or no sub-span matches."""
    if not target:
        return []
    data_n = normalize_fullwidth(data)
    parts  = [p.strip() for p in normalize_fullwidth(target).split(_MULTI_SPAN_DELIM) if p.strip()]
    spans = []
    for p in parts:
        idx = data_n.find(p)
        if idx >= 0:
            spans.append((idx, idx + len(p)))
    return spans


# ── Diagnostic ────────────────────────────────────────────────────────────────

def diagnose(records: list[dict]) -> None:
    fields = {
        "promise":  ("promise_status",  "promise_string"),
        "evidence": ("evidence_status", "evidence_string"),
    }
    stats = {k: {"yes": 0, "full": 0, "partial": 0, "none": 0, "n_subspans": 0, "miss": []}
             for k in fields}

    for r in records:
        for key, (status_field, string_field) in fields.items():
            if r.get(status_field) != "Yes":
                continue
            stats[key]["yes"] += 1
            target = r.get(string_field) or ""
            n_parts = len([p for p in normalize_fullwidth(target).split(_MULTI_SPAN_DELIM) if p.strip()])
            spans   = find_spans(r["data"], target)
            stats[key]["n_subspans"] += len(spans)
            if not spans:
                stats[key]["none"] += 1
                if len(stats[key]["miss"]) < 5:
                    stats[key]["miss"].append({
                        "id": r["id"],
                        "data":   normalize_fullwidth(r["data"])[:250],
                        "string": normalize_fullwidth(target)[:250],
                    })
            elif len(spans) == n_parts:
                stats[key]["full"] += 1
            else:
                stats[key]["partial"] += 1

    def pct(n, d):
        return f"{n}/{d} ({n/d*100:.1f}%)" if d else "n/a"

    print(f"=== Span alignment diagnostic ===")
    print(f"Records: {len(records)}\n")
    for key in ("promise", "evidence"):
        s = stats[key]
        recoverable = s["full"] + s["partial"]
        print(f"{key}=Yes : {s['yes']}")
        print(f"  fully aligned       : {pct(s['full'],    s['yes'])}")
        print(f"  partially aligned   : {pct(s['partial'], s['yes'])}")
        print(f"  completely missed   : {pct(s['none'],    s['yes'])}")
        print(f"  recoverable (any)   : {pct(recoverable,  s['yes'])}")
        print(f"  total sub-spans     : {s['n_subspans']}  (avg {s['n_subspans']/max(s['yes'],1):.2f} per Yes sample)")
        print()

    for key in ("promise", "evidence"):
        misses = stats[key]["miss"]
        if not misses:
            continue
        print(f"--- Sample {key} misses (truly unmatched) ---")
        for m in misses[:3]:
            print(f"id={m['id']}")
            print(f"  data   : {m['data']}")
            print(f"  string : {m['string']}")
            print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/vpesg_4k_train_1000.json"
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    diagnose(records)
