"""Merge train(1000) + val(1000) -> train_plus_val_2000.json for the full-data
(submission) retrain. Casts all ids to int (train is int, val is str) so the
DataLoader collate doesn't choke on mixed id types.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TR = ROOT / "data/raw/vpesg_4k_train_1000.json"
VA = ROOT / "data/raw/vpesg4k_val_1000.json"
OUT = ROOT / "data/raw/train_plus_val_2000.json"


def main():
    tr = json.loads(TR.read_text())
    va = json.loads(VA.read_text())
    assert all("promise_status" in r for r in va), "val 缺標籤"
    both = tr + va
    for r in both:
        r["id"] = int(r["id"])          # unify id type (train int + val str -> int)
    assert len({r["id"] for r in both}) == len(both), "id 重複"
    OUT.write_text(json.dumps(both, ensure_ascii=False))
    print(f"wrote {OUT}  ({len(tr)}+{len(va)}={len(both)} rows, ids unified to int)")


if __name__ == "__main__":
    main()
