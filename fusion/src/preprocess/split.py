"""
Stratified train/val split for the ESG dataset.

Special handling:
- evidence_quality=Misleading (1 sample) → forced into train
- verification_timeline=within_2_years (13 samples) → stratified as separate key
- Composite stratification key: promise_status | evidence_quality | within_2_years_flag
"""

import json
import random
from collections import Counter
from pathlib import Path


TASKS = ["promise_status", "evidence_status", "evidence_quality", "verification_timeline"]


def _strat_key(record: dict) -> str:
    timeline_flag = "w2y" if record["verification_timeline"] == "within_2_years" else "other"
    return f"{record['promise_status']}|{record['evidence_quality']}|{timeline_flag}"


def make_split(
    data_path,
    out_dir,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    Split records into train/val sets with stratification.

    Returns:
        (train_records, val_records)
    """
    records = json.loads(Path(data_path).read_text(encoding="utf-8"))

    # Separate Misleading — too rare to split, goes entirely to train
    misleading = [r for r in records if r["evidence_quality"] == "Misleading"]
    rest = [r for r in records if r["evidence_quality"] != "Misleading"]

    # Group remaining records by composite stratification key
    groups: dict[str, list[dict]] = {}
    for r in rest:
        groups.setdefault(_strat_key(r), []).append(r)

    rng = random.Random(seed)
    train: list[dict] = list(misleading)
    val: list[dict] = []

    for group in groups.values():
        rng.shuffle(group)
        # At least 1 val sample per group when group has >1 record
        n_val = max(1, round(len(group) * val_ratio)) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "val.json").write_text(
        json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _print_summary(train, val)
    return train, val


def _print_summary(train: list[dict], val: list[dict]) -> None:
    total = len(train) + len(val)
    print(f"Split: {len(train)} train / {len(val)} val  (total {total})")
    print()

    for task in TASKS:
        train_counts = Counter(r[task] for r in train)
        val_counts = Counter(r[task] for r in val)
        all_labels = sorted(set(train_counts) | set(val_counts))
        print(f"  {task}:")
        for label in all_labels:
            t, v = train_counts.get(label, 0), val_counts.get(label, 0)
            print(f"    {label:<25} train={t:>4}  val={v:>3}")
        print()


if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/vpesg_4k_train_1000.json"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/splits"
    make_split(data_path, out_dir)
