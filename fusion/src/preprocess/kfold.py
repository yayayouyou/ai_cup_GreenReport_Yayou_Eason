"""Fixed-seed stratified k-fold splits for ESG experiments."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from src.preprocess.split import TASKS, _strat_key


def _record_sort_key(record: dict):
    rid = record.get("id", "")
    try:
        return (0, int(rid))
    except (TypeError, ValueError):
        return (1, str(rid))


def _write_json(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_distribution(records: list[dict], prefix: str) -> None:
    print(f"{prefix}: n={len(records)}")
    for task in TASKS:
        counts = Counter(r[task] for r in records)
        values = ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
        print(f"  {task}: {values}")


def make_kfold_splits(
    data_path,
    out_dir,
    n_splits: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Create deterministic stratified k-fold train/val JSON files.

    Returns a list of fold metadata dictionaries with train/val file paths.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    records = json.loads(Path(data_path).read_text(encoding="utf-8"))
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(_strat_key(record), []).append(record)

    rng = random.Random(seed)
    folds: list[list[dict]] = [[] for _ in range(n_splits)]
    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        for idx, record in enumerate(group):
            folds[idx % n_splits].append(record)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_infos: list[dict] = []
    for fold_idx in range(n_splits):
        fold_dir = out_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_records = sorted(folds[fold_idx], key=_record_sort_key)
        train_records = sorted(
            [record for idx, fold in enumerate(folds) if idx != fold_idx for record in fold],
            key=_record_sort_key,
        )

        train_file = fold_dir / "train.json"
        val_file = fold_dir / "val.json"
        _write_json(train_file, train_records)
        _write_json(val_file, val_records)

        print(f"\nFold {fold_idx}")
        _print_distribution(train_records, "  train")
        _print_distribution(val_records, "  val")

        fold_infos.append({
            "fold": fold_idx,
            "train_file": str(train_file),
            "val_file": str(val_file),
            "train_size": len(train_records),
            "val_size": len(val_records),
        })

    manifest = {
        "data_path": str(data_path),
        "n_splits": n_splits,
        "seed": seed,
        "folds": fold_infos,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return fold_infos


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic ESG k-fold splits.")
    parser.add_argument("data_path", nargs="?", default="data/raw/vpesg_4k_train_1000.json")
    parser.add_argument("out_dir", nargs="?", default="data/splits/kfold_seed42")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    make_kfold_splits(args.data_path, args.out_dir, n_splits=args.n_splits, seed=args.seed)


if __name__ == "__main__":
    main()
