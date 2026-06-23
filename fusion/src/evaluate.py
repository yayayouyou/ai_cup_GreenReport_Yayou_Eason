"""
Competition scoring: weighted Macro-F1 across 4 subtasks.

Total = promise_status  * 0.20
      + evidence_status * 0.30
      + evidence_quality * 0.35
      + verification_timeline * 0.15
"""

import json
import argparse
from pathlib import Path
from sklearn.metrics import f1_score

TASKS = {
    "promise_status":       0.20,
    "evidence_status":      0.30,
    "evidence_quality":     0.35,
    "verification_timeline": 0.15,
}

LABELS = {
    "promise_status":        ["Yes", "No"],
    "evidence_status":       ["Yes", "No", "N/A"],
    "evidence_quality":      ["Clear", "Not Clear", "Misleading", "N/A"],
    "verification_timeline": ["already", "within_2_years", "between_2_and_5_years",
                              "more_than_5_years", "N/A"],
}


def macro_f1(y_true: list, y_pred: list, labels: list) -> float:
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def score(ground_truth: list[dict], predictions: list[dict]) -> dict:
    """
    Args:
        ground_truth: list of dicts with 'id' and task label fields
        predictions:  list of dicts with 'id' and task label fields (same order not required)

    Returns:
        dict with per-task F1, weighted total, and per-task details
    """
    pred_map = {d["id"]: d for d in predictions}

    results = {}
    total = 0.0

    for task, weight in TASKS.items():
        y_true, y_pred = [], []
        for gt in ground_truth:
            pred = pred_map.get(gt["id"])
            if pred is None:
                raise ValueError(f"Missing prediction for id={gt['id']}")
            y_true.append(gt[task])
            y_pred.append(pred[task])

        f1 = macro_f1(y_true, y_pred, LABELS[task])
        weighted = f1 * weight
        total += weighted
        results[task] = {"macro_f1": round(f1, 4), "weight": weight, "weighted": round(weighted, 4)}

    results["total"] = round(total, 4)
    return results


def print_report(results: dict) -> None:
    print(f"\n{'Task':<25} {'Macro-F1':>10} {'Weight':>8} {'Weighted':>10}")
    print("-" * 57)
    for task, weight in TASKS.items():
        r = results[task]
        print(f"{task:<25} {r['macro_f1']:>10.4f} {r['weight']:>8.2f} {r['weighted']:>10.4f}")
    print("-" * 57)
    print(f"{'TOTAL':<25} {'':>10} {'':>8} {results['total']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description="ESG competition scorer")
    parser.add_argument("ground_truth", help="Ground truth JSON file path")
    parser.add_argument("predictions", help="Predictions JSON file path")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    gt = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    pred = json.loads(Path(args.predictions).read_text(encoding="utf-8"))

    results = score(gt, pred)
    print_report(results)

    if args.output:
        Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
