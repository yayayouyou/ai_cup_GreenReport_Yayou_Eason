"""Score one prediction file under the paper's convention AND the organizers' baseline-script
convention, and print the gap. Backs the paper's Section 4.4 claim that identical predictions
move by +0.0059 between the two.

The organizers' released tutorial/baseline code (distributed with the competition materials; not
redistributed here) evaluates with these fixed label sets (its EVAL_FIELDS):

    promise_status         Yes / No
    verification_timeline  already / within_2_years / between_2_and_5_years /
                           longer_than_5_years / N/A
    evidence_status        Yes / No / N/A
    evidence_quality       Clear / Not Clear / Misleading / N/A

Two things differ from the schema convention of the paper's Eq. (1):
  * N/A is averaged into QUALITY and TIMELINE (the official docs exclude it there);
  * the timeline list says 'longer_than_5_years', while every released data file labels that
    class 'more_than_5_years' -- so under the script's set, that class can never have a true
    positive and contributes a structural zero to the timeline macro.

On the deployed validation predictions (io/val_strong4.json vs the released val gold):

    paper convention     S = 0.6095   (timeline macro 0.6607, quality 0.3855)
    baseline convention  S = 0.6153   (timeline macro 0.5318, quality 0.4575)
    gap on identical predictions: +0.0059
      quality  +0.0252 weighted  (N/A averaged in)
      timeline -0.0193 weighted  (N/A averaged in, one class structurally zeroed)
      promise / evidence: exactly 0

Run bare from inside analysis/:  python scorer_conventions.py
"""
import argparse
import io
import json

from sklearn.metrics import f1_score

W = {"promise_status": 0.20, "evidence_status": 0.30,
     "evidence_quality": 0.35, "verification_timeline": 0.15}

PAPER = {
    "promise_status": ["Yes", "No"],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading"],
    "verification_timeline": ["already", "within_2_years",
                              "between_2_and_5_years", "more_than_5_years"],
}

# Transcribed from the organizers' baseline code (EVAL_FIELDS), typo included.
BASELINE = {
    "promise_status": ["Yes", "No"],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
    "verification_timeline": ["already", "within_2_years",
                              "between_2_and_5_years", "longer_than_5_years", "N/A"],
}


def composite(rows, pred_by, sets):
    total = 0.0
    fields = {}
    for f, labels in sets.items():
        g = [r[f] for r in rows]
        p = [pred_by[str(r["id"])][f] for r in rows]
        m = f1_score(g, p, labels=labels, average="macro", zero_division=0)
        fields[f] = m
        total += W[f] * m
    return total, fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="../data_set/vpesg4k_val_1000.json")
    ap.add_argument("--pred", default="io/val_strong4.json")
    args = ap.parse_args()

    rows = json.load(io.open(args.val, encoding="utf-8"))
    pred = json.load(io.open(args.pred, encoding="utf-8"))
    by = {str(p["id"]): p for p in (pred if isinstance(pred, list) else pred["predictions"])}

    sp, fp = composite(rows, by, PAPER)
    sb, fb = composite(rows, by, BASELINE)

    print(f"{'field':24s} {'paper':>8s} {'baseline':>9s} {'weighted diff':>14s}")
    for f in W:
        print(f"{f:24s} {fp[f]:8.4f} {fb[f]:9.4f} {W[f] * (fb[f] - fp[f]):+14.4f}")
    print(f"{'composite S':24s} {sp:8.4f} {sb:9.4f} {sb - sp:+14.4f}")
    print("\nSame predictions, same gold; the whole gap is the scoring convention --")
    print("N/A averaged into quality and timeline, plus one timeline label the released")
    print("files never use ('longer_than_5_years' vs the data's 'more_than_5_years').")


if __name__ == "__main__":
    main()
