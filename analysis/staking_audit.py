"""Reconstruct the rare-class staking history from the submission archive. Backs Section 6.1's
"the survivors of 38 candidates".

During the evaluation the platform capped uploads at three per day. Between 2026-06-11 and
2026-06-18 the team prepared 79 submission variants locally; across them, 38 distinct test items
were at some point forced to Misleading (the supervised head predicts essentially none, so every
Misleading row in a submission file is a staked override). The shipped submission keeps five of
the 38: 12306, 12599, 12606, 12743, 12772. The released labels confirm 12599.

Two of the three gold Misleading items were staked at some point and DROPPED before the final
pick -- 12526 (the LLM detector's rank-3 candidate, staked in variants from 06-11 through 06-17)
and 13851 (probed once on 06-17). Only 12599 survived. That is the paper's "a search, not a
detector" in one line: leaderboard feedback on a single staked item is smaller than the
partition noise the paper measures, so the search could not recognise gold it was already
holding.

The archive of prepared submissions is not redistributed here (it is working history, not a
release artifact); point --archive at it. The shipped FINAL submission is in ../submissions/.

Usage:
    python staking_audit.py --archive <path-to-submissions-archive>
"""
import argparse
import csv
import glob
import io
import os


def misleading_ids(path):
    try:
        rows = list(csv.DictReader(io.open(path, encoding="utf-8-sig")))
    except Exception:
        return None
    if not rows or "evidence_quality" not in rows[0]:
        return None
    return sorted(r["id"] for r in rows if r["evidence_quality"].strip() == "Misleading")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True,
                    help="directory holding the dated submission variants (not redistributed)")
    ap.add_argument("--gold_misleading", default="12526,12599,13851",
                    help="gold Misleading test ids, comma-separated (from the released labels)")
    args = ap.parse_args()
    gold = set(args.gold_misleading.split(","))

    files = sorted(glob.glob(os.path.join(args.archive, "**", "submission*.csv"), recursive=True))
    files += sorted(glob.glob(os.path.join(args.archive, "FINAL_SUBMISSION*.csv")))

    n_files = 0
    ever = set()
    for f in files:
        mis = misleading_ids(f)
        if mis is None:
            continue
        n_files += 1
        tag = os.path.relpath(f, args.archive).replace(os.sep, "/")
        marks = " ".join(("*" + i if i in gold else i) for i in mis)
        print(f"{tag:52s} {len(mis):3d}  {marks}")
        ever.update(mis)

    print(f"\nsubmission variants scanned : {n_files}")
    print(f"distinct items ever staked  : {len(ever)}")
    print("gold items touched by the search: "
          + ", ".join(sorted(ever & gold)) + f"  ({len(ever & gold)} of {len(gold)})")


if __name__ == "__main__":
    main()
