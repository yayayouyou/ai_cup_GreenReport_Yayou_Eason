# -*- coding: utf-8 -*-
"""What can actually be claimed about the leaderboard readings (0.6760 public / 0.6405
private) for the FINAL submission?

Search random public/private splits of the 2,000 released test items at several size ratios,
scoring each side under (a) the schema convention (Eq. 1) and (b) the lenient convention
(labels present in the slice, sklearn default). Report whether any split reproduces the pair.
"""
import csv
import io
import random

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--test_gold", required=True,
                 help="the organizers' released test gold (not redistributed here)")
GOLD = _ap.parse_args().test_gold
PRED = "../submissions/FINAL_SUBMISSION_yayou_0.6760.csv"

W = {"promise_status": .20, "evidence_status": .30, "evidence_quality": .35,
     "verification_timeline": .15}
SCHEMA = {
    "promise_status": ["Yes", "No"],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading"],
    "verification_timeline": ["already", "within_2_years", "between_2_and_5_years",
                              "more_than_5_years"],
}
ALL = {
    "promise_status": ["Yes", "No"],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
    "verification_timeline": ["already", "within_2_years", "between_2_and_5_years",
                              "more_than_5_years", "N/A"],
}

rows = list(csv.DictReader(io.open(GOLD, encoding="utf-8-sig")))
pred = {r["id"]: r for r in csv.DictReader(io.open(PRED, encoding="utf-8-sig"))}
pairs = {f: [(r[f], pred[r["id"]][f]) for r in rows] for f in W}
N = len(rows)


def f1(tp, fn, fp):
    return 2 * tp / (2 * tp + fn + fp) if (2 * tp + fn + fp) else 0.0


def composite(idx, mode):
    tot = 0.0
    for f, w in W.items():
        gp = pairs[f]
        if mode == "schema":
            labels = SCHEMA[f]
        else:
            present = {gp[i][0] for i in idx} | {gp[i][1] for i in idx}
            labels = [c for c in ALL[f] if c in present]
        s = 0.0
        for c in labels:
            tp = fn = fp = 0
            for i in idx:
                g, p = gp[i]
                if g == c and p == c:
                    tp += 1
                elif g == c:
                    fn += 1
                elif p == c:
                    fp += 1
            s += f1(tp, fn, fp)
        tot += w * (s / len(labels) if labels else 0.0)
    return tot


TARGET = (0.6760, 0.6405)
rng = random.Random(20260829)
for mode in ("schema", "lenient"):
    hits = 0
    best_pub = -1.0
    closest = None
    n_splits = 0
    for frac in (0.35, 0.50, 0.65):
        k = int(N * frac)
        for _ in range(1000):
            n_splits += 1
            idx = list(range(N))
            rng.shuffle(idx)
            pub, priv = idx[:k], idx[k:]
            sp = composite(pub, mode)
            sv = composite(priv, mode)
            # leaderboard sides are unlabeled: accept either assignment
            for a, b in ((sp, sv), (sv, sp)):
                d = abs(a - TARGET[0]) + abs(b - TARGET[1])
                if closest is None or d < closest[0]:
                    closest = (d, a, b)
                if a >= TARGET[0] - 0.0005 and abs(b - TARGET[1]) <= 0.0055:
                    hits += 1
            best_pub = max(best_pub, sp, sv)
    print(f"{mode:8s} over {n_splits} random splits (ratios .35/.50/.65):")
    print(f"   splits matching (~0.6760, ~0.6405): {hits}")
    print(f"   highest single-side composite seen: {best_pub:.4f}")
    print(f"   closest pair to the target        : ({closest[1]:.4f}, {closest[2]:.4f})"
          f"  L1-dist {closest[0]:.4f}")
