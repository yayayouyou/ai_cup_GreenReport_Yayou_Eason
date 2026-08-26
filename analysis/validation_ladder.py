"""RQ2 centrepiece, self-contained: the validation debugging ladder on ONE fixed configuration.

The point of the ladder is that a system's self-estimate is not one number — it is a number per
scoring convention, and the differences dwarf most modelling gains. We hold the PREDICTIONS fixed
and vary only the scorer, so every difference below is estimation bias, not a system change.

Rungs (worst-practice to faithful):
  R0  5-split held-out, macro over labels PRESENT IN THE EVALUATION SLICE (sklearn's default when
      `labels=` is omitted), N/A included  -> the rare class silently drops out of most splits
  R1  5-split held-out, macro over the FIXED schema label set, N/A included
  R2  full val, fixed label set, N/A included                        (proxy metric)
  R3  full val, official convention: N/A excluded for quality/timeline  (the scored metric)

Usage:
  .venv_mac/bin/python paper/validation_ladder.py
"""
import argparse, json, random
from collections import defaultdict

from fit_cascade_from_val import FIELDS, W, SCORED, f1_from_counts
from metric_design_ablation import ALL_LABELS


def macro(rows, pred_by, field, labels):
    tot = 0.0
    for c in labels:
        tp = fn = fp = 0
        for r in rows:
            g, p = r[field], pred_by[str(r['id'])][field]
            if g == c and p == c:
                tp += 1
            elif g == c:
                fn += 1
            elif p == c:
                fp += 1
        tot += f1_from_counts(tp, fn, fp)
    return tot / len(labels) if labels else 0.0


def score(rows, pred_by, mode):
    """mode: 'official' | 'proxy' (N/A included, fixed set) | 'dynamic' (labels present in slice)."""
    fields = {}
    for f in FIELDS:
        if mode == 'official':
            labels = SCORED[f]
        elif mode == 'proxy':
            labels = ALL_LABELS[f]
        else:  # dynamic: whatever appears in this slice, gold or predicted (sklearn's default)
            present = {r[f] for r in rows} | {pred_by[str(r['id'])][f] for r in rows}
            labels = [c for c in ALL_LABELS[f] if c in present]
        fields[f] = macro(rows, pred_by, f, labels)
    return sum(W[f] * fields[f] for f in FIELDS), fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', default='../data_set/vpesg4k_val_1000.json')
    ap.add_argument('--pipeline_val', default='out/val_preds_Q0like.json')
    ap.add_argument('--splits', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20260805)
    ap.add_argument('--out', default='out/validation_ladder.json')
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows_all = json.load(open(args.val))
    pred_by = {str(x['id']): x for x in json.load(open(args.pipeline_val))}
    rows = [r for r in rows_all if str(r['id']) in pred_by]
    N = len(rows)

    print("=== VALIDATION LADDER — identical predictions, four scoring conventions ===")
    print(f"    one fixed configuration, n={N}; only the scorer changes\n")

    res = {}
    # held-out rungs: repeated 80/20 splits, report mean +- SD like the original protocol did
    for mode, label in [('dynamic', 'R0  5-split held-out, label set inferred from the slice'),
                        ('proxy',   'R1  5-split held-out, fixed schema label set')]:
        vals = []
        absent = 0
        for _ in range(args.repeats):
            idx = list(range(N))
            rng.shuffle(idx)
            hold = [rows[i] for i in idx[:N // args.splits]]
            t, _ = score(hold, pred_by, mode)
            vals.append(t)
            if not any(r['evidence_quality'] == 'Misleading' for r in hold):
                absent += 1
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** .5
        res[mode] = {'mean': mu, 'sd': sd, 'rare_class_absent_rate': absent / args.repeats}
        print(f"{label}\n      {mu:.4f} ± {sd:.4f}   "
              f"(the support-1 class is absent from {100*absent/args.repeats:.0f}% of held-out slices)")

    for mode, label in [('proxy',    'R2  full val, fixed label set, N/A scored (proxy metric)'),
                        ('official', 'R3  full val, OFFICIAL convention (N/A excluded for Q and T)')]:
        t, f = score(rows, pred_by, mode)
        res[mode + '_full'] = {'total': t, 'fields': f}
        print(f"{label}\n      {t:.4f}   " +
              "  ".join(f"{k.split('_')[0][:4]}={v:.4f}" for k, v in f.items()))

    r0 = res['dynamic']['mean']
    r1 = res['proxy']['mean']
    r2 = res['proxy_full']['total']
    r3 = res['official_full']['total']
    print("\n=== BIAS DECOMPOSITION (same system, same predictions) ===")
    print(f"  R0 -> R1  data-dependent label set          {r1-r0:+.4f}")
    print(f"  R1 -> R2  held-out subsampling              {r2-r1:+.4f}")
    print(f"  R2 -> R3  proxy metric vs official metric   {r3-r2:+.4f}")
    print(f"  TOTAL self-deception R0 -> R3               {r3-r0:+.4f}")
    print(f"\n  A team reading R0 would believe it scored {r0:.4f}; the scored metric says {r3:.4f}.")
    res['decomposition'] = {'label_set': r1 - r0, 'subsampling': r2 - r1,
                            'proxy_vs_official': r3 - r2, 'total': r3 - r0}

    json.dump(res, open(args.out, 'w'), ensure_ascii=False, indent=1)
    print(f"\n✅ {args.out}")


if __name__ == '__main__':
    main()
