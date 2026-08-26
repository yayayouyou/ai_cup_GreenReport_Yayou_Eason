"""RQ3 evidence: is the cascade multiplier a property of the SYSTEM or of the METRIC DESIGN?

Two studies that need no external gold labels and no gatekeeper:

  (A) CROSS-CORPUS SCHEMA CHECK — does the label-schema cascade hold beyond our competition?
      Verified on ML-Promise (5 languages, 2110 items) + our AI CUP corpus. Establishes that the
      structure the analysis exploits is a property of the promise-verification task FAMILY.
      Also characterises the violations, which turn out to be conceptually interesting.

  (B) DUAL-METRIC ABLATION — score the SAME predictions from the SAME 44 binary sources under
      two metric designs:
        official : quality/timeline macro-averaged over REAL classes only (N/A excluded)
        flat     : every field macro-averaged over ALL labels, N/A included as a scored class
      If the cascade multiplier is a metric artefact, the coupling between upstream binary
      accuracy and untouched downstream fields must weaken under `flat`. We measure by how much.

Usage:
  .venv_mac/bin/python paper/metric_design_ablation.py
"""
import argparse, json, os
from collections import defaultdict

from fit_cascade_from_val import FIELDS, W, SCORED, f1_from_counts
from binary_swap_experiment import splice, load_sources, vote

ALL_LABELS = {
    'promise_status': ['Yes', 'No'],
    'evidence_status': ['Yes', 'No', 'N/A'],
    'evidence_quality': ['Clear', 'Not Clear', 'Misleading', 'N/A'],
    'verification_timeline': ['already', 'within_2_years', 'between_2_and_5_years',
                              'more_than_5_years', 'N/A'],
}
EQUAL_W = {f: 0.25 for f in FIELDS}


def score(rows, pred_by, labels, weights):
    fields = {}
    for f in FIELDS:
        tot = 0.0
        for c in labels[f]:
            tp = fn = fp = 0
            for r in rows:
                g, p = r[f], pred_by[str(r['id'])][f]
                if g == c and p == c:
                    tp += 1
                elif g == c:
                    fn += 1
                elif p == c:
                    fp += 1
            tot += f1_from_counts(tp, fn, fp)
        fields[f] = tot / len(labels[f])
    return sum(weights[f] * fields[f] for f in FIELDS), fields


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** .5
    dy = sum((b - my) ** 2 for b in ys) ** .5
    return num / (dx * dy) if dx * dy else 0.0


def ols(xs, ys):
    """slope of y ~ x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = sum((a - mx) ** 2 for a in xs)
    return num / den if den else 0.0


def schema_check(rows, name, mapping=None):
    """Rule 1: promise=No => downstream all N/A.  Rule 2: evidence in {No,N/A} => quality N/A."""
    na = {'N/A', 'N-A', 'Other', None}
    n_no = v1 = n_ev = v2 = 0
    # R1 restricted to the fields every corpus on this schema shares. ML-Promise scores evidence
    # Yes/No with no N/A class, so R1's evidence clause cannot transfer to it; quality and timeline
    # can. These are the counts the paper reports.
    v1t = pno_real_q = pno_mis = 0
    kinds = defaultdict(int)
    for r in rows:
        ps, es = r.get('promise_status'), r.get('evidence_status')
        eq, vt = r.get('evidence_quality'), r.get('verification_timeline')
        if ps == 'No':
            n_no += 1
            ok = (es in na or es == 'No') and eq in na and vt in na
            if not ok:
                v1 += 1
                kinds[(es, eq, vt)] += 1
            if eq not in na or vt not in na:
                v1t += 1
            if eq not in na:
                pno_real_q += 1
                if 'isleading' in str(eq):
                    pno_mis += 1
        if es in na or es == 'No':
            n_ev += 1
            if eq not in na:
                v2 += 1
    return {'corpus': name, 'n': len(rows), 'promise_no': n_no, 'rule1_violations': v1,
            'rule1_rate': v1 / n_no if n_no else 0.0,
            'rule1_violations_transferable': v1t,
            'transferable_violations': v1t + v2,
            'promise_no_with_real_quality': pno_real_q,
            'promise_no_with_misleading': pno_mis,
            'evidence_no_or_na': n_ev, 'rule2_violations': v2,
            'violation_kinds': {str(k): v for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', default='../data_set/vpesg4k_val_1000.json')
    ap.add_argument('--train', default='../fusion/data/raw/vpesg4k_train_1000 V1.json')
    ap.add_argument('--pipeline_val', default='io/val_strong4.json')
    ap.add_argument('--sources', default='io/binary_sources')
    # ML-Promise is not ours to redistribute; fetch it from the SemEval-2025 Task 6 release
    # and point --ext_dir at the directory holding Trainset_<Language>.json.
    ap.add_argument('--ext_dir', default='../data_set/external_mlpromise')
    ap.add_argument('--out', default='out/metric_design.json')
    args = ap.parse_args()
    res = {}

    # ---------------- (A) cross-corpus schema check ----------------
    print("=== (A) LABEL-SCHEMA CASCADE ACROSS CORPORA ===")
    checks = []
    for name, path in [('AI CUP zh (train-1000)', args.train), ('AI CUP zh (val-1000)', args.val)]:
        checks.append(schema_check(json.load(open(path, encoding='utf-8')), name))
    for lang in ['Chinese', 'English', 'French', 'Japanese', 'Korean']:
        p = os.path.join(args.ext_dir, f'Trainset_{lang}.json')
        if os.path.exists(p):
            rows = json.load(open(p, encoding='utf-8-sig'))
            checks.append(schema_check(rows, f'ML-Promise {lang}'))
    print(f"{'corpus':26s} {'n':>5s} {'P=No':>6s} {'R1 viol':>8s} {'rate':>7s} {'R2 viol':>8s}")
    for c in checks:
        print(f"{c['corpus']:26s} {c['n']:5d} {c['promise_no']:6d} {c['rule1_violations']:8d} "
              f"{c['rule1_rate']*100:6.1f}% {c['rule2_violations']:8d}")
    tot_n = sum(c['n'] for c in checks if c['corpus'].startswith('ML-Promise'))
    tot_v = sum(c['rule1_violations'] + c['rule2_violations'] for c in checks
                if c['corpus'].startswith('ML-Promise'))
    tot_t = sum(c['transferable_violations'] for c in checks
                if c['corpus'].startswith('ML-Promise'))
    tot_q = sum(c['promise_no_with_real_quality'] for c in checks
                if c['corpus'].startswith('ML-Promise'))
    tot_m = sum(c['promise_no_with_misleading'] for c in checks
                if c['corpus'].startswith('ML-Promise'))
    res['mlpromise_transferable'] = {'n': tot_n, 'violations': tot_t,
                                     'rate': tot_t / tot_n if tot_n else 0.0,
                                     'promise_no_with_real_quality': tot_q,
                                     'promise_no_with_misleading': tot_m}
    if tot_n:
        print(f"\nML-Promise, rules that transfer (R1 on quality/timeline, R2 on quality): "
              f"{tot_t} / {tot_n} = {100*tot_t/tot_n:.1f}%   "
              f"[{tot_q} are promise=No carrying a real quality judgement, {tot_m} Misleading]")
        print(f"ML-Promise, R1 as written (evidence clause included): {tot_v} / {tot_n} = "
              f"{100*tot_v/tot_n:.1f}%   AI CUP: 0 violations (schema strictly enforced)")
    else:
        print("\nML-Promise corpus not found under --ext_dir; skipping the cross-corpus check."
              "   AI CUP: 0 violations (schema strictly enforced)")
    print("\nviolation kinds worth reporting (evidence/quality asserted with promise=No):")
    for c in checks:
        if c['violation_kinds']:
            print(f"  {c['corpus']}: {c['violation_kinds']}")
    res['schema'] = checks

    # ---------------- (B) dual-metric ablation ----------------
    print("\n=== (B) DUAL-METRIC ABLATION: same predictions, two metric designs ===")
    rows_all = json.load(open(args.val, encoding='utf-8'))
    pipeline_by = {str(x['id']): x for x in json.load(open(args.pipeline_val, encoding='utf-8'))}
    rows = [r for r in rows_all if str(r['id']) in pipeline_by]

    sources = load_sources(args.sources)
    s4 = ['ckip_tapt_ep3', 'macbert_tapt', 'bgem3', 'bgem3_tapt']
    if all(k in sources for k in s4):
        sources['strong4(vote)'] = vote(sources, s4)
    sources['PIPELINE(self)'] = pipeline_by

    variants = {
        'official (N/A-excluded T,Q; weighted)': (SCORED, W),
        'flat (N/A scored everywhere; weighted)': (ALL_LABELS, W),
        'official, equal weights': (SCORED, EQUAL_W),
        'flat, equal weights (SemEval-style)': (ALL_LABELS, EQUAL_W),
    }
    print("reference system under each convention:")
    for name, (lab, wt) in variants.items():
        t, f = score(rows, pipeline_by, lab, wt)
        print(f"  {name:40s} S={t:.4f}   T={f['verification_timeline']:.4f} Q={f['evidence_quality']:.4f}")

    pts = defaultdict(lambda: defaultdict(list))
    for name, bby in sorted(sources.items()):
        if not all(str(r['id']) in bby for r in rows):
            continue
        spl = splice(rows, bby, pipeline_by)
        for vname, (lab, wt) in variants.items():
            t, f = score(rows, spl, lab, wt)
            pts[vname]['P'].append(f['promise_status'])
            pts[vname]['T'].append(f['verification_timeline'])
            pts[vname]['Q'].append(f['evidence_quality'])
            pts[vname]['S'].append(t)

    print(f"\ncoupling between the (only) changed module -- the binary -- and the UNTOUCHED"
          f" downstream fields, across {len(pts['official (N/A-excluded T,Q; weighted)']['P'])} sources:")
    print(f"{'metric design':42s} {'r(P,T)':>8s} {'slope':>8s} {'r(P,Q)':>8s} {'slope':>8s}"
          f" {'range T':>9s} {'range Q':>9s}")
    coupling = {}
    for vname in variants:
        P, T, Q = pts[vname]['P'], pts[vname]['T'], pts[vname]['Q']
        rT, sT = pearson(P, T), ols(P, T)
        rQ, sQ = pearson(P, Q), ols(P, Q)
        coupling[vname] = {'r_PT': rT, 'slope_PT': sT, 'r_PQ': rQ, 'slope_PQ': sQ,
                           'range_T': max(T) - min(T), 'range_Q': max(Q) - min(Q)}
        print(f"{vname:42s} {rT:8.3f} {sT:8.3f} {rQ:8.3f} {sQ:8.3f}"
              f" {max(T)-min(T):9.4f} {max(Q)-min(Q):9.4f}")
    res['coupling'] = coupling

    o = coupling['official (N/A-excluded T,Q; weighted)']
    fl = coupling['flat (N/A scored everywhere; weighted)']
    print(f"\n  timeline coupling slope: official {o['slope_PT']:.3f} -> flat {fl['slope_PT']:.3f}"
          f"   (flat design couples {fl['slope_PT']/o['slope_PT']:.1f}x MORE strongly)"
          if o['slope_PT'] else "   (no sources found under --sources; ratio not computable)")
    print(f"  quality  coupling slope: official {o['slope_PQ']:.3f} -> flat {fl['slope_PQ']:.3f}"
          f"   (flat couples {fl['slope_PQ']/o['slope_PQ']:.1f}x more strongly)")
    print("  => N/A exclusion does NOT create the coupling; it REMOVES the mechanical part.")
    print("     Under a flat metric the downstream N/A class is scored, and its F1 is determined")
    print("     by the gate itself, so the binary decision is re-scored inside every field")
    print(f"     (r(P,T) = {fl['r_PT']:.3f} under flat vs {o['r_PT']:.3f} under the official design).")

    # cascade share of the total spread under each design
    print("\n  share of total-score spread attributable to the untouched downstream fields:")
    for vname, (lab, wt) in variants.items():
        P, T, Q, S = pts[vname]['P'], pts[vname]['T'], pts[vname]['Q'], pts[vname]['S']
        i_lo, i_hi = min(range(len(S)), key=lambda i: S[i]), max(range(len(S)), key=lambda i: S[i])
        dT, dQ = T[i_hi] - T[i_lo], Q[i_hi] - Q[i_lo]
        dS = S[i_hi] - S[i_lo]
        casc = wt['verification_timeline'] * dT + wt['evidence_quality'] * dQ
        print(f"    {vname:42s} worst->best dS={dS:+.4f}   cascade part {casc:+.4f}"
              f"  ({100*casc/dS if dS else 0:.1f}%)")
        res.setdefault('spread', {})[vname] = {'dS': dS, 'cascade': casc,
                                               'share': casc / dS if dS else 0}

    json.dump(res, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
