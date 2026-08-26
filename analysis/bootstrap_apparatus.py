"""RQ2 evidence: uncertainty apparatus for a small-data, rare-class, cascaded shared task.

Four analyses, all on committed artifacts, no GPU:

  (1) CONFIDENCE INTERVALS on the official metric
        - iid bootstrap over rows (the naive default)
        - CLUSTER bootstrap over the 50 companies (respects within-report correlation)
      Reporting both quantifies effective-sample-size shrinkage from document structure.

  (2) THE DEGENERATE-CLASS RESULT: for a support-s class, a fraction of bootstrap replicates
      contain none of its items (-> e^-1 = 36.8% for s=1). Per-class F1 CIs are then a
      non-identifiable mixture. We measure this rate instead of pretending a CI exists.

  (3) PAIRED BOOTSTRAP for the key design decisions (strong4 vs diluted roster, each binary
      source vs the deployed one) — the significance test the venue's accepted papers use.

  (4) PARTITION SIMULATION: split val 500/500 many times and measure how far the two halves'
      scores drift apart for a FIXED system. This is the rank-free, system-agnostic way to
      state what partition-level score differences mean for support<=4 classes.

  (5) SEED VARIANCE: mean+-SD over the per-seed binary sources, spliced through the frozen
      pipeline, plus the minimum detectable effect implied by the CI half-width.

Usage:
  .venv_mac/bin/python paper/bootstrap_apparatus.py --B 10000
"""
import argparse, json, math, random, re
from collections import defaultdict

from fit_cascade_from_val import FIELDS, W, SCORED, f1_from_counts
from binary_swap_experiment import cascade, load_sources, splice, vote


def score_rows(rows, pred_by):
    """Official weighted metric over an arbitrary row multiset (bootstrap-safe)."""
    fields = {}
    for f in FIELDS:
        tot = 0.0
        for c in SCORED[f]:
            tp = fn = fp = 0
            for r in rows:
                g = r[f]
                p = pred_by[str(r['id'])][f]
                if g == c and p == c:
                    tp += 1
                elif g == c:
                    fn += 1
                elif p == c:
                    fp += 1
            tot += f1_from_counts(tp, fn, fp)
        fields[f] = tot / len(SCORED[f])
    return sum(W[f] * fields[f] for f in FIELDS), fields


def percentile(xs, q):
    ys = sorted(xs)
    if not ys:
        return float('nan')
    k = (len(ys) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ys[lo] if lo == hi else ys[lo] * (hi - k) + ys[hi] * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', default='../data_set/vpesg4k_val_1000.json')
    ap.add_argument('--pipeline_val', default='io/val_strong4.json')
    ap.add_argument('--sources', default='io/binary_sources')
    ap.add_argument('--B', type=int, default=10000)
    ap.add_argument('--out', default='out/bootstrap.json')
    ap.add_argument('--seed', type=int, default=20260805)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows_all = json.load(open(args.val, encoding='utf-8'))
    pipeline_by = {str(x['id']): x for x in json.load(open(args.pipeline_val, encoding='utf-8'))}
    rows = [r for r in rows_all if str(r['id']) in pipeline_by]
    N = len(rows)
    res = {}

    base_S, base_f = score_rows(rows, pipeline_by)
    print(f"=== reference system: val WF1 {base_S:.4f}  " +
          "  ".join(f"{f.split('_')[0][:4]}={base_f[f]:.4f}" for f in FIELDS) + f"  (n={N})")

    by_company = defaultdict(list)
    for r in rows:
        by_company[r.get('company') or 'UNK'].append(r)
    comps = sorted(by_company)
    print(f"    {len(comps)} companies, sizes {min(len(v) for v in by_company.values())}"
          f"-{max(len(v) for v in by_company.values())}")

    # ---------- (1) CIs: iid vs cluster ----------
    print(f"\n=== (1) BOOTSTRAP CIs ON THE OFFICIAL METRIC (B={args.B}) ===")
    iid_tot, clus_tot = [], []
    iid_fields = defaultdict(list)
    clus_fields = defaultdict(list)
    for _ in range(args.B):
        samp = [rows[rng.randrange(N)] for _ in range(N)]
        t, fl = score_rows(samp, pipeline_by)
        iid_tot.append(t)
        for f in FIELDS:
            iid_fields[f].append(fl[f])
    for _ in range(args.B):
        samp = []
        for _ in range(len(comps)):
            samp.extend(by_company[comps[rng.randrange(len(comps))]])
        t, fl = score_rows(samp, pipeline_by)
        clus_tot.append(t)
        for f in FIELDS:
            clus_fields[f].append(fl[f])

    def ci(xs):
        return percentile(xs, .025), percentile(xs, .975)
    for name, tot, flds in [('iid (rows)', iid_tot, iid_fields), ('cluster (companies)', clus_tot, clus_fields)]:
        lo, hi = ci(tot)
        print(f"  {name:22s} total {base_S:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  half-width {(hi-lo)/2:.4f}")
        for f in FIELDS:
            flo, fhi = ci(flds[f])
            print(f"      {f:24s} {base_f[f]:.4f}  [{flo:.4f}, {fhi:.4f}]  hw {(fhi-flo)/2:.4f}")
    iid_hw = (ci(iid_tot)[1] - ci(iid_tot)[0]) / 2
    clus_hw = (ci(clus_tot)[1] - ci(clus_tot)[0]) / 2
    print(f"  --> cluster/iid CI half-width ratio = {clus_hw/iid_hw:.2f}x "
          f"(effective sample size shrinks by ~{(clus_hw/iid_hw)**2:.1f}x)")
    res['ci'] = {'iid': {'total': ci(iid_tot), 'half_width': iid_hw},
                 'cluster': {'total': ci(clus_tot), 'half_width': clus_hw},
                 'ratio': clus_hw / iid_hw,
                 'iid_fields': {f: ci(iid_fields[f]) for f in FIELDS},
                 'cluster_fields': {f: ci(clus_fields[f]) for f in FIELDS}}

    # ---------- (1b) is there any evaluation-side clustering to correct for? ----------
    # One-way random-effects ICC on per-item correctness, computed per field.
    print("\n=== (1b) INTRA-COMPANY CORRELATION of per-item correctness (ICC) ===")
    icc = {}
    for f in FIELDS:
        groups = []
        for c in comps:
            ys = [1.0 if r[f] == pipeline_by[str(r['id'])][f] else 0.0 for r in by_company[c]]
            if len(ys) >= 2:
                groups.append(ys)
        n_tot = sum(len(g) for g in groups)
        k = len(groups)
        grand = sum(sum(g) for g in groups) / n_tot
        msb_num = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups)
        msb = msb_num / (k - 1) if k > 1 else 0.0
        msw_num = sum(sum((y - sum(g) / len(g)) ** 2 for y in g) for g in groups)
        msw = msw_num / (n_tot - k) if n_tot > k else 0.0
        # average group size (Shrout-Fleiss n0 for unbalanced designs)
        n0 = (n_tot - sum(len(g) ** 2 for g in groups) / n_tot) / (k - 1) if k > 1 else 1.0
        val = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) > 0 else 0.0
        icc[f] = val
        print(f"  {f:24s} ICC = {val:+.4f}   (k={k} companies, mean size {n0:.1f})")
    print("  ICC near zero => item difficulty is not company-clustered at evaluation time;")
    print("  the company-leak problem documented for random K-fold is a TRAINING-side effect.")
    res['icc'] = icc

    # ---------- (2) degenerate rare classes ----------
    print("\n=== (2) RARE-CLASS DEGENERACY: replicates containing ZERO items of a class ===")
    supports = {}
    for f in FIELDS:
        for c in SCORED[f]:
            supports[(f, c)] = sum(1 for r in rows if r[f] == c)
    deg = {}
    for (f, c), s in sorted(supports.items(), key=lambda kv: kv[1]):
        if s > 60:
            continue
        analytic = (1 - s / N) ** N          # P(no item of the class in an iid replicate)
        deg[f"{f}:{c}"] = {'support': s, 'p_absent_analytic': analytic}
        print(f"  {f:24s} {c:22s} support {s:4d}   P(replicate has none) = {analytic:.4f}")
    print(f"  (for support 1 the limit is e^-1 = {math.exp(-1):.4f}: per-class F1 CIs are a"
          f" non-identifiable mixture, not an interval)")
    res['degenerate'] = deg
    res['supports'] = {f"{f}:{c}": s for (f, c), s in supports.items()}

    # ---------- (3) paired bootstrap over design decisions ----------
    print("\n=== (3) PAIRED BOOTSTRAP: is each binary source different from the deployed one? ===")
    sources = load_sources(args.sources)
    s4 = ['ckip_tapt_ep3', 'macbert_tapt', 'bgem3', 'bgem3_tapt']
    diluted = s4 + ['b0_ckip', 'macbert_base', 'roberta_wwm']
    named = {}
    if all(k in sources for k in s4):
        named['strong4(vote)'] = vote(sources, s4)
    if all(k in sources for k in diluted):
        named['diluted7(vote)'] = vote(sources, diluted)
    for k in ['ckip_tapt_ep3', 'macbert_tapt', 'bgem3', 'b0_ckip', 'macbert_base']:
        if k in sources:
            named[k] = sources[k]

    spliced = {k: splice(rows, v, pipeline_by) for k, v in named.items()
               if all(str(r['id']) in v for r in rows)}
    B2 = min(args.B, 4000)
    idx_sets = [[rng.randrange(N) for _ in range(N)] for _ in range(B2)]
    pair = {}
    if 'strong4(vote)' in spliced and 'diluted7(vote)' in spliced:
        a, b = spliced['strong4(vote)'], spliced['diluted7(vote)']
        sa, _ = score_rows(rows, a)
        sb, _ = score_rows(rows, b)
        diffs = []
        for ix in idx_sets:
            samp = [rows[i] for i in ix]
            diffs.append(score_rows(samp, a)[0] - score_rows(samp, b)[0])
        lo, hi = ci(diffs)
        p = 2 * min(sum(1 for d in diffs if d <= 0), sum(1 for d in diffs if d >= 0)) / len(diffs)
        pair['strong4 vs diluted7'] = {'obs': sa - sb, 'ci': [lo, hi], 'p': p}
        print(f"  strong4 ({sa:.4f}) vs diluted7 ({sb:.4f}): d={sa-sb:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}")
    res['paired'] = pair

    # ---------- (4) partition simulation ----------
    print("\n=== (4) PARTITION SIMULATION: two halves of a FIXED system's own test set ===")
    R = 5000
    gaps, gaps_f = [], defaultdict(list)
    order = list(range(N))
    for _ in range(R):
        rng.shuffle(order)
        h1 = [rows[i] for i in order[:N // 2]]
        h2 = [rows[i] for i in order[N // 2:]]
        t1, f1_ = score_rows(h1, pipeline_by)
        t2, f2_ = score_rows(h2, pipeline_by)
        gaps.append(t1 - t2)
        for f in FIELDS:
            gaps_f[f].append(f1_[f] - f2_[f])
    absg = sorted(abs(g) for g in gaps)
    sd = (sum(g * g for g in gaps) / len(gaps)) ** .5
    print(f"  |difference| between the two halves for the SAME system: "
          f"median {percentile(absg,.5):.4f}  90th pct {percentile(absg,.9):.4f}  "
          f"max {absg[-1]:.4f}  (SD of signed gap {sd:.4f})")
    for f in FIELDS:
        a = sorted(abs(g) for g in gaps_f[f])
        print(f"      {f:24s} median |d| {percentile(a,.5):.4f}   90th {percentile(a,.9):.4f}")
    res['partition'] = {'median_abs': percentile(absg, .5), 'p90_abs': percentile(absg, .9),
                        'max_abs': absg[-1], 'sd': sd,
                        'fields': {f: {'median_abs': percentile(sorted(abs(g) for g in gaps_f[f]), .5),
                                       'p90_abs': percentile(sorted(abs(g) for g in gaps_f[f]), .9)}
                                   for f in FIELDS}}

    # analytic one-sidedness for a rare class
    print("  analytic: for a class with support s split into two halves of a 2000-item test set,")
    for s in [1, 2, 3, 4]:
        p_one_sided = 2 ** (1 - s)
        print(f"      s={s}: P(all items land in ONE half) = {p_one_sided:.3f}"
              f"   -> that half's field F1 moves by up to 1/|C| = {1/3:.3f} => total {0.35/3:.4f}")
    res['analytic_one_sided'] = {s: 2 ** (1 - s) for s in [1, 2, 3, 4]}

    # ---------- (5) seed variance ----------
    print("\n=== (5) SEED VARIANCE (per-seed binary sources through the frozen pipeline) ===")
    fams = defaultdict(list)
    for name in sources:
        m = re.match(r'^(.*?)(?:_s\d+)?$', name)
        base = m.group(1) if m else name
        if re.search(r'_s\d+$', name) or name == base:
            fams[base].append(name)
    seed_stats = {}
    for base, members in sorted(fams.items()):
        vals = []
        for nm in members:
            v = sources[nm]
            if not all(str(r['id']) in v for r in rows):
                continue
            t, _ = score_rows(rows, splice(rows, v, pipeline_by))
            vals.append((nm, t))
        if len(vals) < 3:
            continue
        xs = [v for _, v in vals]
        mu = sum(xs) / len(xs)
        sd_ = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** .5
        seed_stats[base] = {'n': len(xs), 'mean': mu, 'sd': sd_,
                            'min': min(xs), 'max': max(xs),
                            'members': {nm: v for nm, v in vals}}
        print(f"  {base:22s} n={len(xs)}  mean {mu:.4f} +/- {sd_:.4f}  "
              f"range [{min(xs):.4f}, {max(xs):.4f}]  spread {max(xs)-min(xs):.4f}")
    if seed_stats:
        pooled = (sum(v['sd'] ** 2 * (v['n'] - 1) for v in seed_stats.values())
                  / sum(v['n'] - 1 for v in seed_stats.values())) ** .5
        print(f"  pooled within-family seed SD = {pooled:.4f}")
        print(f"  CI half-width (cluster bootstrap) = {clus_hw:.4f} "
              f"=> the validation instrument cannot resolve differences below "
              f"~{clus_hw:.3f}, i.e. ~{clus_hw/pooled:.1f}x the seed SD")
        res['seed'] = {'families': seed_stats, 'pooled_sd': pooled,
                       'ci_half_width_cluster': clus_hw,
                       'resolution_ratio': clus_hw / pooled}

    json.dump(res, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
