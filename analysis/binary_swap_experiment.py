"""Counterfactual binary-swap experiment: the cascade model's predictive test on val.

Design (mirrors the LB source-swap that produced the 0.6540 -> 0.6760 decomposition, but with
gold labels and n=42 independent binary sources instead of 4 leaderboard points):

  For each binary source B (an independently trained model / ensemble):
     final = cascade( B.promise, B.evidence, PIPELINE.raw_timeline, PIPELINE.raw_quality )
     observed  = official metric on `final`
     predicted = forward cascade model, using B's gate parameters (counted on val) and the
                 pipeline's downstream conditional confusions FROZEN at their deployed values

Nothing about the downstream modules changes across sources: every difference in timeline and
quality is cascade, by construction. The model must predict those untouched fields.

Usage:
  .venv_mac/bin/python paper/binary_swap_experiment.py \
      --pipeline_val paper/out/val_preds_Q0like.json \
      --sources paper/io/eason_valeval \
      --out paper/out/binary_swap.json
"""
import argparse, json, os, glob
from collections import defaultdict

from fit_cascade_from_val import (FIELDS, W, SCORED, TCLS, QCLS, macro_f1, f1_from_counts, score_all)

ECLS = SCORED['evidence_status']


def cascade(promise, evidence, timeline, quality):
    """Apply the competition schema cascade + compliance repick semantics of the deployed system."""
    if promise == 'No':
        return {'promise_status': 'No', 'evidence_status': 'N/A',
                'verification_timeline': 'N/A', 'evidence_quality': 'N/A'}
    q = quality
    if evidence in ('No', 'N/A'):
        q = 'N/A'
    return {'promise_status': 'Yes', 'evidence_status': evidence,
            'verification_timeline': timeline, 'evidence_quality': q}




def _rp_from_probs(pp, field, classes):
    """Per-item argmax over the real classes, matching the deployed compliance repick."""
    pr = pp.get('_probs_' + field)
    if not pr:
        return None
    LAB = {'verification_timeline': ['already', 'within_2_years', 'between_2_and_5_years',
                                     'more_than_5_years', 'N/A'],
           'evidence_quality': ['Clear', 'Not Clear', 'Misleading', 'N/A']}[field]
    reals = [i for i, l in enumerate(LAB) if l != 'N/A']
    return LAB[reals[int(max(range(len(reals)), key=lambda k: pr[reals[k]]))]]

def splice(rows, binary_by, pipeline_by):
    """Build spliced predictions: binary from `binary_by`, downstream (pre-cascade) from pipeline."""
    out = {}
    for r in rows:
        sid = str(r['id'])
        b = binary_by[sid]
        pp = pipeline_by[sid]
        t_raw = pp.get('_raw_verification_timeline', pp['verification_timeline'])
        q_raw = pp.get('_raw_evidence_quality', pp['evidence_quality'])
        if t_raw == 'N/A':
            t_raw = _rp_from_probs(pp, 'verification_timeline', None) or 'between_2_and_5_years'
        if q_raw == 'N/A':
            q_raw = _rp_from_probs(pp, 'evidence_quality', None) or 'Clear'
        out[sid] = cascade(b['promise_status'], b['evidence_status'], t_raw, q_raw)
    return out


def gate_params(rows, binary_by):
    """Count the gate parameters of a binary source on val (zero free parameters)."""
    p = {}
    gt_yes = [r for r in rows if r['promise_status'] == 'Yes']
    gt_no = [r for r in rows if r['promise_status'] == 'No']
    p['rho'] = sum(1 for r in gt_yes if binary_by[str(r['id'])]['promise_status'] == 'Yes') / len(gt_yes)
    p['phi'] = sum(1 for r in gt_no if binary_by[str(r['id'])]['promise_status'] == 'Yes') / len(gt_no)
    gT = {}
    for c in TCLS:
        sub = [r for r in rows if r['verification_timeline'] == c]
        gT[c] = (sum(1 for r in sub if binary_by[str(r['id'])]['promise_status'] == 'Yes') / len(sub)) if sub else None
    gQ = {}
    for c in QCLS:
        sub = [r for r in rows if r['evidence_quality'] == c]
        gQ[c] = (sum(1 for r in sub
                     if binary_by[str(r['id'])]['promise_status'] == 'Yes'
                     and binary_by[str(r['id'])]['evidence_status'] == 'Yes') / len(sub)) if sub else None
    p['gamma_T'], p['gamma_Q'] = gT, gQ
    ME, gE = {}, {}
    for c in ECLS:
        sub_all = [r for r in rows if r['evidence_status'] == c]
        sub = [r for r in sub_all if binary_by[str(r['id'])]['promise_status'] == 'Yes']
        gE[c] = (len(sub) / len(sub_all)) if sub_all else None
        row = {c2: 0.0 for c2 in ECLS}
        if sub:
            for r in sub:
                row[binary_by[str(r['id'])]['evidence_status']] += 1
            row = {k: v / len(sub) for k, v in row.items()}
        ME[c] = row
    p['M_evidence'], p['gamma_E'] = ME, gE
    return p


def downstream_params(rows, pipeline_by):
    """Conditional confusions of the FROZEN downstream modules, measured on gate-open rows.

    Conditioning uses the pipeline's own gate (the deployed operating point); these numbers are
    then held fixed across every binary source.
    """
    d = {}
    M = {}
    for c in TCLS:
        sub = [r for r in rows if r['verification_timeline'] == c
               and pipeline_by[str(r['id'])]['promise_status'] == 'Yes']
        row = {c2: 0.0 for c2 in TCLS}
        if sub:
            for r in sub:
                _pr = pipeline_by[str(r['id'])]
                pv = _pr.get('_raw_verification_timeline', _pr['verification_timeline'])
                if pv == 'N/A':
                    pv = _rp_from_probs(_pr, 'verification_timeline', None) or 'between_2_and_5_years'
                row[pv] += 1
            row = {k: v / len(sub) for k, v in row.items()}
        M[c] = row
    d['M_timeline'] = M
    R = {}
    for c in QCLS:
        sub = [r for r in rows if r['evidence_quality'] == c
               and pipeline_by[str(r['id'])]['promise_status'] == 'Yes'
               and pipeline_by[str(r['id'])]['evidence_status'] == 'Yes']
        row = {c2: 0.0 for c2 in QCLS}
        if sub:
            for r in sub:
                _pr = pipeline_by[str(r['id'])]
                pv = _pr.get('_raw_evidence_quality', _pr['evidence_quality'])
                if pv == 'N/A':
                    pv = _rp_from_probs(_pr, 'evidence_quality', None) or 'Clear'
                row[pv] += 1
            row = {k: v / len(sub) for k, v in row.items()}
        R[c] = row
    d['M_quality'] = R
    # module propensity on falsely-opened rows (GT N/A, gate open)
    U = {c: 0.0 for c in TCLS}
    fo = [r for r in rows if r['verification_timeline'] == 'N/A'
          and pipeline_by[str(r['id'])]['promise_status'] == 'Yes']
    for r in fo:
        _pr = pipeline_by[str(r['id'])]
        pv = _pr.get('_raw_verification_timeline', 'N/A')
        if pv == 'N/A':
            pv = _rp_from_probs(_pr, 'verification_timeline', None) or 'between_2_and_5_years'
        U[pv] += 1
    d['u_timeline'] = {k: (v / len(fo) if fo else 0.0) for k, v in U.items()}
    V = {c: 0.0 for c in QCLS}
    foq = [r for r in rows if r['evidence_quality'] == 'N/A'
           and pipeline_by[str(r['id'])]['promise_status'] == 'Yes'
           and pipeline_by[str(r['id'])]['evidence_status'] == 'Yes']
    for r in foq:
        _pr = pipeline_by[str(r['id'])]
        pv = _pr.get('_raw_evidence_quality', 'Clear')
        if pv == 'N/A':
            pv = _rp_from_probs(_pr, 'evidence_quality', None) or 'Clear'
        V[pv] += 1
    d['V_quality'] = {k: (v / len(foq) if foq else 0.0) for k, v in V.items()}
    return d


def forward(rows, gate, down, gamma_mode='class'):
    """Predict the four field macros from gate params + frozen downstream conditionals."""
    N = len(rows)
    dist = {}
    for f in FIELDS:
        c = defaultdict(int)
        for r in rows:
            c[r[f]] += 1
        dist[f] = {k: v / N for k, v in c.items()}
    rho, phi = gate['rho'], gate['phi']
    mP, mE, mT, mQ = (dist['promise_status'], dist['evidence_status'],
                      dist['verification_timeline'], dist['evidence_quality'])

    tpY, fnY = N * mP.get('Yes', 0) * rho, N * mP.get('Yes', 0) * (1 - rho)
    fpY = N * mP.get('No', 0) * phi
    tpN, fnN = N * mP.get('No', 0) * (1 - phi), N * mP.get('No', 0) * phi
    fpN = N * mP.get('Yes', 0) * (1 - rho)
    P = (f1_from_counts(tpY, fnY, fpY) + f1_from_counts(tpN, fnN, fpN)) / 2

    ME, gE = gate['M_evidence'], gate['gamma_E']
    def jE(gt, pl):
        g = (rho if gt != 'N/A' else phi) if (gamma_mode == 'const' or gE.get(gt) is None) else gE[gt]
        mass = N * mE.get(gt, 0)
        closed = mass * (1 - g) if pl == 'N/A' else 0.0
        return mass * g * ME[gt][pl] + closed
    E = sum(f1_from_counts(jE(c, c),
                           sum(jE(c, c2) for c2 in ECLS if c2 != c),
                           sum(jE(c2, c) for c2 in ECLS if c2 != c)) for c in ECLS) / len(ECLS)

    M, gT = down['M_timeline'], gate['gamma_T']
    U = down['u_timeline']
    T_per = {}
    for c in TCLS:
        g = rho if (gamma_mode == 'const' or gT.get(c) is None) else gT[c]
        tp = N * mT.get(c, 0) * g * M[c][c]
        fn = N * mT.get(c, 0) * (1 - g * M[c][c])
        fp = sum(N * mT.get(c2, 0) * (rho if (gamma_mode == 'const' or gT.get(c2) is None) else gT[c2]) * M[c2][c]
                 for c2 in TCLS if c2 != c)
        fp += N * mT.get('N/A', 0) * phi * U[c]
        T_per[c] = f1_from_counts(tp, fn, fp)
    T = sum(T_per.values()) / len(TCLS)

    R, gQ = down['M_quality'], gate['gamma_Q']
    V = down['V_quality']
    # quality gate mass on GT-N/A rows: promise falsely open AND evidence says Yes
    fo_mass = N * (mE.get('No', 0) * (gE.get('No') or rho) * ME['No']['Yes']
                   + mE.get('N/A', 0) * phi * ME['N/A']['Yes'])
    gate_default = rho * ME['Yes']['Yes']
    Q_per = {}
    for c in QCLS:
        g = gate_default if (gamma_mode == 'const' or gQ.get(c) is None) else gQ[c]
        tp = N * mQ.get(c, 0) * g * R[c][c]
        fn = N * mQ.get(c, 0) * (1 - g * R[c][c])
        fp = sum(N * mQ.get(c2, 0) * (gate_default if (gamma_mode == 'const' or gQ.get(c2) is None) else gQ[c2]) * R[c2][c]
                 for c2 in QCLS if c2 != c)
        fp += fo_mass * V[c]
        Q_per[c] = f1_from_counts(tp, fn, fp)
    Q = sum(Q_per.values()) / len(QCLS)

    fields = {'promise_status': P, 'evidence_status': E,
              'verification_timeline': T, 'evidence_quality': Q}
    return sum(W[f] * fields[f] for f in FIELDS), fields


def load_sources(src_dir):
    """Each valeval_* run dir contributes one independent binary source."""
    out = {}
    for d in sorted(glob.glob(os.path.join(src_dir, 'valeval_*'))):
        hits = glob.glob(os.path.join(d, '*', 'val_predictions.json'))
        if not hits:
            continue
        preds = json.load(open(sorted(hits)[-1], encoding='utf-8'))
        out[os.path.basename(d).replace('valeval_', '')] = {str(x['id']): x for x in preds}
    return out


def vote(sources, names):
    """Equal-weight majority vote over hard labels (ties -> first source's label)."""
    ids = set(sources[names[0]])
    for n in names[1:]:
        ids &= set(sources[n])
    out = {}
    for sid in ids:
        v = {}
        for f in ['promise_status', 'evidence_status']:
            c = defaultdict(int)
            for n in names:
                c[sources[n][sid][f]] += 1
            top = max(c.values())
            v[f] = next(sources[names[0]][sid][f] for _ in [0]) if list(c.values()).count(top) > 1 \
                and c[sources[names[0]][sid][f]] == top else max(c, key=lambda k: c[k])
        out[sid] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', default='../data_set/vpesg4k_val_1000.json')
    ap.add_argument('--pipeline_val', default='io/val_strong4.json')
    ap.add_argument('--sources', default='io/binary_sources')
    ap.add_argument('--out', default='out/binary_swap.json')
    args = ap.parse_args()

    rows_all = json.load(open(args.val, encoding='utf-8'))
    pipeline_by = {str(x['id']): x for x in json.load(open(args.pipeline_val, encoding='utf-8'))}
    rows = [r for r in rows_all if str(r['id']) in pipeline_by]

    sources = load_sources(args.sources)
    # the deployed pipeline itself is a source (self-consistency point)
    sources['PIPELINE(self)'] = pipeline_by
    # strong4 = the deployed binary roster, as a majority vote of its 4 members
    s4 = ['ckip_tapt_ep3', 'macbert_tapt', 'bgem3', 'bgem3_tapt']
    if all(k in sources for k in s4):
        sources['strong4(vote)'] = vote(sources, s4)
    print(f"{len(sources)} binary sources; {len(rows)} val rows")

    down = downstream_params(rows, pipeline_by)
    results = []
    for name, bby in sorted(sources.items()):
        if not all(str(r['id']) in bby for r in rows):
            continue
        spl = splice(rows, bby, pipeline_by)
        obs_tot, obs_f, _ = score_all(rows, spl)
        gate = gate_params(rows, bby)
        for mode in ['const', 'class']:
            pred_tot, pred_f = forward(rows, gate, down, gamma_mode=mode)
            results.append({'source': name, 'mode': mode, 'rho': gate['rho'], 'phi': gate['phi'],
                            'obs_total': obs_tot, 'pred_total': pred_tot,
                            'obs': obs_f, 'pred': pred_f})

    # report
    print("\n=== BINARY-SWAP: predicted vs observed (downstream FROZEN) ===")
    hdr = f"{'source':22s} {'obsP':>6s} {'obsT':>6s} {'prdT':>6s} {'errT':>7s} {'obsQ':>6s} {'prdQ':>6s} {'errQ':>7s} {'obsS':>6s} {'prdS':>6s} {'errS':>7s}"
    print(hdr)
    errs = {'const': defaultdict(list), 'class': defaultdict(list)}
    for mode in ['class', 'const']:
        print(f"--- gamma mode = {mode} ---")
        for r in [x for x in results if x['mode'] == mode]:
            eT = r['pred']['verification_timeline'] - r['obs']['verification_timeline']
            eQ = r['pred']['evidence_quality'] - r['obs']['evidence_quality']
            eS = r['pred_total'] - r['obs_total']
            errs[mode]['T'].append(eT); errs[mode]['Q'].append(eQ); errs[mode]['S'].append(eS)
            print(f"{r['source']:22s} {r['obs']['promise_status']:6.3f} "
                  f"{r['obs']['verification_timeline']:6.3f} {r['pred']['verification_timeline']:6.3f} {eT:+7.4f} "
                  f"{r['obs']['evidence_quality']:6.3f} {r['pred']['evidence_quality']:6.3f} {eQ:+7.4f} "
                  f"{r['obs_total']:6.3f} {r['pred_total']:6.3f} {eS:+7.4f}")
    print("\n=== ERROR SUMMARY (mean absolute error across sources) ===")
    for mode in ['const', 'class']:
        n = len(errs[mode]['T'])
        mae = {k: sum(abs(x) for x in v) / len(v) for k, v in errs[mode].items()}
        print(f"gamma={mode:5s} (n={n}): MAE timeline {mae['T']:.4f}  quality {mae['Q']:.4f}  total {mae['S']:.4f}")

    json.dump({'downstream': down, 'results': results}, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
