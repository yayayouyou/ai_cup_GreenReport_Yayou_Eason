"""Load-bearing experiment: fit the cascade model from DIRECTLY COUNTED val confusion matrices.

Replaces the demo-run identification (which anchored gate params to LB per-field macros —
circular) with zero-free-parameter estimation from val-1000 predictions + gold labels.

Estimated quantities (all counts on val-1000):
  Gate (upstream):
    rho   = P(pred promise=Yes | GT promise=Yes)          promise recall
    phi   = P(pred promise=Yes | GT promise=No)           promise false-open rate
    gamma_c = P(gate open | GT field f = c)               CLASS-CONDITIONAL upstream recall  <-- v2
    eYY   = P(pred E=Yes | GT E=Yes, gate open)
    eNN   = P(pred E=No  | GT E=No , gate open)
    qy    = P(pred E=Yes | GT E=N/A, falsely opened)
  Downstream (conditional on gate open):
    M[c][c'] = P(pred T=c' | GT T=c, gate open)           4x4 conditional confusion
    r[c]     = P(pred Q=c  | GT Q=c , quality gate open)
    u_c / V_c = class propensity of the module on FALSELY-opened rows

Then:
  1. self-consistency: rebuild the val field macros from counts, compare to direct scoring
  2. v1 (class-independent gamma) vs v2 (measured gamma_c) forward prediction of LB pairs
  3. effective weights + per-error price list at the fitted operating point

Usage:
  .venv_mac/bin/python paper/fit_cascade_from_val.py \
      --val data_set/vpesg4k_val_1000.json \
      --pred paper/io/val_pipeline_preds.json \
      --out paper/out/cascade_params.json
"""
import argparse, json, os
from collections import defaultdict

FIELDS = ['promise_status', 'evidence_status', 'evidence_quality', 'verification_timeline']
W = {'promise_status': .20, 'evidence_status': .30, 'evidence_quality': .35, 'verification_timeline': .15}
SCORED = {
    'promise_status': ['Yes', 'No'],
    'evidence_status': ['Yes', 'No', 'N/A'],
    'evidence_quality': ['Clear', 'Not Clear', 'Misleading'],
    'verification_timeline': ['already', 'within_2_years', 'between_2_and_5_years', 'more_than_5_years'],
}
TCLS = SCORED['verification_timeline']
QCLS = SCORED['evidence_quality']


def f1_from_counts(tp, fn, fp):
    den = 2 * tp + fn + fp
    return 2 * tp / den if den > 0 else 0.0


def macro_f1(gold, pred, labels):
    """sklearn f1_score(labels=..., average='macro', zero_division=0) reimplemented on counts."""
    out = {}
    for c in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        out[c] = f1_from_counts(tp, fn, fp)
    return sum(out.values()) / len(labels), out


def score_all(rows, pred_by):
    """Official weighted metric on a set of rows given predictions dict id->picks."""
    fields, per_class = {}, {}
    for f in FIELDS:
        gold = [r[f] for r in rows]
        pred = [pred_by[str(r['id'])][f] for r in rows]
        fields[f], per_class[f] = macro_f1(gold, pred, SCORED[f])
    total = sum(W[f] * fields[f] for f in FIELDS)
    return total, fields, per_class


def fit_params(rows, pred_by):
    """Directly count every model parameter. No optimization, no free parameters."""
    p = {}
    n = defaultdict(int)

    # ---- gate: promise ----
    gt_yes = [r for r in rows if r['promise_status'] == 'Yes']
    gt_no = [r for r in rows if r['promise_status'] == 'No']
    open_yes = sum(1 for r in gt_yes if pred_by[str(r['id'])]['promise_status'] == 'Yes')
    open_no = sum(1 for r in gt_no if pred_by[str(r['id'])]['promise_status'] == 'Yes')
    p['rho'] = open_yes / len(gt_yes)
    p['phi'] = open_no / len(gt_no)
    n['gt_promise_yes'], n['gt_promise_no'] = len(gt_yes), len(gt_no)

    # ---- class-conditional gate recall gamma_c (v2 term) ----
    # For timeline: gate = pred promise Yes. For quality: gate = pred promise Yes AND pred evidence Yes.
    gamma_T, gamma_Q = {}, {}
    for c in TCLS:
        sub = [r for r in rows if r['verification_timeline'] == c]
        gamma_T[c] = (sum(1 for r in sub if pred_by[str(r['id'])]['promise_status'] == 'Yes') / len(sub)) if sub else None
        n[f'gt_T_{c}'] = len(sub)
    for c in QCLS:
        sub = [r for r in rows if r['evidence_quality'] == c]
        gamma_Q[c] = (sum(1 for r in sub
                          if pred_by[str(r['id'])]['promise_status'] == 'Yes'
                          and pred_by[str(r['id'])]['evidence_status'] == 'Yes') / len(sub)) if sub else None
        n[f'gt_Q_{c}'] = len(sub)
    p['gamma_T'], p['gamma_Q'] = gamma_T, gamma_Q

    # ---- evidence: FULL 3x3 conditional confusion given promise gate open ----
    # (the module may still emit N/A even when the gate is open, so two scalars do not suffice)
    ECLS = SCORED['evidence_status']
    ME, gamma_E = {}, {}
    for c in ECLS:
        sub_all = [r for r in rows if r['evidence_status'] == c]
        sub = [r for r in sub_all if pred_by[str(r['id'])]['promise_status'] == 'Yes']
        gamma_E[c] = (len(sub) / len(sub_all)) if sub_all else None
        row = {c2: 0.0 for c2 in ECLS}
        if sub:
            for r in sub:
                row[pred_by[str(r['id'])]['evidence_status']] += 1
            row = {k: v / len(sub) for k, v in row.items()}
        ME[c] = row
        n[f'gt_E_{c}'] = len(sub_all)
        n[f'open_E_{c}'] = len(sub)
    p['M_evidence'], p['gamma_E'] = ME, gamma_E
    # legacy scalars kept for reporting/comparison with the analytical model
    p['eYY'] = ME['Yes']['Yes']
    p['eNN'] = ME['No']['No']
    p['qy'] = ME['N/A']['Yes']
    n['ev_open_gtYes'] = n['open_E_Yes']
    n['ev_open_gtNo'] = n['open_E_No']
    n['ev_open_gtNA'] = n['open_E_N/A']

    # ---- downstream timeline conditional confusion M[c][c'] (gate open only) ----
    M = {}
    for c in TCLS:
        sub = [r for r in rows if r['verification_timeline'] == c
               and pred_by[str(r['id'])]['promise_status'] == 'Yes']
        row = {c2: 0.0 for c2 in TCLS}
        row['N/A'] = 0.0
        if sub:
            for r in sub:
                pv = pred_by[str(r['id'])]['verification_timeline']
                row[pv if pv in row else 'N/A'] += 1
            row = {k: v / len(sub) for k, v in row.items()}
        M[c] = row
        n[f'open_T_{c}'] = len(sub)
    p['M_timeline'] = M

    # ---- downstream quality conditional r[c] (quality gate open only) ----
    R = {}
    for c in QCLS:
        sub = [r for r in rows if r['evidence_quality'] == c
               and pred_by[str(r['id'])]['promise_status'] == 'Yes'
               and pred_by[str(r['id'])]['evidence_status'] == 'Yes']
        row = {c2: 0.0 for c2 in QCLS}
        row['N/A'] = 0.0
        if sub:
            for r in sub:
                pv = pred_by[str(r['id'])]['evidence_quality']
                row[pv if pv in row else 'N/A'] += 1
            row = {k: v / len(sub) for k, v in row.items()}
        R[c] = row
        n[f'open_Q_{c}'] = len(sub)
    p['M_quality'] = R

    # ---- module propensity on FALSELY-opened rows (GT N/A but gate opened) ----
    fo_T = [r for r in rows if r['verification_timeline'] == 'N/A'
            and pred_by[str(r['id'])]['promise_status'] == 'Yes']
    u = {c: 0.0 for c in TCLS}
    for r in fo_T:
        pv = pred_by[str(r['id'])]['verification_timeline']
        if pv in u:
            u[pv] += 1
    p['u_timeline'] = {k: (v / len(fo_T) if fo_T else 0.0) for k, v in u.items()}
    n['falsely_open_T'] = len(fo_T)

    fo_Q = [r for r in rows if r['evidence_quality'] == 'N/A'
            and pred_by[str(r['id'])]['promise_status'] == 'Yes'
            and pred_by[str(r['id'])]['evidence_status'] == 'Yes']
    v_ = {c: 0.0 for c in QCLS}
    for r in fo_Q:
        pv = pred_by[str(r['id'])]['evidence_quality']
        if pv in v_:
            v_[pv] += 1
    p['V_quality'] = {k: (val / len(fo_Q) if fo_Q else 0.0) for k, val in v_.items()}
    n['falsely_open_Q'] = len(fo_Q)

    # ---- label distribution (population priors) ----
    dist = {}
    for f in FIELDS:
        d = defaultdict(int)
        for r in rows:
            d[r[f]] += 1
        dist[f] = {k: v / len(rows) for k, v in sorted(d.items())}
    p['label_dist'] = dist
    p['n_rows'] = len(rows)
    p['counts'] = dict(n)
    return p


def rebuild_fields(p, gamma_mode='class'):
    """Forward-simulate the field macros from fitted parameters.

    gamma_mode='const' -> v1 model (class-independent gate recall = rho)
    gamma_mode='class' -> v2 model (measured gamma_c)
    Returns dict field -> macro, using expected counts over the val population.
    """
    dist = p['label_dist']
    N = p['n_rows']
    rho, phi = p['rho'], p['phi']
    eYY, eNN, qy = p['eYY'], p['eNN'], p['qy']
    M, R = p['M_timeline'], p['M_quality']
    U, V = p['u_timeline'], p['V_quality']

    mP = dist['promise_status']
    mE = dist['evidence_status']
    mT = dist['verification_timeline']
    mQ = dist['evidence_quality']

    # promise field
    tpY = N * mP.get('Yes', 0) * rho
    fnY = N * mP.get('Yes', 0) * (1 - rho)
    fpY = N * mP.get('No', 0) * phi
    tpN = N * mP.get('No', 0) * (1 - phi)
    fnN = N * mP.get('No', 0) * phi
    fpN = N * mP.get('Yes', 0) * (1 - rho)
    P_macro = (f1_from_counts(tpY, fnY, fpY) + f1_from_counts(tpN, fnN, fpN)) / 2

    # evidence field (3-class incl N/A): gate closed -> N/A; gate open -> full conditional confusion
    ECLS = SCORED['evidence_status']
    ME, gE = p['M_evidence'], p['gamma_E']
    def joint_E(gt, pred_lab):
        """expected count of (GT=gt, pred=pred_lab)."""
        g = rho if gamma_mode == 'const' or gE.get(gt) is None else gE[gt]
        if gt == 'N/A':
            g = phi if gamma_mode == 'const' or gE.get(gt) is None else gE[gt]
        mass = N * mE.get(gt, 0)
        closed = mass * (1 - g) if pred_lab == 'N/A' else 0.0
        return mass * g * ME[gt][pred_lab] + closed
    E_per = {}
    for c in ECLS:
        # gate-closed rows of GT=c are already routed to pred N/A inside joint_E
        tp = joint_E(c, c)
        fn = sum(joint_E(c, c2) for c2 in ECLS if c2 != c)
        fp = sum(joint_E(c2, c) for c2 in ECLS if c2 != c)
        E_per[c] = f1_from_counts(tp, fn, fp)
    E_macro = sum(E_per.values()) / len(ECLS)

    # timeline field (N/A excluded)
    gT = p['gamma_T']
    T_per = {}
    for c in TCLS:
        g = rho if gamma_mode == 'const' or gT.get(c) is None else gT[c]
        tp = N * mT.get(c, 0) * g * M[c][c]
        fn = N * mT.get(c, 0) * (1 - g * M[c][c])
        fp = sum(N * mT.get(c2, 0) * (rho if gamma_mode == 'const' or gT.get(c2) is None else gT[c2]) * M[c2][c]
                 for c2 in TCLS if c2 != c)
        fp += N * mT.get('N/A', 0) * phi * U[c]
        T_per[c] = f1_from_counts(tp, fn, fp)
    T_macro = sum(T_per.values()) / len(TCLS)

    # quality field (N/A excluded, 3 classes incl Misleading)
    gQ = p['gamma_Q']
    gate_default = rho * eYY
    fo_Q_mass = N * (mE.get('No', 0) * rho * (1 - eNN) + mE.get('N/A', 0) * phi * qy)
    Q_per = {}
    for c in QCLS:
        g = gate_default if gamma_mode == 'const' or gQ.get(c) is None else gQ[c]
        mc = mQ.get(c, 0)
        rcc = R[c][c] if c in R else 0.0
        tp = N * mc * g * rcc
        fn = N * mc * (1 - g * rcc)
        fp = sum(N * mQ.get(c2, 0) * (gate_default if gamma_mode == 'const' or gQ.get(c2) is None else gQ[c2]) * R[c2][c]
                 for c2 in QCLS if c2 != c)
        fp += fo_Q_mass * V[c]
        Q_per[c] = f1_from_counts(tp, fn, fp)
    Q_macro = sum(Q_per.values()) / len(QCLS)

    fields = {'promise_status': P_macro, 'evidence_status': E_macro,
              'verification_timeline': T_macro, 'evidence_quality': Q_macro}
    total = sum(W[f] * fields[f] for f in FIELDS)
    return total, fields, {'timeline': T_per, 'quality': Q_per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', default='../data_set/vpesg4k_val_1000.json')
    ap.add_argument('--pred', required=True, help='val predictions json: list of {id, 4 fields}')
    ap.add_argument('--out', default='out/cascade_params.json')
    ap.add_argument('--label', default='pipeline')
    args = ap.parse_args()

    rows = json.load(open(args.val, encoding='utf-8'))
    preds = json.load(open(args.pred, encoding='utf-8'))
    pred_by = {str(x['id']): x for x in preds}
    rows = [r for r in rows if str(r['id']) in pred_by]

    total, fields, per_class = score_all(rows, pred_by)
    print(f"=== DIRECT SCORING ({args.label}, n={len(rows)}) ===")
    print(f"WF1={total:.4f}  P={fields['promise_status']:.4f} E={fields['evidence_status']:.4f} "
          f"T={fields['verification_timeline']:.4f} Q={fields['evidence_quality']:.4f}")

    p = fit_params(rows, pred_by)
    print("\n=== FITTED PARAMETERS (direct counts, zero free parameters) ===")
    print(f"gate:  rho={p['rho']:.4f} (promise recall, n={p['counts']['gt_promise_yes']})  "
          f"phi={p['phi']:.4f} (false-open, n={p['counts']['gt_promise_no']})")
    print(f"evid:  eYY={p['eYY']:.4f} (n={p['counts']['ev_open_gtYes']})  "
          f"eNN={p['eNN']:.4f} (n={p['counts']['ev_open_gtNo']})  "
          f"qy={p['qy']:.4f} (n={p['counts']['ev_open_gtNA']})")
    print("gamma_T (class-conditional gate recall):  " +
          "  ".join(f"{c}={p['gamma_T'][c]:.3f}(n={p['counts'][f'gt_T_{c}']})"
                    for c in TCLS if p['gamma_T'][c] is not None))
    print("gamma_Q:  " + "  ".join(f"{c}={p['gamma_Q'][c]:.3f}(n={p['counts'][f'gt_Q_{c}']})"
                                   for c in QCLS if p['gamma_Q'][c] is not None))

    print("\n=== SELF-CONSISTENCY: forward model vs direct scoring ===")
    for mode in ['const', 'class']:
        tot_m, fld_m, _ = rebuild_fields(p, gamma_mode=mode)
        name = 'v1 (gamma const)' if mode == 'const' else 'v2 (gamma_c measured)'
        print(f"{name:24s} WF1={tot_m:.4f} (d{tot_m-total:+.4f})  "
              f"P={fld_m['promise_status']:.4f}(Δ{fld_m['promise_status']-fields['promise_status']:+.4f}) "
              f"E={fld_m['evidence_status']:.4f}(Δ{fld_m['evidence_status']-fields['evidence_status']:+.4f}) "
              f"T={fld_m['verification_timeline']:.4f}(Δ{fld_m['verification_timeline']-fields['verification_timeline']:+.4f}) "
              f"Q={fld_m['evidence_quality']:.4f}(Δ{fld_m['evidence_quality']-fields['evidence_quality']:+.4f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {'label': args.label, 'direct': {'total': total, 'fields': fields, 'per_class': per_class},
           'params': p}
    json.dump(out, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"\nwrote params -> {args.out}")


if __name__ == '__main__':
    main()
