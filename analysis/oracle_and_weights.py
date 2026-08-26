"""RQ1 + RQ3 evidence: oracle 2x2 gold-splice, effective weights, and the metric-implied price list.

Three experiments, all on val-1000 with the official N/A-excluded scorer, no GPU:

  (A) ORACLE 2x2 — separates DIRECT binary points from CASCADE points empirically.
        scored-source x gate-source in {model, gold}:
          B   = model scored, model gate      (deployed reference)
          G   = model scored, GOLD gate       -> pure cascade points
          C   = GOLD scored,  model gate      -> pure direct points
          O   = GOLD scored,  GOLD gate       -> total recoverable headroom
      plus O_P / O_E (gold promise only / gold evidence only) and a RESCUE-ROW diagnostic
      (are gate-rescued rows harder for the downstream module than already-open rows?).

  (B) EFFECTIVE WEIGHT — numerically differentiate the fitted forward model at the deployed
      operating point: dS/dM_promise, decomposed by which field the value lands in.

  (C) PRICE LIST — value of repairing one error of each type, in total-score points.

Usage:
  .venv_mac/bin/python paper/oracle_and_weights.py --pipeline_val paper/out/val_preds_Q0like.json
"""
import argparse, json
from collections import defaultdict

from fit_cascade_from_val import FIELDS, W, SCORED, TCLS, QCLS, f1_from_counts, score_all
from binary_swap_experiment import cascade, gate_params, downstream_params, forward, ECLS, _rp_from_probs


def raw_downstream(pipeline_by, sid):
    p = pipeline_by[sid]
    t = p.get('_raw_verification_timeline', p['verification_timeline'])
    q = p.get('_raw_evidence_quality', p['evidence_quality'])
    if t == 'N/A':
        t = _rp_from_probs(p, 'verification_timeline', None) or 'between_2_and_5_years'
    if q == 'N/A':
        q = _rp_from_probs(p, 'evidence_quality', None) or 'Clear'
    return t, q


def build(rows, pipeline_by, gold_promise, gold_evidence, gold_downstream=False):
    """Assemble predictions with selectable gold/model sources for gate and downstream."""
    out = {}
    for r in rows:
        sid = str(r['id'])
        p = pipeline_by[sid]
        pr = r['promise_status'] if gold_promise else p['promise_status']
        ev = r['evidence_status'] if gold_evidence else p['evidence_status']
        if gold_evidence and ev == 'N/A':
            # gold evidence is N/A exactly when gold promise is No; keep it consistent with the gate
            ev = 'N/A' if pr == 'No' else 'Yes'
        t_raw, q_raw = raw_downstream(pipeline_by, sid)
        if gold_downstream:
            t_raw = r['verification_timeline'] if r['verification_timeline'] != 'N/A' else t_raw
            q_raw = r['evidence_quality'] if r['evidence_quality'] != 'N/A' else q_raw
        out[sid] = cascade(pr, ev, t_raw, q_raw)
    return out


def fmt(tag, tot, f):
    return (f"{tag:34s} S={tot:.4f}  P={f['promise_status']:.4f} E={f['evidence_status']:.4f} "
            f"T={f['verification_timeline']:.4f} Q={f['evidence_quality']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', default='../data_set/vpesg4k_val_1000.json')
    ap.add_argument('--pipeline_val', default='io/val_strong4.json')
    ap.add_argument('--out', default='out/oracle_weights.json')
    args = ap.parse_args()

    rows_all = json.load(open(args.val, encoding='utf-8'))
    pipeline_by = {str(x['id']): x for x in json.load(open(args.pipeline_val, encoding='utf-8'))}
    rows = [r for r in rows_all if str(r['id']) in pipeline_by]
    N = len(rows)
    res = {}

    # ---------------- (A) ORACLE 2x2 ----------------
    print("=== (A) ORACLE 2x2 GOLD-SPLICE (val, official N/A-excluded metric) ===")
    configs = {
        'B  deployed (model gate, model scored)': (False, False),
        'O_P gold promise only': (True, False),
        'O_E gold evidence only': (False, True),
        'O_PE gold promise + evidence': (True, True),
    }
    for tag, (gp, ge) in configs.items():
        preds = build(rows, pipeline_by, gp, ge)
        tot, f, _ = score_all(rows, preds)
        res[tag] = {'total': tot, 'fields': f}
        print(fmt(tag, tot, f))

    base_tot = res['B  deployed (model gate, model scored)']['total']
    o_tot = res['O_PE gold promise + evidence']['total']
    print(f"\nTotal cascade-recoverable headroom (O_PE - B) = {o_tot - base_tot:+.4f}")
    bf = res['B  deployed (model gate, model scored)']['fields']
    of = res['O_PE gold promise + evidence']['fields']
    direct = W['promise_status'] * (of['promise_status'] - bf['promise_status']) + \
             W['evidence_status'] * (of['evidence_status'] - bf['evidence_status'])
    casc = W['verification_timeline'] * (of['verification_timeline'] - bf['verification_timeline']) + \
           W['evidence_quality'] * (of['evidence_quality'] - bf['evidence_quality'])
    print(f"  decomposition: DIRECT (P,E fields) {direct:+.4f}   CASCADE (T,Q untouched) {casc:+.4f}"
          f"   cascade share = {100*casc/(direct+casc):.1f}%")
    print(f"  intrinsic downstream ceilings under a perfect gate: "
          f"timeline {of['verification_timeline']:.4f}  quality {of['evidence_quality']:.4f}")
    res['headroom'] = {'total': o_tot - base_tot, 'direct': direct, 'cascade': casc,
                       'cascade_share': casc / (direct + casc)}

    # ---------------- rescue-row diagnostic ----------------
    print("\n=== RESCUE-ROW DIAGNOSTIC (is the cascade dividend real, or are rescued rows harder?) ===")
    open_rows = [r for r in rows if pipeline_by[str(r['id'])]['promise_status'] == 'Yes'
                 and r['promise_status'] == 'Yes' and r['verification_timeline'] != 'N/A']
    rescue_rows = [r for r in rows if pipeline_by[str(r['id'])]['promise_status'] == 'No'
                   and r['promise_status'] == 'Yes' and r['verification_timeline'] != 'N/A']
    def cond_acc(rs):
        if not rs:
            return None, 0
        hit = sum(1 for r in rs if raw_downstream(pipeline_by, str(r['id']))[0] == r['verification_timeline'])
        return hit / len(rs), len(rs)
    a_open, n_open = cond_acc(open_rows)
    a_res, n_res = cond_acc(rescue_rows)
    print(f"timeline conditional accuracy: already-open rows {a_open:.4f} (n={n_open})   "
          f"gate-rescued rows {a_res:.4f} (n={n_res})   penalty {a_open - a_res:+.4f}")
    resq_open = [r for r in rows if pipeline_by[str(r['id'])]['promise_status'] == 'Yes'
                 and pipeline_by[str(r['id'])]['evidence_status'] == 'Yes' and r['evidence_quality'] != 'N/A']
    resq_res = [r for r in rows if r['evidence_quality'] != 'N/A'
                and not (pipeline_by[str(r['id'])]['promise_status'] == 'Yes'
                         and pipeline_by[str(r['id'])]['evidence_status'] == 'Yes')]
    def cond_accq(rs):
        if not rs:
            return None, 0
        hit = sum(1 for r in rs if raw_downstream(pipeline_by, str(r['id']))[1] == r['evidence_quality'])
        return hit / len(rs), len(rs)
    q_open, nq_open = cond_accq(resq_open)
    q_res, nq_res = cond_accq(resq_res)
    print(f"quality  conditional accuracy: already-open rows {q_open:.4f} (n={nq_open})   "
          f"gate-rescued rows {q_res:.4f} (n={nq_res})   penalty {q_open - q_res:+.4f}")
    res['rescue'] = {'timeline': {'open': a_open, 'rescued': a_res, 'n_open': n_open, 'n_rescued': n_res},
                     'quality': {'open': q_open, 'rescued': q_res, 'n_open': nq_open, 'n_rescued': nq_res}}

    # ---------------- (B) EFFECTIVE WEIGHTS ----------------
    print("\n=== (B) EFFECTIVE WEIGHT OF THE BINARY (numeric, at the deployed operating point) ===")
    gate = gate_params(rows, pipeline_by)
    down = downstream_params(rows, pipeline_by)

    def S_fields(rho=None, phi=None):
        g = json.loads(json.dumps(gate))
        if rho is not None:
            scale = rho / gate['rho']
            g['rho'] = rho
            g['gamma_T'] = {k: (min(1.0, v * scale) if v is not None else None) for k, v in gate['gamma_T'].items()}
            g['gamma_Q'] = {k: (min(1.0, v * scale) if v is not None else None) for k, v in gate['gamma_Q'].items()}
            g['gamma_E'] = {k: (min(1.0, v * scale) if v is not None else None) if k != 'N/A' else v
                            for k, v in gate['gamma_E'].items()}
        if phi is not None:
            g['phi'] = phi
            g['gamma_E'] = dict(g['gamma_E'], **{'N/A': phi})
        return forward(rows, g, down, gamma_mode='class')

    d = 1e-3
    s_hi, f_hi = S_fields(rho=min(1.0, gate['rho'] + d))
    s_lo, f_lo = S_fields(rho=gate['rho'] - d)
    dS = (s_hi - s_lo) / (2 * d)
    dMP = (f_hi['promise_status'] - f_lo['promise_status']) / (2 * d)
    print(f"dS/drho = {dS:+.4f}   dM_promise/drho = {dMP:+.4f}   "
          f"=> W_eff(promise via recall) = {dS/dMP:.4f}   (nominal {W['promise_status']})")
    comp = {}
    for f in FIELDS:
        comp[f] = W[f] * (f_hi[f] - f_lo[f]) / (2 * d)
    tot_comp = sum(comp.values())
    SHORT = {'promise_status': 'P', 'evidence_status': 'E',
             'evidence_quality': 'Q', 'verification_timeline': 'T'}
    print("  where the value lands (weighted dM/drho):  " +
          "  ".join(f"{SHORT[f]}={comp[f]:+.4f}" for f in FIELDS))
    outside = 1 - comp['promise_status'] / tot_comp if tot_comp else 0
    print(f"  share of promise's marginal value falling OUTSIDE the promise field: {100*outside:.1f}%")

    sp_hi, fp_hi = S_fields(phi=max(0.0, gate['phi'] - d))
    sp_lo, fp_lo = S_fields(phi=gate['phi'] + d)
    dSp = (sp_hi - sp_lo) / (2 * d)
    dMPp = (fp_hi['promise_status'] - fp_lo['promise_status']) / (2 * d)
    print(f"W_eff(promise via specificity) = {dSp/dMPp:.4f}")
    res['effective_weight'] = {'via_recall': dS / dMP, 'via_specificity': dSp / dMPp,
                               'components': comp, 'share_outside': outside,
                               'rho': gate['rho'], 'phi': gate['phi']}

    # ---------------- (C) PRICE LIST ----------------
    print("\n=== (C) PRICE LIST: total-score points from repairing ONE error (x1e-4) ===")
    base_preds = build(rows, pipeline_by, False, False)
    base_S, base_f, _ = score_all(rows, base_preds)

    def repair(pick_fn, label, limit=None):
        """Flip specific rows to gold and measure the score change."""
        preds = dict(base_preds)
        touched = 0
        for r in rows:
            sid = str(r['id'])
            if pick_fn(r, pipeline_by[sid]):
                p = pipeline_by[sid]
                t_raw, q_raw = raw_downstream(pipeline_by, sid)
                preds[sid] = cascade(r['promise_status'],
                                     r['evidence_status'] if r['promise_status'] == 'Yes' else 'N/A',
                                     t_raw, q_raw)
                touched += 1
                if limit and touched >= limit:
                    break
        if touched == 0:
            return None, 0
        tot, _, _ = score_all(rows, preds)
        return (tot - base_S) / touched * 1e4, touched

    kinds = {
        'promise FN (gold Yes, pred No)': lambda r, p: r['promise_status'] == 'Yes' and p['promise_status'] == 'No',
        '  ...on a within_2_years item': lambda r, p: r['promise_status'] == 'Yes' and p['promise_status'] == 'No'
                                                      and r['verification_timeline'] == 'within_2_years',
        '  ...on an already item': lambda r, p: r['promise_status'] == 'Yes' and p['promise_status'] == 'No'
                                                and r['verification_timeline'] == 'already',
        'promise FP (gold No, pred Yes)': lambda r, p: r['promise_status'] == 'No' and p['promise_status'] == 'Yes',
        'evidence error (gate already open)': lambda r, p: r['promise_status'] == 'Yes' and p['promise_status'] == 'Yes'
                                                           and r['evidence_status'] != p['evidence_status'],
    }
    price = {}
    for label, fn in kinds.items():
        v, n = repair(fn, label)
        price[label] = {'points_per_error_x1e4': v, 'n_available': n}
        print(f"  {label:38s} {v:8.2f}   (n={n} such errors on val)" if v is not None
              else f"  {label:38s} {'--':>8s}   (n=0)")

    # downstream-only repairs, for contrast (no cascade component)
    def repair_downstream(field, label):
        preds = {}
        touched = 0
        for r in rows:
            sid = str(r['id'])
            p = pipeline_by[sid]
            t_raw, q_raw = raw_downstream(pipeline_by, sid)
            if field == 'T' and p['promise_status'] == 'Yes' and r['verification_timeline'] not in ('N/A',) \
                    and t_raw != r['verification_timeline']:
                t_raw = r['verification_timeline']; touched += 1
            if field == 'Q' and p['promise_status'] == 'Yes' and p['evidence_status'] == 'Yes' \
                    and r['evidence_quality'] not in ('N/A',) and q_raw != r['evidence_quality']:
                q_raw = r['evidence_quality']; touched += 1
            preds[sid] = cascade(p['promise_status'], p['evidence_status'], t_raw, q_raw)
        tot, _, _ = score_all(rows, preds)
        return (tot - base_S) / touched * 1e4 if touched else None, touched
    for fld, lab in [('T', 'timeline conditional fix (all)'), ('Q', 'quality conditional fix (all)')]:
        v, n = repair_downstream(fld, lab)
        price[lab] = {'points_per_error_x1e4': v, 'n_available': n}
        print(f"  {lab:38s} {v:8.2f}   (n={n})" if v is not None else f"  {lab:38s} {'--':>8s}   (n=0)")

    # one true Misleading: analytic (support-1 class on val)
    mis_rows = [r for r in rows if r['evidence_quality'] == 'Misleading']
    if mis_rows:
        preds = dict(base_preds)
        for r in mis_rows:
            sid = str(r['id'])
            preds[sid] = cascade('Yes', 'Yes', raw_downstream(pipeline_by, sid)[0], 'Misleading')
        tot, _, _ = score_all(rows, preds)
        v = (tot - base_S) / len(mis_rows) * 1e4
        price['ONE true Misleading caught'] = {'points_per_error_x1e4': v, 'n_available': len(mis_rows)}
        print(f"  {'ONE true Misleading caught':38s} {v:8.2f}   (n={len(mis_rows)} on val -- support-1 class!)")
    res['price_list'] = price

    json.dump(res, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
