# -*- coding: utf-8 -*-
"""Uncertainty and step-sensitivity for the effective weight. Backs Section 5.2's
"(bootstrap 95% CI 0.485--0.542)" and "(0.403--0.450)".

Q2: what is the sampling uncertainty of W_eff 0.5216 / 0.4268 and the 61.7% share?
    -> nonparametric bootstrap over the 1,000 validation items: resample rows, RE-COUNT every
       cascade parameter on the replicate, re-differentiate. Percentile CI.
Q3: is the central-difference step 1e-3 doing anything?
    -> W_eff at steps 1e-4 / 1e-3 / 1e-2 on the full validation count.
"""
import io
import json
import random
import sys

from binary_swap_experiment import gate_params, downstream_params, forward

VAL = "../data_set/vpesg4k_val_1000.json"
PRED = "io/val_strong4.json"

rows_all = json.load(io.open(VAL, encoding="utf-8"))
pred = json.load(io.open(PRED, encoding="utf-8"))
by = {str(p["id"]): p for p in (pred if isinstance(pred, list) else pred["predictions"])}

FIELDS = ["promise_status", "evidence_status", "evidence_quality", "verification_timeline"]
W = {"promise_status": .20, "evidence_status": .30, "evidence_quality": .35,
     "verification_timeline": .15}


def weff(rows, d=1e-3):
    gate = gate_params(rows, by)
    down = downstream_params(rows, by)

    def S_fields(rho=None, phi=None):
        g = json.loads(json.dumps(gate))
        if rho is not None:
            sc = rho / gate["rho"] if gate["rho"] else 1.0
            g["rho"] = rho
            for key in ("gamma_T", "gamma_Q"):
                g[key] = {k: (min(1.0, v * sc) if v is not None else None)
                          for k, v in gate[key].items()}
            g["gamma_E"] = {k: (min(1.0, v * sc) if v is not None else None) if k != "N/A" else v
                            for k, v in gate["gamma_E"].items()}
        if phi is not None:
            g["phi"] = phi
            g["gamma_E"] = dict(g["gamma_E"], **{"N/A": phi})
        return forward(rows, g, down, gamma_mode="class")

    s_hi, f_hi = S_fields(rho=min(1.0, gate["rho"] + d))
    s_lo, f_lo = S_fields(rho=gate["rho"] - d)
    dS = (s_hi - s_lo) / (2 * d)
    dMP = (f_hi["promise_status"] - f_lo["promise_status"]) / (2 * d)
    comp = {f: W[f] * (f_hi[f] - f_lo[f]) / (2 * d) for f in FIELDS}
    share = 1 - comp["promise_status"] / sum(comp.values())

    sp_hi, fp_hi = S_fields(phi=max(0.0, gate["phi"] - d))
    sp_lo, fp_lo = S_fields(phi=gate["phi"] + d)
    dSp = (sp_hi - sp_lo) / (2 * d)
    dMPp = (fp_hi["promise_status"] - fp_lo["promise_status"]) / (2 * d)
    return dS / dMP, dSp / dMPp, share


print("=== Q3: step sensitivity on the full validation count ===")
for d in (1e-4, 1e-3, 1e-2):
    r, s, sh = weff(rows_all, d)
    print(f"  step {d:g}:  recall {r:.4f}   specificity {s:.4f}   share outside {100*sh:.1f}%")

print("\n=== company halves: W_eff recounted on two disjoint company splits ===")
comps = sorted({r.get("company", "?") for r in rows_all})
for name, keep in (("companies A (odd)", set(comps[0::2])),
                   ("companies B (even)", set(comps[1::2]))):
    rr = [r for r in rows_all if r.get("company") in keep]
    r_, s_, sh_ = weff(rr)
    print(f"  {name:18s} n={len(rr):4d}  recall {r_:.4f}  "
          f"specificity {s_:.4f}  share {100 * sh_:.1f}%")

print("\n=== Q2: bootstrap over the 1,000 validation items (recount everything) ===")
B = int(sys.argv[1]) if len(sys.argv) > 1 else 400
rng = random.Random(20260829)
rec, spec, shares = [], [], []
fail = 0
for b in range(B):
    sample = [rows_all[rng.randrange(len(rows_all))] for _ in range(len(rows_all))]
    try:
        r, s, sh = weff(sample)
        rec.append(r); spec.append(s); shares.append(sh)
    except Exception:
        fail += 1


def pct(a, q):
    a = sorted(a)
    i = q * (len(a) - 1)
    lo, hi = int(i), min(int(i) + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (i - lo)


for name, a in (("W_eff recall", rec), ("W_eff specificity", spec),
                ("share outside", shares)):
    print(f"  {name:18s} mean {sum(a)/len(a):.4f}   "
          f"95% CI [{pct(a, .025):.4f}, {pct(a, .975):.4f}]   "
          f"(B={len(a)}, failed {fail})")
