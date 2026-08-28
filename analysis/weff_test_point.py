"""Re-count every cascade parameter on the RELEASED TEST LABELS at the shipped submission's own
operating point, and re-evaluate the effective weight there. Backs Section 5.2's sentence

    "Recounted on the released test labels at the shipped submission's own operating point,
     all three reappear: 0.5267, 0.4374, 62.0% outside."

Why this exists: the paper's decision layer is fitted on validation (Section 4.4 says so), so a
skeptic can ask whether the headline W_eff = 0.5216 is an artifact of the operating point that
fitting chose. This script answers by moving to the one operating point fitting could not have
chosen: the shipped submission scored against the test gold released after the evaluation
closed. Every gate and head parameter is re-counted there (nothing is carried over from
validation), and the same central-difference derivative is taken.

    test-counted operating point: rho = 0.9655, phi = 0.3813   (val: 0.9828 / 0.4385)
    W_eff via recall       0.5267   (val 0.5216)
    W_eff via specificity  0.4374   (val 0.4268)
    share outside promise  62.0%    (val 61.7%)

The test gold is NOT redistributed here -- the organizers released it after the evaluation
closed; obtain it from them and pass --test_gold. The shipped submission file is in
../submissions/.

Run from inside analysis/:
    python weff_test_point.py --test_gold <path-to-released-test-gold.csv>
"""
import argparse
import csv
import io
import json

from binary_swap_experiment import gate_params, downstream_params, forward
from fit_cascade_from_val import FIELDS, W


def load_csv(path):
    return list(csv.DictReader(io.open(path, encoding="utf-8-sig")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_gold", required=True,
                    help="the organizers' released test gold (CSV with the four fields)")
    ap.add_argument("--pred", default="../submissions/FINAL_SUBMISSION_yayou_0.6760.csv")
    args = ap.parse_args()

    rows = load_csv(args.test_gold)
    pred = {r["id"]: r for r in load_csv(args.pred)}
    assert len(rows) == 2000, f"expected 2000 test rows, got {len(rows)}"

    gate = gate_params(rows, pred)
    down = downstream_params(rows, pred)
    print(f"test-counted operating point: rho={gate['rho']:.4f}  phi={gate['phi']:.4f}")

    def S_fields(rho=None, phi=None):
        g = json.loads(json.dumps(gate))
        if rho is not None:
            scale = rho / gate["rho"]
            g["rho"] = rho
            g["gamma_T"] = {k: (min(1.0, v * scale) if v is not None else None)
                            for k, v in gate["gamma_T"].items()}
            g["gamma_Q"] = {k: (min(1.0, v * scale) if v is not None else None)
                            for k, v in gate["gamma_Q"].items()}
            g["gamma_E"] = {k: (min(1.0, v * scale) if v is not None else None) if k != "N/A" else v
                            for k, v in gate["gamma_E"].items()}
        if phi is not None:
            g["phi"] = phi
            g["gamma_E"] = dict(g["gamma_E"], **{"N/A": phi})
        return forward(rows, g, down, gamma_mode="class")

    d = 1e-3
    s_hi, f_hi = S_fields(rho=min(1.0, gate["rho"] + d))
    s_lo, f_lo = S_fields(rho=gate["rho"] - d)
    dS = (s_hi - s_lo) / (2 * d)
    dMP = (f_hi["promise_status"] - f_lo["promise_status"]) / (2 * d)
    print(f"dS/drho={dS:+.4f}  dM_P/drho={dMP:+.4f}  W_eff(recall)={dS / dMP:.4f}")

    comp = {f: W[f] * (f_hi[f] - f_lo[f]) / (2 * d) for f in FIELDS}
    tot = sum(comp.values())
    short = {"promise_status": "promise", "evidence_status": "evidence",
             "evidence_quality": "quality", "verification_timeline": "timeline"}
    print("destination split: " + "  ".join(f"{short[f]}={v:+.4f}"
                                            for f, v in comp.items()))
    print(f"share outside promise: {100 * (1 - comp['promise_status'] / tot):.1f}%")

    sp_hi, fp_hi = S_fields(phi=max(0.0, gate["phi"] - d))
    sp_lo, fp_lo = S_fields(phi=gate["phi"] + d)
    dSp = (sp_hi - sp_lo) / (2 * d)
    dMPp = (fp_hi["promise_status"] - fp_lo["promise_status"]) / (2 * d)
    print(f"W_eff(specificity)={dSp / dMPp:.4f}")


if __name__ == "__main__":
    main()
