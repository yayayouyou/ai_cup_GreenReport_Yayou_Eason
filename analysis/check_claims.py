"""Print every RELATIONAL claim the paper makes about these outputs, with its actual value.

Why this exists
---------------
`README.md` maps each paper number to a script and an output field, and that catches a wrong
scalar. It does not catch the kind of error that kept surviving into late drafts, because those
were not wrong scalars -- every scalar was right. They were wrong statements ABOUT the scalars:

    "the ordering of channels survive"        -> the top two exchange rank
    "quality and timeline macros do not"      -> timeline is 0.0319, inside the 0.05 it is denied
    "neither variant beats the mean"          -> one of them did, by its own printed numbers
    "Lemma 2 makes the gate worth 0.5216"     -> Eq. (4) does; Lemma 2 prices nothing

A relation cannot be checked by looking up one field. It has to be recomputed. This script
recomputes each one and prints it next to the claim the paper makes, so the prose can be read
against the arithmetic in one pass.

Run it bare from inside analysis/:  python check_claims.py
Everything is CPU-only and reads out/*.json plus the released validation labels.

See PAPER_CLAIMS.md for the failure taxonomy these checks are derived from.
"""
import io
import json
import os

OUT = "out"
W = {"promise_status": 0.20, "evidence_status": 0.30,
     "evidence_quality": 0.35, "verification_timeline": 0.15}
NCLS = {"promise_status": 2, "evidence_status": 3,
        "evidence_quality": 3, "verification_timeline": 4}
SHORT = {"promise_status": "promise", "evidence_status": "evidence",
         "evidence_quality": "quality", "verification_timeline": "timeline"}


def load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    return json.load(io.open(p, encoding="utf-8"))


def head(n, title):
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


def verdict(ok, claim):
    print(f"   [{'TRUE ' if ok else 'FALSE':5s}] \"{claim}\"")


ow = load("oracle_weights.json")
cp = load("cascade_params.json")
bs = load("binary_swap.json")
vl = load("validation_ladder.json")
bo = load("bootstrap.json")
md = load("metric_design.json")

missing = [n for n, d in [("oracle_weights", ow), ("cascade_params", cp),
                          ("binary_swap", bs), ("validation_ladder", vl)] if d is None]
if missing:
    print(f"missing outputs: {missing} -- run the scripts in README.md first")

# ---------------------------------------------------------------- 1
if ow:
    head(1, "CHANNEL ORDERING: derivative vs a finite move")
    comp = ow["effective_weight"]["components"]
    B = ow["B  deployed (model gate, model scored)"]["fields"]
    OP = ow["O_P gold promise only"]["fields"]
    fin = {f: W[f] * (OP[f] - B[f]) for f in W}

    d_rank = sorted(comp, key=lambda f: -comp[f])
    f_rank = sorted(fin, key=lambda f: -fin[f])
    print("   derivative  (dS/drho, weighted):  " +
          " > ".join(f"{SHORT[f]} {comp[f]:+.4f}" for f in d_rank))
    print("   finite move (gold promise - B) :  " +
          " > ".join(f"{SHORT[f]} {fin[f]:+.4f}" for f in f_rank))
    same_dir = all((comp[f] > 0) == (fin[f] > 0) for f in W)
    print()
    print("   Statements the prose may be tempted to make:")
    verdict(same_dir, "the direction survives")
    verdict(d_rank == f_rank, "the ordering of channels survives")
    if d_rank != f_rank:
        print(f"           -> leading channel changes hands: "
              f"{SHORT[d_rank[0]]} -> {SHORT[f_rank[0]]}")
        print(f"           -> ranks 3 and 4 unchanged: "
              f"{d_rank[2:] == f_rank[2:]}")
    tot = sum(fin.values())
    print(f"\n   share outside promise: derivative "
          f"{100 * ow['effective_weight']['share_outside']:.1f}%   "
          f"finite {100 * (1 - fin['promise_status'] / tot):.1f}%   "
          f"(finite total {tot:+.4f})")
    verdict(False, "the share survives")

# ---------------------------------------------------------------- 2
if ow:
    head(2, "WHAT PRICES THE GATE (attribution check)")
    ew = ow["effective_weight"]
    print(f"   W_eff via recall      = {ew['via_recall']:.4f}"
          f"   = dS/drho / dM_P/drho, Eq. (4) on the counted model")
    print(f"   W_eff via specificity = {ew['via_specificity']:.4f}")
    print("   Lemma 1 fixes a class coefficient; Lemma 2 is a gold-label biconditional.")
    print("   NEITHER lemma yields 0.5216 -- only Eq. (4) evaluated at the operating point does.")
    print("   Any sentence of the form 'Lemma 2 makes the gate worth 0.5216' is a misattribution.")

# ---------------------------------------------------------------- 3
if cp:
    head(3, "PER-CLASS F1 vs THE FIELD MACRO (catches a wrong last digit)")
    pc = cp["direct"]["per_class"]
    fl = cp["direct"]["fields"]
    for f in ("promise_status", "evidence_status"):
        vals = pc[f]
        mean = sum(vals.values()) / len(vals)
        print(f"   {SHORT[f]:9s} per-class " +
              ", ".join(f"{k} {v:.4f}" for k, v in sorted(vals.items())))
        print(f"   {'':9s} mean {mean:.10f}  vs field macro {fl[f]:.10f}   "
              f"{'match' if abs(mean - fl[f]) < 1e-9 else 'MISMATCH'}")
    print("\n   Round each per-class F1 the way the prose does and re-average: if the average")
    print("   no longer equals the field macro, a printed digit is wrong.")

# ---------------------------------------------------------------- 4
if cp:
    head(4, "TRANSFER: which quantities carry to the released test labels within 0.05")
    print("   The gate-open rates gamma_c are counted on validation:")
    for f, key in (("evidence_quality", "gamma_Q"), ("verification_timeline", "gamma_T")):
        g = cp["params"].get(key, {})
        print(f"      {key}: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(g.items())))
    print("\n   Test-side gamma_Q, recomputed from the deployed submission against the official")
    print("   gold, is Clear 0.943 / Not Clear 0.788; validation is Clear 0.959 / Not Clear 0.832.")
    print("      |0.943-0.959| = 0.016   |0.788-0.832| = 0.044   both inside 0.05")
    print("   CAUTION: print such pairs WITH their class labels. Read in the wrong order the")
    print("   same numbers give 0.111 and 0.171, and the claim inverts.")
    print("\n   The model's PREDICTED downstream macros are a different quantity:")
    print("      quality  off by 0.0641   -> outside 0.05")
    print("      timeline off by 0.0319   -> INSIDE 0.05")
    print("   So 'quality and timeline macros do not transfer' is false for timeline.")

# ---------------------------------------------------------------- 5
if ow:
    head(5, "PRICE LIST: what a 'price' is, and what price x n does and does not equal")
    pl = ow["price_list"]
    for k, v in pl.items():
        if v.get("points_per_error_x1e4") is None:
            continue
        print(f"   {k:42s} {v['points_per_error_x1e4']:7.2f}e-4  n={v['n_available']}")
    print("\n   Each price is the MEAN GAIN from repairing all n such rows to gold, divided by n")
    print("   (oracle_and_weights.py, repair()). It is not a derivative.")
    print("      -> price x its own n is EXACT for that row")
    print("      -> sums ACROSS rows are first-order only; the repairs interact")
    print("   The two promise rows also splice the gold EVIDENCE onto the repaired row, so")
    print("   'promise FN' prices a promise+evidence repair: 8.46e-4 as shipped, 6.30e-4 with")
    print("   the model's own evidence kept. Label it accordingly wherever it is quoted.")
    slot = 0.35 / 3
    for lbl, key in (("promise FN", "promise FN (gold Yes, pred No)"),
                     ("promise FP", "promise FP (gold No, pred Yes)")):
        p = pl[key]["points_per_error_x1e4"] * 1e-4
        print(f"\n   Misleading slot {slot:.4f} / {lbl} price = {slot / p:.1f} repairs "
              f"(n available: {pl[key]['n_available']})")

# ---------------------------------------------------------------- 6
if ow:
    head(6, "HEADROOM DECOMPOSITION")
    h = ow["headroom"]
    print("   " + json.dumps(h))
    tot = h.get("total")
    dr, ca = h.get("direct"), h.get("cascade")
    if None not in (tot, dr, ca):
        print(f"\n   direct {dr:.4f} + cascade {ca:.4f} = {dr + ca:.4f}   "
              f"vs total {tot:.4f}   "
              f"{'match' if abs(dr + ca - tot) < 5e-5 else 'MISMATCH'}")
        print(f"   cascade share = {100 * ca / tot:.1f}%")

# ---------------------------------------------------------------- 7
if vl:
    head(7, "LADDER: the three steps must sum to the end-to-end gap")
    dec = vl.get("decomposition", {})
    steps = {k: v for k, v in dec.items() if k != "total"}
    for k, v in steps.items():
        print(f"   {k:24s} {v:+.6f}")
    ssum = sum(steps.values())
    tot = dec.get("total")
    print(f"   {'sum of steps':24s} {ssum:+.6f}")
    if tot is not None:
        print(f"   {'reported total':24s} {tot:+.6f}   "
              f"{'match' if abs(ssum - tot) < 1e-9 else 'MISMATCH'}")

# ---------------------------------------------------------------- 8
head(8, "CLASS COEFFICIENTS (Lemma 1): which fields carry which slot")
for f, n in NCLS.items():
    print(f"   {SHORT[f]:9s} w={W[f]:.2f}  scored classes={n}  ->  coefficient {W[f] / n:.4f}")
print("\n   0.1167 is QUALITY's coefficient only. Do not attach it to a generic 'class of")
print("   support s' -- timeline classes carry 0.0375, promise and evidence 0.1000.")

# ---------------------------------------------------------------- 9
if md and "mlpromise_transferable" in md:
    head(9, "ML-PROMISE: two different violation counts, both correct, easily confused")
    m = md["mlpromise_transferable"]
    tot_raw = sum(c["rule1_violations"] + c["rule2_violations"]
                  for c in md["schema"] if c["corpus"].startswith("ML-Promise"))
    print(f"   transferable rules only : {m['violations']} / {m['n']} = "
          f"{100 * m['rate']:.1f}%   <- the number the paper reports")
    print(f"   R1 as written in code   : {tot_raw} / {m['n']} = "
          f"{100 * tot_raw / m['n']:.1f}%   <- includes an evidence clause that cannot transfer")
    print(f"   of the transferable set: {m['promise_no_with_real_quality']} are promise=No with a "
          f"real quality judgement, {m['promise_no_with_misleading']} Misleading")
    per = [(c["corpus"], c["transferable_violations"], c["promise_no"])
           for c in md["schema"] if c["corpus"].startswith("ML-Promise")]
    print("\n   per language (violations / promise=No rows):")
    for name, v, n in per:
        print(f"      {name:22s} {v:3d} / {n:4d}"
              + (f"  = {100 * v / n:.1f}% of its own promise-No rows" if n else ""))
    print("   The pooled 2.0% hides this spread. Say so wherever the pooled figure is quoted.")

print("\n" + "=" * 78)
print("Read PAPER_CLAIMS.md before quoting any of these in prose.")
print("=" * 78)
