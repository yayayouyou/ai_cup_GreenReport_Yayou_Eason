"""Paired bootstrap for system DIFFERENCES on val-1000. Backs Table 2's bottom block
(diluted7 0.6053 / deployed 0.6095 / 14-ensemble 0.6108) and Section 8's "a paired bootstrap
of the difference spans [-.0116, +.0143]".

Why paired: comparing a 0.0013 gap to a SINGLE system's CI half-width (0.0222) answers the
wrong question -- two systems scored on the same 1,000 items have highly correlated errors, so
the difference's own resampling distribution is the right yardstick. Each replicate draws one
item sample and scores BOTH systems on it; the delta cancels shared noise.

Result (seed 20260830, B=2000): the paired CI still straddles zero decisively
(P(delta<0)=0.41), so the paper's claim -- validation cannot resolve the 14-vs-deployed
choice -- survives its own strictest test. The strong4-vs-diluted7 gap (+0.0042) straddles
zero too ([-.0025, +.0107]): the roster choice itself sits below paired resolution, which is
Section 8's point about deployed decisions and variance. Same-config seed twins run last, at
B=400: every twin's CI straddles zero as well.

Run bare from inside analysis/:  python paired_delta.py
Writes out/paired_delta.json.
"""
import json
import random

from binary_swap_experiment import load_sources, splice, vote
from bootstrap_apparatus import score_rows
from fit_cascade_from_val import FIELDS, W, SCORED, f1_from_counts

VAL = "../data_set/vpesg4k_val_1000.json"
PIPE = "io/val_strong4.json"
B_MAIN = 2000
B_SEED = 400
SEED = 20260830


def compile_marks(rows, pred_by):
    """Per-item (gold==c, pred==c) marks for every scored (field, class): O(1) lookups later."""
    marks = []
    for f in FIELDS:
        for c in SCORED[f]:
            g = [1 if r[f] == c else 0 for r in rows]
            p = [1 if pred_by[str(r["id"])][f] == c else 0 for r in rows]
            marks.append((f, [gi & pi for gi, pi in zip(g, p)], g, p))
    return marks


def fast_score(marks, idx):
    per_field = {f: 0.0 for f in FIELDS}
    for f, tp_a, g_a, p_a in marks:
        tp = sum(tp_a[i] for i in idx)
        gold = sum(g_a[i] for i in idx)
        pred = sum(p_a[i] for i in idx)
        per_field[f] += f1_from_counts(tp, gold - tp, pred - tp)
    return sum(W[f] * per_field[f] / len(SCORED[f]) for f in FIELDS)


def paired(rows, marks_a, marks_b, b, rng):
    n = len(rows)
    ds = []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        ds.append(fast_score(marks_a, idx) - fast_score(marks_b, idx))
    ds.sort()
    return {"mean": sum(ds) / b, "lo": ds[int(0.025 * b)], "hi": ds[int(0.975 * b) - 1],
            "p_neg": sum(1 for d in ds if d < 0) / b}


def main():
    rows = json.load(open(VAL, encoding="utf-8"))
    pipeline_by = {str(x["id"]): x for x in json.load(open(PIPE, encoding="utf-8"))}
    rows = [r for r in rows if str(r["id"]) in pipeline_by]
    sources = load_sources("io/binary_sources")
    s4 = ["ckip_tapt_ep3", "macbert_tapt", "bgem3", "bgem3_tapt"]
    d7 = s4 + ["b0_ckip", "macbert_base", "roberta_wwm"]
    raw14 = {sid: {"promise_status": p["_raw_promise_status"],
                   "evidence_status": p["_raw_evidence_status"]}
             for sid, p in pipeline_by.items()}

    systems = {
        "deployed(strong4)": splice(rows, vote(sources, s4), pipeline_by),
        "diluted7(vote)": splice(rows, vote(sources, d7), pipeline_by),
        "14-ensemble(raw binary)": splice(rows, raw14, pipeline_by),
    }

    out = {"points": {}, "paired": {}, "seed_twins": {}}
    marks = {}
    for name, p in systems.items():
        s, _ = score_rows(rows, p)
        out["points"][name] = round(s, 4)
        marks[name] = compile_marks(rows, p)
        idx0 = list(range(len(rows)))
        assert abs(fast_score(marks[name], idx0) - s) < 1e-9
        print(f"point  {name:26s} S={s:.4f}")

    rng = random.Random(SEED)
    for a, b in (("14-ensemble(raw binary)", "deployed(strong4)"),
                 ("deployed(strong4)", "diluted7(vote)")):
        r = paired(rows, marks[a], marks[b], B_MAIN, rng)
        out["paired"][f"{a} - {b}"] = {k: round(v, 4) for k, v in r.items()}
        verdict = "EXCLUDES 0" if r["lo"] > 0 or r["hi"] < 0 else "straddles 0"
        print(f"\n{a} - {b}:")
        print(f"  mean {r['mean']:+.4f}  95% CI [{r['lo']:+.4f}, {r['hi']:+.4f}]  "
              f"P(d<0)={r['p_neg']:.3f}  -> {verdict}")

    print("\nsame-config seed twins (paired, B=400):")
    twins = [("ckip_tapt_ep3", "ckip_tapt_ep3_s99"), ("macbert_tapt", "macbert_tapt_s777"),
             ("bgem3_tapt", "bgem3_tapt_s777"), ("bgem3", "bgem3_s99")]
    for a, b in twins:
        if a not in sources or b not in sources:
            print(f"  {a} vs {b}: source missing, skipped")
            continue
        ma = compile_marks(rows, splice(rows, sources[a], pipeline_by))
        mb = compile_marks(rows, splice(rows, sources[b], pipeline_by))
        r = paired(rows, ma, mb, B_SEED, rng)
        out["seed_twins"][f"{a} - {b}"] = {k: round(v, 4) for k, v in r.items()}
        verdict = "EXCLUDES 0" if r["lo"] > 0 or r["hi"] < 0 else "straddles 0"
        print(f"  {a:14s} - {b:20s} mean {r['mean']:+.4f}  "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  -> {verdict}")

    json.dump(out, open("out/paired_delta.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote out/paired_delta.json")


if __name__ == "__main__":
    main()
