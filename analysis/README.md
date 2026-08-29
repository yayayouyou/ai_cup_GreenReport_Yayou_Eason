# Analysis — every number in the paper, and where it comes from

This directory holds the analysis behind *What a Field Is Really Worth: Effective Weights under a
Gated, Rare-Class Composite Metric* (TEAM_10049, NTCIR-19 AI CUP VeriPromise ESG Track).

The rest of this repository is the **system**: how the models were trained and how the submission
was produced (see [`REPRODUCE.md`](../REPRODUCE.md)). This directory is the **measurement**: the
scripts that turn one system's error rates into scores, and the inputs they read.

Everything here is CPU-only and runs in seconds. No model checkpoints are needed — the
predictions are already here.

---

## Layout

```
analysis/
  *.py                     the scripts
  out/                     their outputs, as the paper's tables read them
  io/
    val_strong4.json       the deployed system on val-1000 (the anchor)
    binary_sources/        val predictions from the 42 independently trained binary sources
    downstream/            val + test predictions from the 14 multi-task checkpoints
```

One system is analysed throughout: **strong4** supplies promise and evidence, and the downstream
fields come from a calibrated ensemble of the 14 checkpoints. `io/val_strong4.json` is that
system's validation output, and every parameter in the paper is counted on it.

---

## Which script produces which number

| Paper | Number | Script | Output |
|---|---|---|---|
| Table 2 (top) | val `0.6095`, per-field `.8148 .7083 .3855 .6607` | `oracle_and_weights.py` | `out/oracle_weights.json` → `B deployed` |
| Table 2, intervals | iid half-width `0.0222`, cluster `0.0230`, ICC ≤ `0.034` | `bootstrap_apparatus.py` | `out/bootstrap.json` → `ci`, `icc` |
| §5.2 RQ1 | `dS/dρ = 0.6078`, `dM_P/dρ = 1.1652`, **W_eff = 0.5216** | `oracle_and_weights.py` | `out/oracle_weights.json` → `effective_weight.via_recall` |
| §5.2 RQ1 | specificity direction `0.4268` | same | `effective_weight.via_specificity` |
| §5.2 RQ1 | `61.7%` collected outside promise | same | `effective_weight.share_outside` |
| §5.2 | decomposition vs finite move, `51.9%` | `oracle_and_weights.py` | `O_P gold promise only` − `B`, weighted per field |
| §5.3 RQ2 | swap over the 42 sources, `U` span `0.0092`, MAE `0.0020 / 0.0014 / 0.0023` | `binary_swap_experiment.py` | `out/binary_swap.json` |
| §5.3 | `γ_Q = 0.832` vs `0.959`, `γ_T` from `0.974` | `fit_cascade_from_val.py` | `out/cascade_params.json` → `params` |
| §5.4 RQ3a | headroom `+0.1565 = 0.1246 + 0.0319`, promise alone `0.0770` | `oracle_and_weights.py` | `out/oracle_weights.json` → `headroom`, `O_P` |
| §5.4 | ceilings `0.6905` / `0.4640` | same | `O_PE gold promise + evidence` |
| Table 3 (bottom) | per-repair prices `63.9 / 7.0 / 8.5 / 6.6 / 8.8 / 1.7` and their `n` | same | `price_list` |
| §5.5 RQ3b | `0.1167 = 0.35/3`; `138` and `133` repairs | arithmetic on `price_list` | — |
| §7.1 | ladder `0.6360 / 0.6302 / 0.6355 / 0.6095`, steps `−0.0058 / +0.0053 / −0.0260` | `validation_ladder.py` | `out/validation_ladder.json` |
| §7.1 | the convention's price, `−0.0265` | `metric_design_ablation.py` | `out/metric_design.json` |
| §7.1 | seed SD `0.0055`, roster gap `0.0013` | `bootstrap_apparatus.py` | `out/bootstrap.json` |
| §3.3 | bootstrap misses a support-1 class, `(1−1/N)^N` | `bootstrap_apparatus.py` | `out/bootstrap.json` → `degenerate` |
| §3.3 | ML-Promise `42` of `2,110` (`2.0%`), `30` of them promise=No with real quality, `1` *Misleading* | `metric_design_ablation.py` | `out/metric_design.json` → `mlpromise_transferable` |
| §4.4 | baseline-script convention gap on identical predictions, `+0.0059` | `scorer_conventions.py` | printed |
| §5.2 | W_eff bootstrap CI `[0.485, 0.542]` / `[0.403, 0.450]` (every rate recounted per replicate, B=400) and step sensitivity | `weff_ci.py` | printed |
| §7.1 | leaderboard pair: `0` of `3,000` random splits under the schema set, within `0.001` under the present-labels default | `lb_split_search.py` (needs the released test gold, not redistributed) | printed |
| §5.2 | test-point recount: `0.5267` / `0.4374`, `62.0%` outside (at test-counted `rho 0.9655` / `phi 0.3813`) | `weff_test_point.py` (needs the released test gold, not redistributed) | printed |
| §4.3 | the compliance repick | `predict_repick.py` | — |
| §4.1–4.2 | the cascade model itself (v1 / v2, forward simulation) | `cascade_model.py` | — |

`check_claims.py` recomputes every **relational** claim the paper makes about these outputs --
channel orderings, what transfers within `0.05`, what a price is, which field carries `0.1167` --
and prints each next to the statement it licenses. Run it before quoting any of these numbers in
prose. [`PAPER_CLAIMS.md`](PAPER_CLAIMS.md) explains why: across five review passes almost every
defect was a wrong *sentence* about a right *number*, and a field lookup cannot catch those.

`make_figures.py` draws the predicted-vs-observed scatter. The figure was cut from the final
paper; the script is kept because the numbers behind it are in `out/binary_swap.json`.

---

## Running it

Run these from inside `analysis/`. Python 3.10+, `numpy` and `scikit-learn` are all that is
needed; `make_figures.py` also wants `matplotlib`. **Every flag below is now the default**, so each
script also runs bare — `python oracle_and_weights.py` — and rewrites its own file under `out/`.

```bash
python oracle_and_weights.py \
    --val ../data_set/vpesg4k_val_1000.json \
    --pipeline_val io/val_strong4.json \
    --out out/oracle_weights.json

python binary_swap_experiment.py \
    --val ../data_set/vpesg4k_val_1000.json \
    --pipeline_val io/val_strong4.json \
    --sources io/binary_sources \
    --out out/binary_swap.json

python bootstrap_apparatus.py --B 10000 \
    --val ../data_set/vpesg4k_val_1000.json \
    --pipeline_val io/val_strong4.json \
    --out out/bootstrap.json
```

```bash
python validation_ladder.py     --val ../data_set/vpesg4k_val_1000.json     --pipeline_val io/val_strong4.json     --out out/validation_ladder.json

python metric_design_ablation.py     --val ../data_set/vpesg4k_val_1000.json     --pipeline_val io/val_strong4.json     --sources io/binary_sources     --out out/metric_design.json
```

Re-running all four rewrites `out/` with values identical to what is committed here.

**One external corpus is not ours to redistribute.** The cross-corpus check in
`metric_design_ablation.py` reads ML-Promise (SemEval-2025 Task 6) as
`Trainset_<Language>.json` under `--ext_dir`. Without it the script still runs and still produces
everything else; it prints that it is skipping that section. The counts it would produce are
already recorded in `out/metric_design.json` under `schema` and `mlpromise_transferable`.

Note that the paper reports **42** violations and the `rule1_violations` column sums to **45**. Both
are printed. ML-Promise scores evidence Yes/No with no N/A class, so R1's evidence clause cannot
transfer to it; restricting R1 to quality and timeline — the fields both schemas share — gives 38,
plus 4 R2 violations, which is the 42 the paper uses.

---

## Three things to know before reading the outputs

**The scorer.** Every number uses the official convention: macro-F1 per field over a **fixed**
class set, weighted `0.20 / 0.30 / 0.35 / 0.15`. N/A is a scored class in *evidence* and is
excluded from the averages of *quality* and *timeline*. Calling a library default instead moves
the composite by `−0.0265` on validation and `+0.0242` on test — that is §7.1's whole subject, so
it is not a detail. The organizers' released baseline script scores yet another convention — N/A
averaged into quality and timeline, plus a timeline label (`longer_than_5_years`) that no released
data file uses — worth `+0.0059` on identical predictions; `scorer_conventions.py` prints the
decomposition.

**What depends on what.** The paper's claims sit in three tiers. The two lemmas (§3.3) use no
predictions at all — only Eq. (1), the schema rules and the label counts. The effective weights
and price list use counted error rates with zero fitted parameters; re-counting every rate on the
released test labels at the shipped submission's own operating point reproduces the headline
(`weff_test_point.py`: `0.5267 / 0.4374`, `62.0%` outside, vs validation's
`0.5216 / 0.4268 / 61.7%`). Only the deployed score `0.6095` depends on the decision layer that
§4.4 discloses was fitted on validation, and §7.1 audits what that cost.

**What a "price" is.** The `price_list` rows are **not** derivatives. Each is the mean total-score
gain from flipping *all* `n` offending rows to gold and dividing by `n` — a finite repair, so a row
times its own `n` is exact, while sums *across* rows are first-order only, because the repairs
interact. The two promise rows also carry the gold **evidence** onto the repaired row, so
`promise FN` prices a promise-plus-evidence repair: with the model's own evidence kept instead it
is `6.30e-4` rather than `8.46e-4`. The paper's Table 3 caption says so.

**One defect of ours is in here.** The three data-only rules guard on `promise_status`, a field
the validation file carries and the test file does not, so the same code was a different function
on the two splits. `io/val_strong4.json` is post-correction: the guard is blinded, and every
validation number in the paper is the lower, corrected one. Section 4.4 of the paper describes it.

---

## License

Same as the repository root: see [`LICENSE`](../LICENSE) and [`NOTICE`](../NOTICE).
The official VeriPromiseESG4K data under `../data_set/` belongs to the AI CUP organizers and is
redistributed here under their terms.
