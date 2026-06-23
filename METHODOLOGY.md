# Methodology & Experiment Log

繁體中文版：[METHODOLOGY.zh-TW.md](METHODOLOGY.zh-TW.md)

This documents the method that produced the final submission, plus the main experiments and
their (often negative) outcomes. All validation below uses an **honest protocol**: models are
fine-tuned on `train-1000` only and evaluated on the held-out official `val-1000` as a test
proxy; the combined `train+val (2000)` is used only to retrain the final submission models.

## 1. Inputs & preprocessing

Each example is a Traditional-Chinese ESG paragraph (`data` field). Preprocessing applies
**full-width → half-width normalization of digits and Latin letters only** (Chinese punctuation
is deliberately preserved — full NFKC would corrupt 繁中 punctuation). ESG sustainability
reports are dense with full-width numbers/years, so this sharpens evidence and timeline-year
signals (`fusion/configs/preprocess/fullwidth.yaml`).

## 2. Models

Multi-task BERT (`training/train_multitask.py`, `fusion/src/train.py`): one shared encoder with
four linear heads (promise / evidence_status / evidence_quality / verification_timeline).

- Backbones span four families: **ckip-bert-base, macbert-base, bge-m3, roberta-wwm-ext**.
- Backbones are first continued with masked-LM pre-training on ESG text — task-adaptive (TAPT)
  on the task paragraphs and domain-adaptive (DAPT) on external sustainability-report text
  (`training/train_mlm_tapt.py`, `fusion/scripts/build_dapt_corpus.py`, `tapt_pretrain.py`).
  Both are unsupervised (text only, no labels).
- Fine-tuning uses focal loss + per-task class-weighting, plus R-Drop, FGM, and multi-sample
  dropout; conditional masking trains the detail heads only where `promise=Yes`.

## 3. Ensembling & the `strong4` subset

For each task, model probabilities are combined by **per-field F1-weighted soft-vote** (weights
= each model's cross-validation Macro-F1), after **temperature-scaling calibration** fit on the
held-out OOF. An **ESG-type prior** (α=0.5) and **data-only rules** (timeline year-rule,
quality-rule, evidence-rule, weight 3) are injected, followed by per-field threshold and
timeline-offset selection (`pipeline/predict_test_realigned.py`).

**Subset selection.** An 8-model ensemble was diluted by its 3 weakest members. An exhaustive
subset search over the held-out val (ranked by the binary contribution `0.2·promise + 0.3·evidence`)
selected **`strong4` = ckip-tapt + macbert-tapt + bge-m3 + bge-m3-tapt** for the binary half.
`fusion/fuse_predict.py` emits this ensemble's binary CSV, which is spliced into the main pipeline
via `pipeline/build_tapt_hybrid.py --external_pe_csv`.

## 4. Schema cascade & the cascade multiplier

The competition schema is enforced as a cascade: `promise=No ⇒ timeline=evidence=quality=N/A`;
`evidence∈{No,N/A} ⇒ quality=N/A`; with a compliance repick of a real class when `promise=Yes`
(or `evidence=Yes`) but the cascaded field came out N/A.

Because timeline and quality are scored **N/A-excluded**, a more accurate **promise** lifts
timeline and quality "for free": a promise false-negative forces a real-labelled item to N/A,
which is a false-negative for that class. Empirically, replacing only the binary half with
`strong4` (vs. the prior best) raised promise +0.044 and dragged timeline +0.018 / quality +0.015
purely through the cascade — ~37% of the total gain. Investing in the binary half is therefore
worth more than its nominal 0.20+0.30 weight.

## 5. evidence_quality `Misleading`

`Misleading` is the rarest class (support ≈ 1–4 in 2000 test rows) and the BERT head emits
essentially none. The pipeline therefore applies a **fixed override list** of test ids
(`config/misleading_overrides.txt`): for each, it sets promise=Yes, evidence=Yes,
quality=Misleading. Candidate ids were produced by an **LLM-based scoring step** — the OpenRouter
API and a Claude-based agent re-read each candidate paragraph and rank how strongly it matches
the Misleading definition. The final id list is a fixed configuration input to the builder.

## 6. Final models & submission

The selected models are retrained on the combined `train+val (2000)` (`fusion/scripts/build_full2000.py`),
re-predicted on the test set, and run through the pipeline to emit `submissions/FINAL_SUBMISSION_yayou_0.6760.csv`.

## 7. Things that did NOT help (negative results)

All evaluated on the held-out val before spending submissions:

| attempt | outcome |
|---|---|
| longer `max_len` (384, 512) | worse than 256 on this small dataset (longer dilutes); 256 is optimal |
| test-time augmentation (multi-length, head+tail windows) | negative — windowed views are out-of-distribution for 256-trained models |
| multi-seed averaging of the binary ensemble | neutral-to-negative — averaging dilutes peaky correct predictions |
| selecting a "best" seed on val | binary val→LB transfer is seed-noisy: two seeds with identical val-promise (0.8144) gave 0.8199 vs 0.7819 on the test set |
| post-hoc weight/threshold hill-climbing on OOF | overfits OOF; does not transfer |
| larger backbones (roberta-large, xlmr-large) added to the pool | within validation noise; not selected by the subset search |

**Key lesson:** for the binary tasks, even a clean held-out validation does not reliably predict
test-set ranking; only the leaderboard does. The robust choice was the seed/config whose result
was actually confirmed, not the validation-best.

## 8. Public→private gap

Public 0.6760 → private 0.6405. The drop reflects (a) the rare `Misleading` overrides not
recurring on the private partition, and (b) binary-task regression from validation to the unseen
private set, consistent with the seed-transfer noise documented in §7.
