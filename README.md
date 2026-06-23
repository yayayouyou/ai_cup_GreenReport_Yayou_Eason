# VeriPromiseESG 2026 — Traditional-Chinese ESG Promise Verification

繁體中文版：[README.zh-TW.md](README.zh-TW.md)

Final submission code for the AI CUP 2026 / VeriPromiseESG task.
Authors: **yayou**, **Eason** (team).

The task: for each Traditional-Chinese ESG paragraph (Taiwan 2024 sustainability reports),
predict four fields, scored by weighted Macro-F1:

```
total = 0.20·F1(promise_status)        # Yes / No
      + 0.30·F1(evidence_status)       # Yes / No / N/A
      + 0.35·F1(evidence_quality)      # Clear / Not Clear / Misleading / N/A   (N/A-excluded)
      + 0.15·F1(verification_timeline) # already / within_2_years /
                                       #   between_2_and_5_years / more_than_5_years / N/A  (N/A-excluded)
```

## Result

| | weighted score |
|---|---|
| public leaderboard | **0.6760** |
| **private (final)** | **0.6405** |

Final submitted file: [`submissions/FINAL_SUBMISSION_yayou_0.6760.csv`](submissions/FINAL_SUBMISSION_yayou_0.6760.csv)

## Method (overview)

A multi-task BERT ensemble produces per-field class probabilities, which are combined and
post-processed into the final labels:

```
paragraph text
   │  full-width → half-width digit/letter normalization (Chinese punctuation preserved)
   ▼
multi-task BERT ensemble  (4 backbone families: ckip-bert / macbert / bge-m3 / roberta-wwm)
   │   • domain-adaptive + task-adaptive MLM pre-training of the backbones (DAPT/TAPT)
   │   • each backbone fine-tuned with focal loss + class-weighting (multi-task heads)
   │   • subset of the strongest models selected on a held-out validation split
   ▼
per-field F1-weighted soft-vote  → temperature calibration  → ESG-type prior
   ▼
rule injection (timeline year-rule / quality-rule / evidence-rule)  → threshold selection
   ▼
schema cascade (promise=No ⇒ rest = N/A; evidence∈{No,N/A} ⇒ quality = N/A) + compliance repick
   ▼
fixed Misleading override list (config/misleading_overrides.txt)
   ▼
submission.csv
```

The binary half (promise / evidence) is produced by `fusion/` (full-width-normalized BERT
ensemble, the `strong4` subset); the timeline / quality half and all post-processing live in
`pipeline/`. They are joined by `pipeline/build_tapt_hybrid.py --external_pe_csv`.

## Repository layout

```
pipeline/      our post-processing pipeline that emits the final CSV
  build_tapt_hybrid.py        final-CSV builder (calibration + F1-weights + rules + cascade +
                              Misleading overrides; --external_pe_csv splices the fused binary)
  predict_test_realigned.py   core scoring/realignment pipeline (calibration, rules, priors,
                              threshold + timeline offset selection)
  metrics.py                  weighted Macro-F1 scorer
  rules/esg_type_priors.json  ESG-type class priors
training/      BERT training utilities
  train_multitask.py          multi-task BERT fine-tuning (focal loss, class-weighting,
                              R-Drop, FGM, multi-sample dropout)
  train_mlm_tapt.py           domain/task-adaptive MLM pre-training (DAPT/TAPT)
  predict_probs.py            K-fold OOF + test probability prediction
fusion/        full-width-normalized BERT ensemble (binary half); team-shared training code
  fuse_predict.py             ensembles the selected models -> binary CSV
  src/, scripts/, configs/    multi-task BERT training + per-field soft-vote ensemble
config/
  misleading_overrides.txt    the fixed Misleading override id list (configuration input)
submissions/
  FINAL_SUBMISSION_yayou_0.6760.csv
METHODOLOGY.md   detailed method + design decisions
REPRODUCE.md     step-by-step reproduction
```

## Techniques used (references)

All techniques are standard published methods, reimplemented here:

- **TAPT / DAPT** — task/domain-adaptive masked-LM pre-training: Gururangan et al., *Don't Stop
  Pretraining*, ACL 2020.
- **Focal loss** for class imbalance: Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017.
- **R-Drop** regularization: Liang et al., NeurIPS 2021.
- **FGM** adversarial training: Miyato et al., *Adversarial Training Methods*, ICLR 2017.
- **Multi-sample dropout**: Inoue, 2019.
- **Temperature scaling** calibration: Guo et al., ICML 2017.
- Full-width→half-width digit/letter normalization; per-field F1-weighted soft-vote ensembling;
  test-time aggregation experiments (see METHODOLOGY.md).

See [METHODOLOGY.md](METHODOLOGY.md) for the full method and [REPRODUCE.md](REPRODUCE.md) to run it.
