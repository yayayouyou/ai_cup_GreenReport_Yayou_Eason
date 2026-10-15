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

> Misleading detection (prompt + models + candidate list): [docs/MISLEADING.md](docs/MISLEADING.md)

## Authors

| | GitHub | Affiliation |
|---|---|---|
| Ya-You Shih (yayou) | [@yayayouyou](https://github.com/yayayouyou) | National Central University |
| Yi-Hsin Wang (Eason) | [@Eason20050201](https://github.com/Eason20050201) | National Taiwan Normal University |

The accompanying NTCIR-19 paper is co-authored with Chia-Hui Chang (National Central University).

## Citation

If you use this code, please cite our NTCIR-19 paper:

```bibtex
@inproceedings{shih2026team10049,
  title     = {{TEAM\_10049} at the {NTCIR-19} {AI CUP} {VeriPromise ESG} Track: What a Field Is Really Worth under a Gated, Rare-Class Metric},
  author    = {Shih, Ya-You and Wang, Yi-Hsin and Chang, Chia-Hui},
  booktitle = {Proceedings of the 19th NTCIR Conference on Evaluation of Information Access Technologies},
  address   = {Tokyo, Japan},
  year      = {2026}
}
```

## License

The source code in this repository is licensed under the **GNU Affero General Public
License v3.0 (AGPL-3.0)** — a strong copyleft license. You are free to use, study,
modify, and share this code, **but any derivative work — and any service you build on
it that is offered to users over a network — must also be released as open source
under the AGPL-3.0, with complete corresponding source code made available**
(AGPL-3.0 §13 closes the network/SaaS loophole). See [LICENSE](LICENSE).

**Scope.** The license covers our source code only. The bundled official AI CUP 2026
dataset (`data_set/`, `fusion/data/raw/`) is **not** covered — it belongs to the
competition organizers and is included solely for reproducibility. Referenced
Hugging Face backbones and third-party libraries remain under their own licenses.
Full scope and copyright in [NOTICE](NOTICE).

Copyright (C) 2026 yayou (SHIH Ya-You) and Eason (WANG Yi-Hsin).
