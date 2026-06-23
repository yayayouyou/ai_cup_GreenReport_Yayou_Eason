# Reproduction

End-to-end steps to regenerate the final submission. GPU recommended (training was done on a
single RTX 3090). The pipeline is deterministic given the trained model checkpoints and the
fixed configuration inputs.

## 0. Environment & data

```bash
pip install -r requirements.txt          # torch, transformers, scikit-learn, scipy, pyyaml, numpy
```

Place the official data under `fusion/data/raw/`:
`vpesg_4k_train_1000.json`, `vpesg4k_val_1000.json`, `vpesg4k_test_2000.json`.

## 1. Adaptive MLM pre-training of the backbones (unsupervised)

```bash
# build the (unlabeled) ESG corpus and continue masked-LM pretraining each backbone
python fusion/scripts/build_dapt_corpus.py --add-task --out fusion/data/dapt/dapt_corpus_full.txt
python fusion/scripts/tapt_pretrain.py --pretrained hfl/chinese-macbert-base \
    --corpus fusion/data/dapt/dapt_corpus_full.txt --out fusion/models/macbert-dapt \
    --max-len 384 --max-epochs 3 --fp16
# (repeat per backbone: ckip-bert, bge-m3, roberta-wwm; TAPT uses the task-text corpus)
```

## 2. Fine-tune the multi-task BERT models (full-width preprocessing)

```bash
# (a) validation models — train on train-1000, evaluate on held-out val-1000
python fusion/src/train.py --config fusion/configs/experiment/valeval/ckip_tapt_ep3.yaml
#     ... macbert_tapt, bgem3, bgem3_tapt
# (b) submission models — train on the combined train+val (2000)
python fusion/scripts/build_full2000.py
python fusion/src/train.py --config fusion/configs/experiment/full2000/ckip_tapt_ep3.yaml
#     ... macbert_tapt, bgem3, bgem3_tapt
```

(`PYTHONPATH=fusion` so `src` / `scripts` import correctly.)

## 3. Produce the fused binary (promise + evidence)

```bash
python fusion/fuse_predict.py --split full2000 \
    --only ckip_tapt_ep3,macbert_tapt,bgem3,bgem3_tapt \
    --data fusion/data/raw/vpesg4k_test_2000.json --out strong4_test.csv
```

## 4. Emit the final submission (timeline/quality + Misleading + post-processing)

```bash
python pipeline/build_tapt_hybrid.py \
    --out submissions/FINAL_SUBMISSION_yayou_0.6760.csv \
    --external_pe_csv strong4_test.csv \
    --misleading 12772,12599,12306,12606,12743 \
    --force_evidence_thresh 0.8
```

`build_tapt_hybrid.py` loads the timeline/quality probabilities (from the multi-task models),
runs calibration + F1-weighting + rules + ESG prior + threshold/offset selection + the schema
cascade (`pipeline/predict_test_realigned.py`), splices the fused binary from
`--external_pe_csv`, and applies the `--misleading` override list. The result is the submitted
file. The authoritative artifact is the committed CSV in `submissions/`.

## Validation (optional)

```bash
# honest val: train-1000 models predict the held-out val-1000 (labels present -> auto-scored)
python fusion/fuse_predict.py --split valeval --only ckip_tapt_ep3,macbert_tapt,bgem3,bgem3_tapt \
    --data fusion/data/raw/vpesg4k_val_1000.json --out strong4_val.csv --score
```

## Inputs / dependencies for the final builder

`pipeline/build_tapt_hybrid.py` consumes, in addition to the fused binary CSV:

- the official data JSONs (`data_set/vpesg4k_val_1000.json`, `vpesg4k_test_2000.json`);
- the **timeline / quality** model probabilities for the val and test sets, written by
  `training/predict_probs.py` into per-model `*_io/bert_<name>/` directories (default
  `/tmp/val_io`, `/tmp/test_io`); these come from the multi-task BERT models;
- evidence span files (`data_set/claude_spans/*_spans_raw.json`) used by the data-only rules.

These artifacts are produced by the training/prediction code in `training/` and `fusion/`; paths
are configurable via the argparse defaults in `pipeline/predict_test_realigned.py`. The committed
`submissions/FINAL_SUBMISSION_yayou_0.6760.csv` is the authoritative output.
