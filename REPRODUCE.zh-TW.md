# 重現步驟

從零重新產生最終提交的步驟。建議使用 GPU(訓練是在單張 RTX 3090 上完成)。在給定訓練好的模型
checkpoint 與固定設定輸入下,pipeline 是確定性的。

## 0. 環境與資料

```bash
pip install -r requirements.txt          # torch, transformers, scikit-learn, scipy, pyyaml, numpy
```

官方資料已附在 `fusion/data/raw/` 與 `data_set/`
(`vpesg_4k_train_1000.json`、`vpesg4k_val_1000.json`、`vpesg4k_test_2000.json` 等)。

## 1. backbone 的自適應 MLM 預訓練(非監督)

```bash
# 建立(未標註)ESG 語料,對每個 backbone 繼續 masked-LM 預訓練
python fusion/scripts/build_dapt_corpus.py --add-task --out fusion/data/dapt/dapt_corpus_full.txt
python fusion/scripts/tapt_pretrain.py --pretrained hfl/chinese-macbert-base \
    --corpus fusion/data/dapt/dapt_corpus_full.txt --out fusion/models/macbert-dapt \
    --max-len 384 --max-epochs 3 --fp16
# (每個 backbone 重複:ckip-bert、bge-m3、roberta-wwm;TAPT 使用任務文字語料)
```

## 2. 微調多任務 BERT 模型(全形前處理)

```bash
# (a) 驗證模型 — 用 train-1000 訓練,在 held-out val-1000 上評估
python fusion/src/train.py --config fusion/configs/experiment/valeval/ckip_tapt_ep3.yaml
#     ... macbert_tapt, bgem3, bgem3_tapt
# (b) 提交模型 — 用合併的 train+val (2000) 訓練
python fusion/scripts/build_full2000.py
python fusion/src/train.py --config fusion/configs/experiment/full2000/ckip_tapt_ep3.yaml
#     ... macbert_tapt, bgem3, bgem3_tapt
```

(設定 `PYTHONPATH=fusion`,讓 `src` / `scripts` 能正確匯入。)

## 3. 產生融合二元結果(promise + evidence)

```bash
python fusion/fuse_predict.py --split full2000 \
    --only ckip_tapt_ep3,macbert_tapt,bgem3,bgem3_tapt \
    --data fusion/data/raw/vpesg4k_test_2000.json --out strong4_test.csv
```

## 4. 產出最終提交(timeline/quality + Misleading + 後處理)

```bash
python pipeline/build_tapt_hybrid.py \
    --out submissions/FINAL_SUBMISSION_yayou_0.6760.csv \
    --external_pe_csv strong4_test.csv \
    --misleading 12772,12599,12306,12606,12743 \
    --force_evidence_thresh 0.8
```

`build_tapt_hybrid.py` 載入 timeline/quality 機率(來自多任務模型),執行校準 + F1 權重 + 規則 +
ESG 先驗 + 門檻/offset 選擇 + schema 級聯(`pipeline/predict_test_realigned.py`),接入
`--external_pe_csv` 的融合二元結果,並套用 `--misleading` override 清單。結果即為提交檔。
最權威的產物是 `submissions/` 中已 commit 的 CSV。

## 驗證(選用)

```bash
# 誠實驗證:train-1000 模型預測 held-out val-1000(含標籤 -> 自動評分)
python fusion/fuse_predict.py --split valeval --only ckip_tapt_ep3,macbert_tapt,bgem3,bgem3_tapt \
    --data fusion/data/raw/vpesg4k_val_1000.json --out strong4_val.csv --score
```

## 最終建構器的輸入/依賴

`pipeline/build_tapt_hybrid.py` 除了融合二元 CSV 外,還會使用:

- 官方資料 JSON(`data_set/vpesg4k_val_1000.json`、`vpesg4k_test_2000.json`);
- val 與 test 的 **timeline / quality** 模型機率,由 `training/predict_probs.py` 寫入各模型的
  `*_io/bert_<name>/` 目錄(預設 `/tmp/val_io`、`/tmp/test_io`);這些來自多任務 BERT 模型;
- 證據 span 檔(`data_set/claude_spans/*_spans_raw.json`),供只讀 data 的規則使用。

這些產物由 `training/` 與 `fusion/` 的訓練/預測程式產生;路徑可透過
`pipeline/predict_test_realigned.py` 的 argparse 預設值調整。已 commit 的
`submissions/FINAL_SUBMISSION_yayou_0.6760.csv` 為權威輸出。
