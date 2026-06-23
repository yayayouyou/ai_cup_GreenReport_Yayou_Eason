# VeriPromiseESG 2026 — 繁體中文 ESG 承諾驗證

AI CUP 2026 / VeriPromiseESG 競賽最終提交程式。
作者：**yayou**、**Eason**(隊伍)。

任務：對每個繁體中文 ESG 段落(台灣 2024 永續報告書),預測四個欄位,以加權 Macro-F1 評分：

```
總分 = 0.20·F1(promise_status)        # 承諾識別  Yes / No
     + 0.30·F1(evidence_status)       # 證據連結  Yes / No / N/A
     + 0.35·F1(evidence_quality)      # 證據清晰度 Clear / Not Clear / Misleading / N/A（排除 N/A）
     + 0.15·F1(verification_timeline) # 驗證時機  already / within_2_years /
                                      #   between_2_and_5_years / more_than_5_years / N/A（排除 N/A）
```

## 成績

| | 加權分數 |
|---|---|
| public leaderboard | **0.6760** |
| **private（最終）** | **0.6405** |

最終提交檔：[`submissions/FINAL_SUBMISSION_yayou_0.6760.csv`](submissions/FINAL_SUBMISSION_yayou_0.6760.csv)

## 方法總覽

多任務 BERT 集成輸出各欄位的類別機率,再經組合與後處理產生最終標籤：

```
段落文字
   │  全形數字/字母 → 半形正規化（保留中文標點）
   ▼
多任務 BERT 集成（4 個 backbone 家族：ckip-bert / macbert / bge-m3 / roberta-wwm）
   │   • backbone 先做領域/任務自適應 MLM 預訓練（DAPT/TAPT）
   │   • 每個 backbone 用 focal loss + class-weighting 微調（多任務 head）
   │   • 在 held-out 驗證集上挑出最強的子集
   ▼
逐欄位 F1 加權軟投票 → 溫度校準 → ESG 類型先驗
   ▼
規則注入（timeline 年份規則 / quality 規則 / evidence 規則）→ 門檻選擇
   ▼
schema 級聯（promise=No ⇒ 其餘=N/A；evidence∈{No,N/A} ⇒ quality=N/A）+ 合規補選
   ▼
固定的 Misleading override 清單（config/misleading_overrides.txt）
   ▼
submission.csv
```

二元部分（promise / evidence)由 `fusion/`(全形正規化的 BERT 集成,即 `strong4` 子集)產生;
timeline / quality 與所有後處理在 `pipeline/`。兩者由 `pipeline/build_tapt_hybrid.py --external_pe_csv` 接起來。

## 專案結構

```
pipeline/      產生最終 CSV 的後處理 pipeline
  build_tapt_hybrid.py        最終 CSV 建構器（校準 + F1 權重 + 規則 + 級聯 + Misleading override；
                              --external_pe_csv 接入融合二元結果）
  predict_test_realigned.py   核心評分/realignment pipeline（校準、規則、先驗、門檻+timeline offset 選擇）
  metrics.py                  加權 Macro-F1 計分
  rules/esg_type_priors.json  ESG 類型類別先驗
training/      BERT 訓練工具
  train_multitask.py          多任務 BERT 微調（focal loss、class-weighting、R-Drop、FGM、multi-sample dropout）
  train_mlm_tapt.py           領域/任務自適應 MLM 預訓練（DAPT/TAPT）
  predict_probs.py            K-fold OOF + test 機率預測
fusion/        全形正規化 BERT 集成（二元部分）；隊伍共享訓練程式
  fuse_predict.py             集成所選模型 -> 二元 CSV
  src/, scripts/, configs/    多任務 BERT 訓練 + 逐欄位軟投票集成
config/
  misleading_overrides.txt    固定的 Misleading override id 清單（設定輸入）
data_set/, fusion/data/raw/   官方資料集 + 證據 span 檔
submissions/
  FINAL_SUBMISSION_yayou_0.6760.csv
METHODOLOGY.md / METHODOLOGY.zh-TW.md   詳細方法 + 設計決策
REPRODUCE.md  / REPRODUCE.zh-TW.md       逐步重現
```

## 使用的技術(參考文獻)

皆為標準已發表方法,本專案自行重新實作：

- **TAPT / DAPT** — 任務/領域自適應 masked-LM 預訓練：Gururangan et al., *Don't Stop Pretraining*, ACL 2020。
- **Focal loss** 處理類別不平衡：Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017。
- **R-Drop** 正則化：Liang et al., NeurIPS 2021。
- **FGM** 對抗訓練：Miyato et al., *Adversarial Training Methods*, ICLR 2017。
- **Multi-sample dropout**：Inoue, 2019。
- **溫度縮放校準**：Guo et al., ICML 2017。
- 全形→半形數字/字母正規化;逐欄位 F1 加權軟投票集成;test-time aggregation 實驗(見 METHODOLOGY)。

詳見 [METHODOLOGY.zh-TW.md](METHODOLOGY.zh-TW.md) 與 [REPRODUCE.zh-TW.md](REPRODUCE.zh-TW.md)。
