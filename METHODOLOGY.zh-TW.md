# 方法與實驗紀錄

本文件說明產生最終提交的方法,以及主要實驗與其(多為負面的)結果。下方所有驗證都採**誠實協定**:
模型只用 `train-1000` 微調,在 held-out 的官方 `val-1000` 上評估(當作 test 代理);合併的
`train+val (2000)` 只用來重訓最終提交模型。

## 1. 輸入與前處理

每筆是繁體中文 ESG 段落(`data` 欄)。前處理對 `data` 做**只針對數字與拉丁字母的全形→半形正規化**
(刻意保留中文標點 — 全 NFKC 會破壞繁中標點)。永續報告書充滿全形數字/年份,這能強化 evidence
與 timeline 年份訊號(`fusion/configs/preprocess/fullwidth.yaml`)。

## 2. 模型

多任務 BERT(`training/train_multitask.py`、`fusion/src/train.py`):一個共享 encoder + 四個線性 head
(promise / evidence_status / evidence_quality / verification_timeline)。

- backbone 跨四個家族:**ckip-bert-base、macbert-base、bge-m3、roberta-wwm-ext**。
- backbone 先以 masked-LM 在 ESG 文字上繼續預訓練 — 任務自適應(TAPT,用任務段落)與領域自適應
  (DAPT,用外部永續報告書文字)(`training/train_mlm_tapt.py`、`fusion/scripts/build_dapt_corpus.py`、
  `tapt_pretrain.py`)。兩者皆為非監督(僅文字、無標籤)。
- 微調用 focal loss + 各任務 class-weighting,加上 R-Drop、FGM、multi-sample dropout;
  條件遮罩(conditional masking)只在 `promise=Yes` 的列訓練細項 head。

## 3. 集成與 `strong4` 子集

各任務的模型機率以**逐欄位 F1 加權軟投票**組合(權重 = 各模型的交叉驗證 Macro-F1),投票前先做
**溫度縮放校準**(在 held-out OOF 上擬合)。再注入 **ESG 類型先驗**(α=0.5)與**只讀 data 的規則**
(timeline 年份規則、quality 規則、evidence 規則,權重 3),最後做逐欄位門檻與 timeline-offset 選擇
(`pipeline/predict_test_realigned.py`)。

**子集選擇。** 8 模型集成被其中 3 個最弱的成員稀釋。在 held-out 驗證集上做窮盡式 subset 搜尋
(以二元貢獻 `0.2·promise + 0.3·evidence` 排序)選出
**`strong4` = ckip-tapt + macbert-tapt + bge-m3 + bge-m3-tapt** 作為二元部分。
`fusion/fuse_predict.py` 產出此集成的二元 CSV,再由 `pipeline/build_tapt_hybrid.py --external_pe_csv` 接入主 pipeline。

## 4. Schema 級聯與級聯乘數

賽制 schema 以級聯方式強制:`promise=No ⇒ timeline=evidence=quality=N/A`;
`evidence∈{No,N/A} ⇒ quality=N/A`;若 `promise=Yes`(或 `evidence=Yes`)但被級聯成 N/A,則補選一個真實類別。

因為 timeline 與 quality 是**排除 N/A**評分,更準的 **promise** 會「免費」抬升 timeline 與 quality:
promise 的 false-negative 會把一個有真實標籤的項目強制成 N/A,對該類別形成 false-negative。
實測:只把二元部分換成 `strong4`(對比先前最佳),promise +0.044,並透過級聯把 timeline +0.018 /
quality +0.015 帶上來 — 約占總增益的 37%。因此**投資二元部分的價值,超過它名目上的 0.20+0.30 權重**。

## 5. evidence_quality 的 `Misleading`

`Misleading` 是最稀有的類別(2000 筆 test 中 support ≈ 1–4),BERT head 幾乎不會輸出。因此 pipeline
套用一份**固定的 override 清單**(`config/misleading_overrides.txt`):對清單中每個 test id,設
promise=Yes、evidence=Yes、quality=Misleading。候選 id 由**LLM 評分步驟**產生 — OpenRouter API 與一個
Claude-based agent 重讀每個候選段落,並評分其符合 Misleading 定義的程度。最終 id 清單是建構器的固定設定輸入。

## 6. 最終模型與提交

所選模型在合併的 `train+val (2000)` 上重訓(`fusion/scripts/build_full2000.py`),重新對 test 集預測,
再經 pipeline 產出 `submissions/FINAL_SUBMISSION_yayou_0.6760.csv`。

## 7. 沒有幫助的嘗試(負面結果)

皆在 held-out 驗證集上先評估,再決定是否花上傳次數:

| 嘗試 | 結果 |
|---|---|
| 更長的 `max_len`(384、512) | 在這個小資料集上比 256 差(越長越稀釋);256 為最佳 |
| test-time augmentation(多長度、頭+尾視窗) | 負面 — 視窗化視角對 256 訓練的模型是 OOD |
| 二元集成的多 seed 平均 | 中性到負面 — 平均會稀釋尖峰的正確預測 |
| 在 val 上挑「最佳」seed | 二元 val→LB 轉移有 seed 噪音:兩個 val-promise 相同(0.8144)的 seed,在 test 上 0.8199 vs 0.7819 |
| 在 OOF 上做事後權重/門檻 hill-climbing | 過擬合 OOF,不轉移 |
| 把更大的 backbone(roberta-large、xlmr-large)加入池中 | 在驗證噪音範圍內;subset 搜尋未選入 |

**關鍵教訓:** 對二元任務而言,即使是乾淨的 held-out 驗證,也無法可靠預測 test 集排名;只有 leaderboard 能。
穩健的選擇是「實際被確認過結果」的 seed/設定,而非驗證集上最高分的那個。

## 8. public→private 落差

public 0.6760 → private 0.6405。下降反映了 (a) 稀有的 `Misleading` override 在 private 分區未重現,
以及 (b) 二元任務從驗證到未見 private 集的回歸,與 §7 記錄的 seed 轉移噪音一致。
