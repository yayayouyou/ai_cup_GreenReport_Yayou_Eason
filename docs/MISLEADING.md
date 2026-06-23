# `evidence_quality = Misleading` — detection

繁體中文版：[MISLEADING.zh-TW.md](MISLEADING.zh-TW.md)

`Misleading` (greenwashing) is the rarest evidence_quality class — roughly 1–3 per 1000 paragraphs —
so the supervised BERT head produces essentially none. We therefore identify candidate paragraphs with
a **high-precision LLM detector**, and the final submission applies a small fixed override list
(`config/misleading_overrides.txt`).

## Definition (the detector prompt, v3)

A paragraph is `Misleading` only when the *promise* is a concrete/verifiable claim but its *evidence*
fails to substantiate it via one of three patterns. The detector is told to default to "No" for almost
everything and to GROUND key terms (company aliases, framework names, program names) before judging
subject-match:

- **Pattern A — narrow claim + different-subject evidence.** Promise is a single narrow verifiable
  claim (a target / certification / number), but the evidence's core is a genuinely different subject
  (e.g. an SBTi net-zero target ↔ a miscellaneous awards list).
- **Pattern B — ESG metric co-opted for a non-ESG purpose.** An ESG indicator is repurposed as a
  pay/bonus trigger or to dress up a pure governance/financial mechanism.
- **Pattern C — vacuous ("said-but-nothing-said"), strict.** Both promise and evidence are pure generic
  boilerplate with zero concrete content. Strict: any concrete element (a number, a named program, a
  specific action, a certification) ⇒ Clear/Not-Clear, not Misleading.

Output format (per id): `{"m":"Yes"|"No","c":"high"|"med"|"low","pat":"A"|"B"|"C"|"-","why":"<=8 words"}`.

## Models

The candidate paragraphs were scored independently by several LLMs and cross-checked:

| role | model |
|---|---|
| primary semantic scan (via OpenRouter API) | `google/gemini-3.1-pro-preview` (temperature 0.2) |
| agent re-read / scoring | Claude **Opus 4.8** |
| cross-check | `deepseek-reasoner`, Qwen, GLM |

(API keys are read from environment variables — none are stored in this repository.)

## Detector output — confidence-ranked candidates (example run)

The detector returns a ranked candidate list (high→low). One run over the test set produced the
following (id · short reason); the `★` rows were its highest-confidence (top-5):

| rank | id | reason (subject mismatch) |
|---|---|---|
| 1 ★ | 12473 | promise = employee safety/HR; evidence = external parent-child arts event |
| 2 ★ | 12475 | promise = green-office program; evidence = IC-waste gold recovery |
| 3 ★ | 12526 | promise = free public AI course; evidence = AI-customer-service throughput/accuracy |
| 4 ★ | 12599 | promise = anti-fraud public service; evidence = 3G-sunset elderly connectivity / store repair |
| 5 ★ | 12772 | promise = SBTi net-zero target; evidence = a miscellaneous corporate awards list |
| 6 | 12902 | promise = ecology documentary; evidence = a different radio-program project |
| 7 | 13343 | promise = specific reduction target; evidence = "no carbon offsets / RECs used" |
| 8 | 13594 | promise = digital/finance/HR direction; evidence = ecology / tree-planting / forums |
| 9 | 13995 | promise = workforce ageing & succession; evidence = M&A / low-carbon portfolio |
| 10 | 12075 | promise = code of conduct; evidence = external awards + board meeting counts |
| 11 | 12402 | promise = disclose individual director pay; evidence = exec variable pay + anti-bribery cert |
| 12 | 12411 | promise = internal employee learning; evidence = foundation's external STEM outreach |
| 13 | 13068 | promise = AGM procedure; evidence = shareholding structure |
| 14 | 13075 | promise = director training/succession; evidence = nomination-committee composition |
| 15 | 13630 | promise = environment sub-committee reductions; evidence = a different (risk) committee |
| 16 | 13644 | promise = setting a time range; evidence = TNFD scope/drivers |
| 17 | 13700 | promise = annual quantified resource-reduction; evidence = GHG inventory/assurance |
| 18 | 13924 | promise = recruitment diversity; evidence = total headcount figure |

## Final override set (submitted)

The configured override list used in the final submission is the five ids in
[`../config/misleading_overrides.txt`](../config/misleading_overrides.txt):

```
12772   12599   12306   12606   12743
```

These are applied as fixed overrides (promise=Yes, evidence=Yes, quality=Misleading) by
`pipeline/build_tapt_hybrid.py --misleading ...`. The list is a configuration input to the builder;
it overlaps the detector candidates above (e.g. 12772, 12599) but is not identical to any single
detector run's top-k.
