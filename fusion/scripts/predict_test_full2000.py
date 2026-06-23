"""Test submission from the FULL-2000 (train+val) retrained models.
Same ensemble recipe as our best: 8-model F1-weighted soft-vote (weights from the
original 1000-CV, relative strengths are stable) + data-timeline rule + cascade.
"""
import csv
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import TASKS
from src.preprocess.pipeline import run_pipeline
from scripts.ensemble_val import _load, predict_probs, decode

TEST = ROOT / "data/raw/vpesg4k_test_2000.json"
PCFG = yaml.safe_load((ROOT / "configs/preprocess/fullwidth.yaml").read_text())
OUT = ROOT / "submissions/ourbest_8model_full2000_test.csv"
COLS = ["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"]

# name -> (model_cfg, full2000 ckpt, cv_dir for F1 weights)
MODELS = {
    "b0_ckip":       ("configs/model/bert_ckip_base.yaml",       "experiments/full2000_b0_ckip/best_model.pt",       "experiments/exp008_bert_fullwidth_cv5_seed42"),
    "ckip_tapt_ep3": ("configs/model/bert_ckip_tapt_ep3.yaml",   "experiments/full2000_ckip_tapt_ep3/best_model.pt", "experiments/exp048_tapt_ep3_cv5_seed42"),
    "ckip_tapt_2e5": ("configs/model/bert_ckip_tapt_lr2e5.yaml", "experiments/full2000_ckip_tapt_2e5/best_model.pt", "experiments/exp047_tapt_lr2e5_cv5_seed42"),
    "macbert_base":  ("configs/model/macbert_base.yaml",         "experiments/full2000_macbert_base/best_model.pt",  "experiments/exp036_zht_macbert_cv5_seed42"),
    "macbert_tapt":  ("configs/model/macbert_tapt.yaml",         "experiments/full2000_macbert_tapt/best_model.pt",  "experiments/exp050_tapt_macbert_cv5_seed42"),
    "macbert_dapt":  ("configs/model/macbert_dapt.yaml",         "experiments/full2000_macbert_dapt/best_model.pt",  "experiments/exp051_dapt_macbert_cv5_seed42"),
    "roberta_wwm":   ("configs/model/roberta_wwm_base.yaml",     "experiments/full2000_roberta_wwm/best_model.pt",   "experiments/exp052_roberta_wwm_cv5_seed42"),
    "bgem3":         ("configs/model/bge_m3.yaml",               "experiments/full2000_bgem3/best_model.pt",         "experiments/exp033_bge_m3_lr1e5_cv5_seed42"),
}


def main():
    missing = [k for k, (_, ck, _) in MODELS.items() if not (ROOT / ck).exists()]
    if missing:
        print("MISSING checkpoints (training not done?):", missing); sys.exit(1)
    test = run_pipeline(json.loads(TEST.read_text()), PCFG)
    text_by_id = {r["id"]: r["data"] for r in test}
    ids = [r["id"] for r in test]

    probs, f1w = {}, {}
    for k, (mc, ck, cvd) in MODELS.items():
        print(f"[{k}] predict test ...", flush=True)
        model, tok, dev = _load(mc, ck)
        probs[k] = predict_probs(model, tok, dev, test)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        agg = json.loads((ROOT / cvd / "cv_results.json").read_text())["aggregate"]
        f1w[k] = {t: agg[t]["macro_f1_mean"] for t in TASKS}

    rows = []
    for sid in ids:
        pt = {}
        for t in TASKS:
            w = sum(f1w[k][t] for k in MODELS)
            pt[t] = sum(f1w[k][t] * probs[k][sid][t] for k in MODELS) / w
        d = decode(pt, text_by_id[sid], span=None, rule_mode="dataonly")
        rows.append({"id": sid, **{c: d[c] for c in COLS[1:]}})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(f"\nwrote {OUT}  ({len(rows)} rows)")
    for c in COLS[1:]:
        print(f"  {c}: {dict(Counter(r[c] for r in rows))}")


if __name__ == "__main__":
    main()
