"""Fused binary predictor: ensembles a chosen subset of the fullwidth-normalized BERT models
(per-field softmax soft-vote) and writes a 4-field CSV. The promise/evidence columns of this
output are then spliced into the main pipeline via build_tapt_hybrid.py --external_pe_csv.

  python fuse_predict.py --split full2000 \
      --only ckip_tapt_ep3,macbert_tapt,bgem3,bgem3_tapt \
      --data data/raw/vpesg4k_test_2000.json --out strong4_test.csv
"""
import argparse, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch, yaml, numpy as np
from src.evaluate import TASKS, score
from src.preprocess.pipeline import run_pipeline
from scripts.ensemble_val import _load, predict_probs, decode

# the 4-model "strong4" roster selected on the held-out val-1000 (see METHODOLOGY.md)
ROSTER = {
    "b0_ckip": "bert_ckip_base", "ckip_tapt_ep3": "bert_ckip_tapt_ep3",
    "macbert_base": "macbert_base", "macbert_tapt": "macbert_tapt", "macbert_dapt": "macbert_dapt",
    "bgem3": "bge_m3", "roberta_wwm": "roberta_wwm_base", "bgem3_tapt": "bge_m3_tapt",
}
PCFG = yaml.safe_load((ROOT / "configs/preprocess/fullwidth.yaml").read_text())
COLS = ["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["valeval", "full2000"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma subset of roster names")
    ap.add_argument("--seed_suffix", default="", help="e.g. _s777 to use that seed's models")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    roster = ROSTER if not a.only else {k: ROSTER[k] for k in a.only.split(",")}
    recs = run_pipeline(json.loads(Path(a.data).read_text()), PCFG)
    text_by_id = {r["id"]: r["data"] for r in recs}
    ids = list(text_by_id)
    raw, avail = {}, []
    for name, mcfg in roster.items():
        ck = ROOT / ("experiments/%s_%s%s/best_model.pt" % (a.split, name, a.seed_suffix))
        if not ck.exists():
            print("  SKIP %s (no ckpt)" % name, flush=True); continue
        model, tok, dev = _load("configs/model/%s.yaml" % mcfg, str(ck))
        raw[name] = predict_probs(model, tok, dev, recs); avail.append(name)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    preds = []
    for sid in ids:
        pt = {t: sum(raw[n][sid][t] for n in avail) / len(avail) for t in TASKS}
        d = decode(pt, text_by_id[sid], span=None, rule_mode="dataonly")
        d["id"] = sid
        preds.append(d)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for d in preds:
            w.writerow({c: d[c] for c in COLS})
    print("  WROTE %d -> %s" % (len(preds), a.out), flush=True)
    if a.score:
        m = score(json.loads(Path(a.data).read_text()), preds)
        print("  promise %.4f evid %.4f eq %.4f tl %.4f total %.4f" % (
            m["promise_status"]["macro_f1"], m["evidence_status"]["macro_f1"],
            m["evidence_quality"]["macro_f1"], m["verification_timeline"]["macro_f1"], m["total"]))


if __name__ == "__main__":
    main()
