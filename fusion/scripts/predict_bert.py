"""Run a trained BERT checkpoint over a dataset (val = scored, test = submission).

Handles both architectures:
  - bert_multitask        (4 independent heads)
  - bert_multitask_merged (promise / timeline / merged 3-class evidence head)

Preserves the original `id` type (official val/test use string ids), so the
official scorer can match predictions to ground truth.

Examples:
    # val as simulated test (records carry labels -> auto-scored)
    python scripts/predict_bert.py \
        --config configs/experiment/exp029_merged_evidence.yaml \
        --checkpoint experiments/exp029_full/best_model.pt \
        --data data/raw/vpesg4k_val_1000.json \
        --output experiments/exp029_full/val_predictions.json

    # test submission (no labels)
    python scripts/predict_bert.py --config ... --checkpoint ... \
        --data data/raw/vpesg4k_test_2000.json --output submissions/exp029_test.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from src.train import build_model, enforce_constraints, get_device
from src.dataset import IDX_MAPS, MERGED_EVIDENCE_MAP
from src.preprocess.pipeline import run_pipeline
from src.evaluate import score, print_report

_MERGED_DECODE = {
    MERGED_EVIDENCE_MAP["clear"]:      ("Yes", "Clear"),
    MERGED_EVIDENCE_MAP["weak"]:       ("Yes", "Not Clear"),
    MERGED_EVIDENCE_MAP["no_support"]: ("No",  "N/A"),
}

_FOUR_HEAD_TASKS = ["promise_status", "evidence_status", "evidence_quality", "verification_timeline"]


def _decode_4head(logits: dict, ids: list) -> list[dict]:
    idxs = {t: logits[t].argmax(dim=1).cpu().tolist() for t in _FOUR_HEAD_TASKS}
    preds = []
    for i, rid in enumerate(ids):
        d = {"id": rid}
        for t in _FOUR_HEAD_TASKS:
            d[t] = IDX_MAPS[t][idxs[t][i]]
        enforce_constraints(d)
        preds.append(d)
    return preds


def _decode_merged(logits: dict, ids: list) -> list[dict]:
    p  = logits["promise_status"].argmax(dim=1).cpu().tolist()
    tl = logits["verification_timeline"].argmax(dim=1).cpu().tolist()
    m  = logits["merged_evidence"].argmax(dim=1).cpu().tolist()
    preds = []
    for i, rid in enumerate(ids):
        d = {
            "id": rid,
            "promise_status":        IDX_MAPS["promise_status"][p[i]],
            "verification_timeline": IDX_MAPS["verification_timeline"][tl[i]],
        }
        es, eq = _MERGED_DECODE[m[i]]
        d["evidence_status"]  = es
        d["evidence_quality"] = eq
        enforce_constraints(d)
        preds.append(d)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained BERT checkpoint over a dataset.")
    parser.add_argument("--config", required=True, help="Experiment YAML (for model + preprocess)")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--data", required=True, help="Input JSON (val for scoring, test for submission)")
    parser.add_argument("--output", required=True, help="Where to write predictions JSON")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-score", action="store_true", help="Skip scoring even if labels present")
    args = parser.parse_args()

    exp_cfg   = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(Path(exp_cfg["model"]).read_text(encoding="utf-8"))
    pre_cfg   = yaml.safe_load(Path(exp_cfg["preprocess"]).read_text(encoding="utf-8"))

    is_merged = model_cfg.get("type") == "bert_multitask_merged"
    decode = _decode_merged if is_merged else _decode_4head

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["pretrained"])
    max_len = model_cfg.get("max_length", 256)

    model = build_model(model_cfg)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    records = json.loads(Path(args.data).read_text(encoding="utf-8"))
    records = run_pipeline(records, pre_cfg)

    print(f"Model      : {model_cfg['pretrained']} ({'merged' if is_merged else '4-head'})")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Data       : {args.data}  (n={len(records)})")
    print(f"Device     : {device}")

    all_preds = []
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            chunk = records[start:start + args.batch_size]
            ids   = [r["id"] for r in chunk]
            enc = tokenizer(
                [r["data"] for r in chunk],
                max_length=max_len, padding="max_length", truncation=True, return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            tt_ids = enc.get("token_type_ids")
            if tt_ids is not None:
                tt_ids = tt_ids.to(device)
            logits = model(input_ids, attn_mask, tt_ids)
            all_preds.extend(decode(logits, ids))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_preds, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_preds)} predictions → {out_path}")

    has_labels = records and all(k in records[0] for k in ("promise_status", "evidence_status"))
    if has_labels and not args.no_score:
        metrics = score(records, all_preds)
        print_report(metrics)
        metrics_path = out_path.with_name(out_path.stem + "_metrics.json")
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved metrics → {metrics_path}")


if __name__ == "__main__":
    main()
