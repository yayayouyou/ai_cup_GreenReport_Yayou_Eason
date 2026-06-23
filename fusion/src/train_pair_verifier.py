"""Train a gold promise/evidence pair verifier and score with official labels.

Usage:
    python -m src.train_pair_verifier --config configs/experiment/exp026_pair_verifier_gold_3way.yaml
"""

import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import yaml

from src.evaluate import score, print_report
from src.models.bert_pair_verifier import BertPairVerifier
from src.pair_dataset import (
    PAIR_IDX_MAP,
    PAIR_LABEL_MAP,
    PAIR_LABELS,
    PairExample,
    PairVerifierDataset,
    build_data_direct_pair_examples,
    build_official_pair_examples,
    build_official_data_direct_examples,
    build_pair_examples,
    make_official_data_direct_predictions,
    make_official_predictions,
    pair_label_counts,
)
from src.preprocess.pipeline import run_pipeline
from src.preprocess.split import make_split


def resolve_pretrained(model_cfg: dict) -> str:
    local_path = model_cfg.get("local_pretrained")
    if local_path and Path(local_path).exists():
        return str(local_path)
    return model_cfg["pretrained"]


def pretrained_kwargs(model_cfg: dict) -> dict:
    kwargs = {}
    if "local_files_only" in model_cfg:
        kwargs["local_files_only"] = bool(model_cfg["local_files_only"])
    if "use_safetensors" in model_cfg:
        kwargs["use_safetensors"] = bool(model_cfg["use_safetensors"])
    return kwargs


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_records(data_cfg: dict, output_dir: Path) -> tuple[list[dict], list[dict]]:
    split_dir = output_dir / "splits"
    if data_cfg.get("train_file") and data_cfg.get("val_file"):
        train_records = json.loads(Path(data_cfg["train_file"]).read_text(encoding="utf-8"))
        val_records = json.loads(Path(data_cfg["val_file"]).read_text(encoding="utf-8"))
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train.json").write_text(
            json.dumps(train_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (split_dir / "val.json").write_text(
            json.dumps(val_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Split      : precomputed ({len(train_records)} train / {len(val_records)} val)")
        return train_records, val_records

    return make_split(
        data_cfg["train"],
        split_dir,
        val_ratio=data_cfg.get("val_split", 0.2),
        seed=data_cfg.get("seed", 42),
    )


def _ids_to_list(ids) -> list:
    if isinstance(ids, torch.Tensor):
        return ids.detach().cpu().tolist()
    return list(ids)


def _batch_logits(model, batch: dict, device: torch.device):
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    tt_ids = batch.get("token_type_ids")
    if tt_ids is not None:
        tt_ids = tt_ids.to(device)
    return model(input_ids, attn_mask, tt_ids)


def evaluate_pair_model(model, loader, device, criterion) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    pair_predictions: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            logits = _batch_logits(model, batch, device)
            labels = batch["label"].to(device)
            loss = criterion(logits, labels)
            pred_ids = logits.argmax(dim=1).detach().cpu().tolist()
            true_ids = labels.detach().cpu().tolist()
            record_ids = _ids_to_list(batch["record_id"])
            pair_ids = list(batch["id"])

            total_loss += loss.item()
            n_batches += 1
            y_true.extend(true_ids)
            y_pred.extend(pred_ids)
            for pair_id, record_id, true_id, pred_id in zip(pair_ids, record_ids, true_ids, pred_ids):
                pair_predictions.append({
                    "id": pair_id,
                    "record_id": record_id,
                    "true_label": PAIR_IDX_MAP[true_id],
                    "pred_label": PAIR_IDX_MAP[pred_id],
                })

    per_label_f1 = {
        label: round(f1_score(y_true, y_pred, labels=[idx], average="macro", zero_division=0), 4)
        for label, idx in PAIR_LABEL_MAP.items()
    }
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(PAIR_IDX_MAP.keys()),
        average="macro",
        zero_division=0,
    )
    return {
        "loss": round(total_loss / max(n_batches, 1), 4),
        "macro_f1": round(macro_f1, 4),
        "per_label_f1": per_label_f1,
        "true_counts": {
            label: Counter(PAIR_IDX_MAP[idx] for idx in y_true).get(label, 0)
            for label in PAIR_LABELS
        },
        "pred_counts": {
            label: Counter(PAIR_IDX_MAP[idx] for idx in y_pred).get(label, 0)
            for label in PAIR_LABELS
        },
        "predictions": pair_predictions,
    }


def predict_official_pairs(
    model,
    tokenizer,
    examples: list[PairExample],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> dict[int | str, str]:
    if not examples:
        return {}

    dataset = PairVerifierDataset(examples, tokenizer, max_length=max_length, has_labels=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    pair_predictions: dict[int | str, str] = {}
    with torch.no_grad():
        for batch in loader:
            logits = _batch_logits(model, batch, device)
            pred_ids = logits.argmax(dim=1).detach().cpu().tolist()
            record_ids = _ids_to_list(batch["record_id"])
            for record_id, pred_id in zip(record_ids, pred_ids):
                pair_predictions[record_id] = PAIR_IDX_MAP[pred_id]
    return pair_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument("--max-train-records", type=int, help="Smoke-test limit for train records")
    parser.add_argument("--max-val-records", type=int, help="Smoke-test limit for val records")
    parser.add_argument("--epochs", type=int, help="Override training.epochs for smoke tests")
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size for smoke tests")
    args = parser.parse_args()

    exp_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pre_cfg = yaml.safe_load(Path(exp_cfg["preprocess"]).read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(Path(exp_cfg["model"]).read_text(encoding="utf-8"))
    train_cfg = dict(exp_cfg["training"])
    pair_cfg = exp_cfg.get("pairing", {})

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    if model_cfg.get("num_labels", 3) != 3:
        raise ValueError("exp026 pair verifier expects num_labels=3")

    split_seed = exp_cfg.get("data", {}).get("seed", 42)
    seed = train_cfg.get("seed", split_seed)
    set_seed(seed)

    output_dir = Path(exp_cfg["output_dir"])
    device = get_device()
    pretrained_path = resolve_pretrained(model_cfg)
    load_kwargs = pretrained_kwargs(model_cfg)
    tokenizer_kwargs = {
        key: value for key, value in load_kwargs.items()
        if key in {"local_files_only"}
    }
    print(f"Device     : {device}")
    print(f"Experiment : {exp_cfg['name']}")
    print(f"Model      : {pretrained_path}")
    print(f"Split seed : {split_seed}")
    print(f"Train seed : {seed}")

    train_records, val_records = load_records(exp_cfg["data"], output_dir)
    if args.max_train_records:
        train_records = train_records[:args.max_train_records]
        print(f"Smoke train: first {len(train_records)} records")
    if args.max_val_records:
        val_records = val_records[:args.max_val_records]
        print(f"Smoke val  : first {len(val_records)} records")

    train_records = run_pipeline(train_records, pre_cfg)
    val_records = run_pipeline(val_records, pre_cfg)

    pair_mode = str(pair_cfg.get("mode", "gold_pair"))
    negative_ratio = int(pair_cfg.get("negative_ratio", 1))
    negative_seed = int(pair_cfg.get("negative_seed", seed))
    if pair_mode == "gold_pair":
        train_pairs = build_pair_examples(train_records, negative_ratio=negative_ratio, seed=negative_seed)
        val_pairs = build_pair_examples(val_records, negative_ratio=negative_ratio, seed=negative_seed)
    elif pair_mode == "data_direct":
        train_pairs = build_data_direct_pair_examples(train_records)
        val_pairs = build_data_direct_pair_examples(val_records)
    else:
        raise ValueError(f"Unknown pairing.mode: {pair_mode}")
    if not train_pairs or not val_pairs:
        raise ValueError("Pair verifier requires non-empty train and val pair examples")

    print(f"Pair mode  : {pair_mode}")
    print(f"Train pairs: {len(train_pairs)} {pair_label_counts(train_pairs)}")
    print(f"Val pairs  : {len(val_pairs)} {pair_label_counts(val_pairs)}")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, **tokenizer_kwargs)
    max_length = int(model_cfg.get("max_length", 256))
    batch_size = int(train_cfg["batch_size"])

    train_ds = PairVerifierDataset(train_pairs, tokenizer, max_length=max_length, has_labels=True)
    val_ds = PairVerifierDataset(val_pairs, tokenizer, max_length=max_length, has_labels=True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    model = BertPairVerifier(
        pretrained_path,
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_labels=int(model_cfg.get("num_labels", 3)),
        **load_kwargs,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_epoch = 0
    patience = int(train_cfg.get("early_stopping_patience", 3))
    no_improve = 0

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            logits = _batch_logits(model, batch, device)
            labels = batch["label"].to(device)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)
        pair_metrics = evaluate_pair_model(model, val_loader, device, criterion)
        improved = pair_metrics["macro_f1"] > best_score
        if improved:
            best_score = pair_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = f" (no improve {no_improve}/{patience})"

        print(
            f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
            f"pair_val_f1={pair_metrics['macro_f1']:.4f}{marker}"
        )
        if no_improve >= patience:
            print("Early stopping.")
            break

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True))
    final_pair_metrics = evaluate_pair_model(model, val_loader, device, criterion)
    final_pair_metrics["best_epoch"] = best_epoch
    final_pair_metrics["train_pair_counts"] = pair_label_counts(train_pairs)
    final_pair_metrics["val_pair_counts"] = pair_label_counts(val_pairs)

    if pair_mode == "gold_pair":
        official_examples = build_official_pair_examples(val_records)
    else:
        official_examples = build_official_data_direct_examples(val_records)
    official_pair_preds = predict_official_pairs(
        model,
        tokenizer,
        official_examples,
        device,
        max_length=max_length,
        batch_size=batch_size * 2,
    )
    if pair_mode == "gold_pair":
        official_predictions = make_official_predictions(val_records, official_pair_preds)
    else:
        official_predictions = make_official_data_direct_predictions(val_records, official_pair_preds)
    official_metrics = score(val_records, official_predictions)

    print_report(official_metrics)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(exp_cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "pair_metrics.json").write_text(
        json.dumps(final_pair_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(official_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "val_predictions.json").write_text(
        json.dumps(official_predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "pair_val_predictions.json").write_text(
        json.dumps(final_pair_metrics["predictions"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Results saved -> {run_dir}")


if __name__ == "__main__":
    main()
