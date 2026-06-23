"""Train a data-only BIO span candidate extractor for exp027A."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import yaml

from src.dataset import ESGDataset, SPAN_IGNORE_INDEX
from src.models.bert_bio_span_extractor import BertBioSpanExtractor
from src.preprocess.pipeline import run_pipeline
from src.preprocess.split import make_split
from src.span_candidate import build_candidate_record, evaluate_candidate_records


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

    if data_cfg.get("val"):
        train_records = json.loads(Path(data_cfg["train"]).read_text(encoding="utf-8"))
        val_records = json.loads(Path(data_cfg["val"]).read_text(encoding="utf-8"))
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train.json").write_text(
            json.dumps(train_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (split_dir / "val.json").write_text(
            json.dumps(val_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Split      : train/val files ({len(train_records)} train / {len(val_records)} val)")
        return train_records, val_records

    return make_split(
        data_cfg["train"],
        split_dir,
        val_ratio=data_cfg.get("val_split", 0.2),
        seed=data_cfg.get("seed", 42),
    )


def _safe_span_loss(ce_fn, logits, targets, ign: int = SPAN_IGNORE_INDEX):
    if (targets != ign).sum() == 0:
        return logits.new_zeros(())
    return ce_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def span_loss(outputs: dict, batch: dict, device: torch.device, ce_fn) -> torch.Tensor:
    promise_labels = batch["promise_token_labels"].to(device)
    evidence_labels = batch["evidence_token_labels"].to(device)
    return (
        _safe_span_loss(ce_fn, outputs["promise"], promise_labels)
        + _safe_span_loss(ce_fn, outputs["evidence"], evidence_labels)
    )


class SpanInferenceDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        enc = self.tokenizer(
            record["data"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        item = {
            "record_idx": torch.tensor(idx, dtype=torch.long),
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "offset_mapping": enc["offset_mapping"].squeeze(0),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item


def _model_outputs(model, batch: dict, device: torch.device) -> dict:
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    tt_ids = batch.get("token_type_ids")
    if tt_ids is not None:
        tt_ids = tt_ids.to(device)
    return model(input_ids, attn_mask, tt_ids)


def predict_candidate_records(
    model,
    tokenizer,
    records: list[dict],
    device: torch.device,
    max_length: int,
    batch_size: int,
    top_k: int,
) -> list[dict]:
    dataset = SpanInferenceDataset(records, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    candidate_records: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            outputs = _model_outputs(model, batch, device)
            record_indices = batch["record_idx"].detach().cpu().tolist()
            offsets = batch["offset_mapping"].detach().cpu()
            promise_logits = outputs["promise"].detach().cpu()
            evidence_logits = outputs["evidence"].detach().cpu()
            for row_idx, record_idx in enumerate(record_indices):
                candidate_records.append(build_candidate_record(
                    records[record_idx],
                    promise_logits[row_idx],
                    evidence_logits[row_idx],
                    offsets[row_idx],
                    top_k=top_k,
                ))
    return candidate_records


def evaluate_candidates(
    model,
    tokenizer,
    records: list[dict],
    device: torch.device,
    model_cfg: dict,
    candidate_cfg: dict,
    batch_size: int,
) -> tuple[dict, list[dict]]:
    top_k = int(candidate_cfg.get("top_k", 5))
    metric_ks = tuple(int(k) for k in candidate_cfg.get("metric_ks", [1, 3, 5]))
    threshold = float(candidate_cfg.get("hit_threshold", 0.8))
    candidate_records = predict_candidate_records(
        model,
        tokenizer,
        records,
        device,
        max_length=int(model_cfg.get("max_length", 512)),
        batch_size=batch_size,
        top_k=top_k,
    )
    metrics = evaluate_candidate_records(
        records,
        candidate_records,
        ks=metric_ks,
        threshold=threshold,
    )
    return metrics, candidate_records


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
    candidate_cfg = exp_cfg.get("candidate", {})
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    if model_cfg.get("type") != "bert_bio_span_extractor":
        raise ValueError("exp027A expects model type bert_bio_span_extractor")

    split_seed = exp_cfg.get("data", {}).get("seed", 42)
    seed = int(train_cfg.get("seed", split_seed))
    set_seed(seed)

    output_dir = Path(exp_cfg["output_dir"])
    device = get_device()
    print(f"Device     : {device}")
    print(f"Experiment : {exp_cfg['name']}")
    pretrained_path = resolve_pretrained(model_cfg)
    load_kwargs = pretrained_kwargs(model_cfg)
    tokenizer_kwargs = {
        key: value for key, value in load_kwargs.items()
        if key in {"local_files_only"}
    }
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

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, **tokenizer_kwargs)
    max_length = int(model_cfg.get("max_length", 512))
    batch_size = int(train_cfg["batch_size"])

    train_ds = ESGDataset(
        train_records,
        tokenizer,
        max_length=max_length,
        has_labels=True,
        return_spans=True,
        span_label_scheme="bio",
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
    )

    model = BertBioSpanExtractor(
        pretrained_path,
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_labels=int(model_cfg.get("num_labels", 3)),
        **load_kwargs,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=SPAN_IGNORE_INDEX)
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
    early_metric = str(train_cfg.get("early_stopping_metric", "mean_subspan_recall@3"))

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            outputs = _model_outputs(model, batch, device)
            loss = span_loss(outputs, batch, device, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        val_metrics, _ = evaluate_candidates(
            model,
            tokenizer,
            val_records,
            device,
            model_cfg,
            candidate_cfg,
            batch_size=batch_size * 2,
        )
        val_score = float(val_metrics[early_metric])
        improved = val_score > best_score
        if improved:
            best_score = val_score
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = f" (no improve {no_improve}/{patience})"

        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | {early_metric}={val_score:.4f}{marker}")
        if no_improve >= patience:
            print("Early stopping.")
            break

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True))
    final_metrics, val_candidates = evaluate_candidates(
        model,
        tokenizer,
        val_records,
        device,
        model_cfg,
        candidate_cfg,
        batch_size=batch_size * 2,
    )
    final_metrics["best_epoch"] = best_epoch
    final_metrics["early_stopping_metric"] = early_metric
    final_metrics["early_stopping_score"] = round(best_score, 4)
    final_metrics["train_size"] = len(train_records)
    final_metrics["val_size"] = len(val_records)

    print(
        "Final      : "
        f"mean@1={final_metrics.get('mean_subspan_recall@1', 0.0):.4f} "
        f"mean@3={final_metrics.get('mean_subspan_recall@3', 0.0):.4f} "
        f"mean@5={final_metrics.get('mean_subspan_recall@5', 0.0):.4f}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(exp_cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "span_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "val_candidates.json").write_text(
        json.dumps(val_candidates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Results saved -> {run_dir}")


if __name__ == "__main__":
    main()
