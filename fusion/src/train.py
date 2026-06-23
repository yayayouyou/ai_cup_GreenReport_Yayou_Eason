"""
Training script for BERT multi-task ESG classifier.

Usage:
    python -m src.train --config configs/experiment/exp003_bert_ckip_baseline.yaml
"""

import json
import argparse
import os
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import yaml


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

from src.dataset import (
    ESGDataset, IDX_MAPS, LABEL_MAPS, TASKS, SPAN_IGNORE_INDEX,
    MERGED_EVIDENCE_MAP, MERGED_IGNORE_INDEX,
)
from src.models.bert_multitask import BertMultiTask
from src.models.bert_multitask_span import BertMultiTaskSpan
from src.models.bert_multitask_qa import BertMultiTaskQA
from src.models.bert_multitask_merged import BertMultiTaskMerged
from src.evaluate import score, print_report
from src.preprocess.pipeline import run_pipeline
from src.preprocess.split import make_split


def build_model(model_cfg: dict):
    model_type = model_cfg.get("type", "bert_multitask")
    pretrained = model_cfg["pretrained"]
    dropout    = model_cfg.get("dropout", 0.1)
    if model_type == "bert_multitask":
        return BertMultiTask(pretrained, dropout=dropout)
    if model_type == "bert_multitask_span":
        span_label_scheme = model_cfg.get("span_label_scheme", "binary")
        if span_label_scheme not in {"binary", "bio"}:
            raise ValueError(f"Unknown span_label_scheme: {span_label_scheme}")
        default_num_labels = 3 if span_label_scheme == "bio" else 1
        span_num_labels = int(model_cfg.get("span_num_labels", default_num_labels))
        if span_label_scheme == "binary" and span_num_labels != 1:
            raise ValueError("binary span labels require span_num_labels=1")
        if span_label_scheme == "bio" and span_num_labels != 3:
            raise ValueError("BIO span labels require span_num_labels=3")
        return BertMultiTaskSpan(pretrained, dropout=dropout, span_num_labels=span_num_labels)
    if model_type == "bert_multitask_qa":
        return BertMultiTaskQA(pretrained, dropout=dropout)
    if model_type == "bert_multitask_merged":
        return BertMultiTaskMerged(pretrained, dropout=dropout)
    raise ValueError(f"Unknown model type: {model_type}")


def unpack_outputs(out):
    """Returns (cls_logits, promise_token_logits, evidence_token_logits).
    Token logits are None for plain (non-span) models."""
    if isinstance(out, tuple):
        return out
    return out, None, None


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def enforce_constraints(pred: dict) -> dict:
    if pred["promise_status"] == "No":
        pred["evidence_status"]       = "N/A"
        pred["evidence_quality"]      = "N/A"
        pred["verification_timeline"] = "N/A"
    if pred["evidence_status"] == "No":
        pred["evidence_quality"] = "N/A"
    return pred


def batch_to_preds(ids, logits: dict, evidence_2class: bool = False) -> list[dict]:
    logits = dict(logits)
    if evidence_2class:  # head trained N/A->No; never predict N/A directly (recovered via promise gate)
        es = logits["evidence_status"].clone()
        es[:, LABEL_MAPS["evidence_status"]["N/A"]] = float("-inf")
        logits["evidence_status"] = es
    pred_idx = {task: logits[task].argmax(dim=1).cpu().tolist() for task in TASKS}
    preds = []
    for i, rid in enumerate(ids):
        d = {"id": rid}
        for task in TASKS:
            d[task] = IDX_MAPS[task][pred_idx[task][i]]
        enforce_constraints(d)
        preds.append(d)
    return preds


_MERGED_DECODE = {
    MERGED_EVIDENCE_MAP["clear"]:      ("Yes", "Clear"),
    MERGED_EVIDENCE_MAP["weak"]:       ("Yes", "Not Clear"),
    MERGED_EVIDENCE_MAP["no_support"]: ("No",  "N/A"),
}


def batch_to_preds_merged(ids, logits: dict) -> list[dict]:
    """Decode merged-evidence model outputs back to the 4 official fields."""
    p_idx = logits["promise_status"].argmax(dim=1).cpu().tolist()
    t_idx = logits["verification_timeline"].argmax(dim=1).cpu().tolist()
    m_idx = logits["merged_evidence"].argmax(dim=1).cpu().tolist()
    preds = []
    for i, rid in enumerate(ids):
        d = {"id": rid}
        d["promise_status"]        = IDX_MAPS["promise_status"][p_idx[i]]
        d["verification_timeline"] = IDX_MAPS["verification_timeline"][t_idx[i]]
        es, eq = _MERGED_DECODE[m_idx[i]]
        d["evidence_status"]  = es
        d["evidence_quality"] = eq
        enforce_constraints(d)  # promise=No -> all N/A (overrides the merged head)
        preds.append(d)
    return preds


def _safe_span_loss(ce_fn, logits, targets, ign: int = SPAN_IGNORE_INDEX):
    """CrossEntropy that returns 0 (not NaN) when all targets are the ignore index."""
    if (targets != ign).sum() == 0:
        return logits.new_zeros(())
    return ce_fn(logits, targets)


def _raw_task_f1(model, loader, val_records, device, tasks):
    """Mean macro-F1 over `tasks` using RAW head argmax (no enforce_constraints cascade).

    Used for single-task early stopping: the other (untrained) heads are random, so the
    cascade would corrupt the score — evaluate each trained task in isolation.
    """
    from src.evaluate import LABELS
    from sklearn.metrics import f1_score
    model.eval()
    gold = {r["id"]: r for r in val_records}
    pred = {t: {} for t in tasks}
    with torch.no_grad():
        for batch in loader:
            ids       = batch["id"].tolist() if torch.is_tensor(batch["id"]) else list(batch["id"])
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            tt_ids    = batch.get("token_type_ids")
            if tt_ids is not None:
                tt_ids = tt_ids.to(device)
            cls_logits, _, _ = unpack_outputs(model(input_ids, attn_mask, tt_ids))
            for t in tasks:
                am = cls_logits[t].argmax(dim=1).cpu().tolist()
                for i, rid in enumerate(ids):
                    pred[t][rid] = IDX_MAPS[t][am[i]]
    f1s = []
    for t in tasks:
        yt = [gold[i][t] for i in pred[t]]
        yp = [pred[t][i] for i in pred[t]]
        f1s.append(f1_score(yt, yp, labels=LABELS[t], average="macro", zero_division=0))
    return sum(f1s) / len(f1s)


def evaluate_span(model, loader, device):
    """Stage 1 evaluation: returns average val span CE loss (lower = better)."""
    model.eval()
    span_ce = nn.CrossEntropyLoss(ignore_index=SPAN_IGNORE_INDEX)
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            tt_ids    = batch.get("token_type_ids")
            if tt_ids is not None:
                tt_ids = tt_ids.to(device)
            pad_mask    = (1 - attn_mask.float()) * -1e9
            span_logits = model(input_ids, attn_mask, tt_ids)
            for key in span_logits:
                span_logits[key] = span_logits[key] + pad_mask
            loss = (_safe_span_loss(span_ce, span_logits["promise_start"],  batch["promise_start"].to(device)) +
                    _safe_span_loss(span_ce, span_logits["promise_end"],    batch["promise_end"].to(device)) +
                    _safe_span_loss(span_ce, span_logits["evidence_start"], batch["evidence_start"].to(device)) +
                    _safe_span_loss(span_ce, span_logits["evidence_end"],   batch["evidence_end"].to(device)))
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def evaluate(model, loader, val_records, device, merged: bool = False, evidence_2class: bool = False):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            ids       = batch["id"].tolist() if torch.is_tensor(batch["id"]) else list(batch["id"])
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            tt_ids    = batch.get("token_type_ids")
            if tt_ids is not None:
                tt_ids = tt_ids.to(device)

            cls_logits, _, _ = unpack_outputs(model(input_ids, attn_mask, tt_ids))
            if merged:
                all_preds.extend(batch_to_preds_merged(ids, cls_logits))
            else:
                all_preds.extend(batch_to_preds(ids, cls_logits, evidence_2class=evidence_2class))

    return score(val_records, all_preds), all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    args = parser.parse_args()

    exp_cfg   = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pre_cfg   = yaml.safe_load(Path(exp_cfg["preprocess"]).read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(Path(exp_cfg["model"]).read_text(encoding="utf-8"))
    train_cfg = exp_cfg["training"]

    split_seed = exp_cfg.get("data", {}).get("seed", 42)
    seed = train_cfg.get("seed", split_seed)
    if seed is not None:
        set_seed(seed)

    device = get_device()
    print(f"Device     : {device}")
    print(f"Experiment : {exp_cfg['name']}")
    print(f"Model      : {model_cfg['pretrained']}")
    print(f"Split seed : {split_seed}")
    print(f"Train seed : {seed if seed is not None else '(not set)'}")
    print()

    # ── Data split ────────────────────────────────────────────────────────────
    data_cfg  = exp_cfg["data"]
    split_dir = Path(exp_cfg["output_dir"]) / "splits"
    if data_cfg.get("train_file") and data_cfg.get("val_file"):
        train_records = json.loads(Path(data_cfg["train_file"]).read_text(encoding="utf-8"))
        val_records   = json.loads(Path(data_cfg["val_file"]).read_text(encoding="utf-8"))
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train.json").write_text(
            json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (split_dir / "val.json").write_text(
            json.dumps(val_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Split      : precomputed ({len(train_records)} train / {len(val_records)} val)")
    else:
        train_records, val_records = make_split(
            data_cfg["train"], split_dir,
            val_ratio=data_cfg.get("val_split", 0.2),
            seed=data_cfg.get("seed", 42),
        )
    train_records = run_pipeline(train_records, pre_cfg)
    val_records   = run_pipeline(val_records,   pre_cfg)

    augment_path = data_cfg.get("augment")
    if augment_path:
        aug_text = Path(augment_path).read_text(encoding="utf-8")
        if augment_path.endswith(".jsonl"):
            aug_records = [json.loads(l) for l in aug_text.splitlines() if l.strip()]
        else:
            aug_records = json.loads(aug_text)
        aug_records = run_pipeline(aug_records, pre_cfg)
        train_records = train_records + aug_records
        print(f"Augment    : +{len(aug_records)} records from {augment_path}")

    # ── Tokenizer + DataLoaders ───────────────────────────────────────────────
    tokenizer  = AutoTokenizer.from_pretrained(model_cfg["pretrained"])
    max_len    = model_cfg.get("max_length", 256)
    batch_size = train_cfg["batch_size"]

    extra_tokens = train_cfg.get("extra_tokens")  # e.g. ["[NUM]","[DOC]","[VAGUE]"] for cue injection
    if extra_tokens:
        added = tokenizer.add_tokens(list(extra_tokens))
        print(f"Extra toks : added {added} cue tokens {extra_tokens} (vocab now {len(tokenizer)})")

    mode      = train_cfg.get("mode", "cls")
    use_span  = model_cfg.get("type") == "bert_multitask_span"
    is_merged = model_cfg.get("type") == "bert_multitask_merged"
    span_label_scheme = model_cfg.get("span_label_scheme", train_cfg.get("span_label_scheme", "binary"))
    if span_label_scheme not in {"binary", "bio"}:
        raise ValueError(f"Unknown span_label_scheme: {span_label_scheme}")
    use_hints = train_cfg.get("use_hints", False)
    use_qa    = (mode == "span_qa")
    na_as_no  = train_cfg.get("evidence_na_as_no", False)
    if na_as_no:
        print("Mode       : evidence_status N/A folded into No (recovered via promise gate at inference)")
    train_ds = ESGDataset(
        train_records,
        tokenizer,
        max_len,
        return_spans=use_span,
        use_hints=use_hints,
        return_qa_spans=use_qa,
        span_label_scheme=span_label_scheme,
        merged_evidence=is_merged,
        evidence_na_as_no=na_as_no,
    )
    val_ds   = ESGDataset(
        val_records,
        tokenizer,
        max_len,
        return_spans=use_span,
        use_hints=False,
        return_qa_spans=use_qa,
        span_label_scheme=span_label_scheme,
        merged_evidence=is_merged,
    )

    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(model_cfg)
    if extra_tokens:
        model.bert.resize_token_embeddings(len(tokenizer))  # new cue-token embeddings
    model.to(device)

    stage1_ckpt = train_cfg.get("stage1_checkpoint")
    if stage1_ckpt:
        state      = torch.load(stage1_ckpt, map_location=device, weights_only=True)
        bert_state = {k: v for k, v in state.items() if k.startswith("bert.")}
        model.load_state_dict(bert_state, strict=False)
        print(f"Backbone   : loaded from {stage1_ckpt}")

    if use_qa:
        qa_ce = nn.CrossEntropyLoss(ignore_index=SPAN_IGNORE_INDEX)
        print("Mode       : span_qa (Stage 1 — QA span extraction)")

    if use_span:
        span_loss_weight = train_cfg.get("span_loss_weight", 0.5)
        if span_label_scheme == "bio":
            span_ce = nn.CrossEntropyLoss(ignore_index=SPAN_IGNORE_INDEX)
            span_label_desc = "BIO (O/B/I)"
        else:
            span_bce = nn.BCEWithLogitsLoss(reduction="none")
            span_label_desc = "binary"
        print(f"Span heads : ON  (span_loss_weight = {span_loss_weight})")
        print(f"Span labels: {span_label_desc}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    cw_tasks = train_cfg.get("class_weight", [])
    if cw_tasks is True:
        cw_tasks = TASKS
    cw_manual = train_cfg.get("class_weight_manual", {})
    if cw_tasks or cw_manual:
        from collections import Counter
        from src.dataset import LABEL_MAPS as _LMAPS
        cw_dict = {}
        for task in set(list(cw_tasks) + list(cw_manual.keys())):
            lmap = _LMAPS[task]
            if task in cw_manual:
                manual = cw_manual[task]
                w = torch.tensor(
                    [float(manual.get(label, 1.0)) for label, _ in sorted(lmap.items(), key=lambda x: x[1])],
                    dtype=torch.float,
                )
            else:
                counts  = Counter(r[task] for r in train_records)
                n       = len(lmap)
                total_n = sum(counts.values())
                w = torch.tensor(
                    [total_n / (n * counts.get(label, 1)) for label, _ in sorted(lmap.items(), key=lambda x: x[1])],
                    dtype=torch.float,
                )
            cw_dict[task] = w.to(device)
            print(f"class_weight [{task}]: { {label: f'{w[idx].item():.3f}' for label, idx in lmap.items()} }")
    ls = float(train_cfg.get("label_smoothing", 0.0))   # smooth noisy/subjective labels
    if ls > 0:
        print(f"Label smooth: {ls}")
    criteria = {
        task: nn.CrossEntropyLoss(weight=cw_dict[task], label_smoothing=ls) if task in cw_dict
        else nn.CrossEntropyLoss(label_smoothing=ls)
        for task in TASKS
    } if (cw_tasks or cw_manual) else {task: nn.CrossEntropyLoss(label_smoothing=ls) for task in TASKS}

    if is_merged:
        # promise + timeline keep all-record CE; merged_evidence gates promise=No /
        # Misleading via ignore_index (those labels are MERGED_IGNORE_INDEX).
        loss_tasks = ["promise_status", "verification_timeline", "merged_evidence"]
        criteria = {
            "promise_status":        nn.CrossEntropyLoss(),
            "verification_timeline": nn.CrossEntropyLoss(),
            "merged_evidence":       nn.CrossEntropyLoss(ignore_index=MERGED_IGNORE_INDEX),
        }
        print("Mode       : merged evidence head (3-class gated; tasks 2+3 fused)")
    else:
        loss_tasks = TASKS

    # Single/partial-task specialization: only train the listed head(s).
    train_tasks = train_cfg.get("train_tasks")
    if train_tasks and not is_merged:
        loss_tasks = train_tasks
        print(f"Mode       : single/partial-task training on {train_tasks} (other heads untrained)")

    task_loss_weights = train_cfg.get("task_loss_weights", {})  # up-weight hard/high-value tasks
    if task_loss_weights:
        print(f"Task loss w: {task_loss_weights}")

    warmup_ratio = train_cfg.get("warmup_ratio", 0.0)
    if warmup_ratio > 0:
        total_steps  = train_cfg["epochs"] * len(train_loader)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        print(f"Scheduler  : warmup {warmup_steps} steps / total {total_steps} steps")
    else:
        scheduler = None

    # ── Training loop ─────────────────────────────────────────────────────────
    out_dir = Path(exp_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_score = float("inf") if use_qa else 0.0
    patience   = train_cfg.get("early_stopping_patience", 3)
    no_improve = 0

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            tt_ids    = batch.get("token_type_ids")
            if tt_ids is not None:
                tt_ids = tt_ids.to(device)

            if use_qa:
                pad_mask    = (1 - attn_mask.float()) * -1e9
                span_logits = model(input_ids, attn_mask, tt_ids)
                for key in span_logits:
                    span_logits[key] = span_logits[key] + pad_mask
                loss = (_safe_span_loss(qa_ce, span_logits["promise_start"],  batch["promise_start"].to(device)) +
                        _safe_span_loss(qa_ce, span_logits["promise_end"],    batch["promise_end"].to(device)) +
                        _safe_span_loss(qa_ce, span_logits["evidence_start"], batch["evidence_start"].to(device)) +
                        _safe_span_loss(qa_ce, span_logits["evidence_end"],   batch["evidence_end"].to(device)))
            else:
                cls_logits, p_tok_logits, e_tok_logits = unpack_outputs(
                    model(input_ids, attn_mask, tt_ids)
                )
                loss = sum(task_loss_weights.get(task, 1.0) * criteria[task](cls_logits[task], batch[task].to(device)) for task in loss_tasks)

                if use_span:
                    for tok_logits, key in [
                        (p_tok_logits, "promise_token_labels"),
                        (e_tok_logits, "evidence_token_labels"),
                    ]:
                        tok_labels = batch[key].to(device)
                        if (tok_labels != SPAN_IGNORE_INDEX).sum() == 0:
                            continue
                        if span_label_scheme == "bio":
                            loss = loss + span_loss_weight * span_ce(
                                tok_logits.reshape(-1, tok_logits.size(-1)),
                                tok_labels.reshape(-1),
                            )
                        else:
                            mask       = (tok_labels != SPAN_IGNORE_INDEX).float()
                            safe_labels = tok_labels.masked_fill(tok_labels == SPAN_IGNORE_INDEX, 0).float()
                            per_tok     = span_bce(tok_logits, safe_labels)
                            loss        = loss + span_loss_weight * (per_tok * mask).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        marker = ""
        if use_qa:
            val_loss = evaluate_span(model, val_loader, device)
            improved = val_loss < best_score
            if improved:
                best_score = val_loss
                torch.save(model.state_dict(), out_dir / "best_model.pt")
                no_improve = 0
                marker = " ✓"
            else:
                no_improve += 1
                marker = f" (no improve {no_improve}/{patience})"
            print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | val_span_loss={val_loss:.4f}{marker}")
        elif train_tasks and not is_merged:
            # single/partial-task: early-stop on RAW F1 of trained task(s), no cascade
            cur = _raw_task_f1(model, val_loader, val_records, device, train_tasks)
            if cur > best_score:
                best_score = cur
                torch.save(model.state_dict(), out_dir / "best_model.pt")
                no_improve = 0
                marker = " ✓"
            else:
                no_improve += 1
                marker = f" (no improve {no_improve}/{patience})"
            print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | raw_f1[{'+'.join(train_tasks)}]={cur:.4f}{marker}")
        else:
            metrics, _ = evaluate(model, val_loader, val_records, device, merged=is_merged, evidence_2class=na_as_no)
            total_f1   = metrics["total"]
            if total_f1 > best_score:
                best_score = total_f1
                torch.save(model.state_dict(), out_dir / "best_model.pt")
                no_improve = 0
                marker = " ✓"
            else:
                no_improve += 1
                marker = f" (no improve {no_improve}/{patience})"
            task_f1s = "  ".join(
                f"{t[:4]}={metrics[t]['macro_f1']:.3f}" for t in TASKS
            )
            print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | {task_f1s} | total={total_f1:.4f}{marker}")

        if no_improve >= patience:
            print("Early stopping.")
            break

    # ── Final evaluation ──────────────────────────────────────────────────────
    label = "val_span_loss" if use_qa else "val_f1"
    print(f"\nBest {label}: {best_score:.4f}")
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = out_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(yaml.dump(exp_cfg, allow_unicode=True), encoding="utf-8")

    if use_qa:
        print("Stage 1 complete. Backbone saved → use stage1_checkpoint in Stage 2 config.")
    else:
        metrics, final_preds = evaluate(model, val_loader, val_records, device, merged=is_merged, evidence_2class=na_as_no)
        print_report(metrics)
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "val_predictions.json").write_text(
            json.dumps(final_preds, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"Results saved → {run_dir}")


if __name__ == "__main__":
    main()
