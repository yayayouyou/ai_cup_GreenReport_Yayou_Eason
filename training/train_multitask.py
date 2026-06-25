# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yayou (SHIH Ya-You) and Eason (WANG Yi-Hsin) — AI CUP 2026 / TEAM_10049
"""Multi-task BERT trainer — self-contained for 3090 Docker.

Train: vpesg4k_train_1000 V1.json (the full 1000-sample pool)
Val:   vpesg4k_val_1000.json (the official held-out 1000)

Loss: FocalLoss + conditional masking — only train detail fields where Promise=Yes.
Metric: weighted Macro-F1 using OFFICIAL competition weights
  (P 0.20, E 0.30, T 0.15, Q 0.35), with N/A EXCLUDED for timeline & quality (SCORED_LABELS).

Usage inside container:
    docker run --rm --gpus all \\
      -v ~/.cache/huggingface:/root/.cache/huggingface \\
      -v ~/ai_cup_GreenReport/remote/io:/io \\
      esg-trainer \\
      --train /io/train_1000.json --val /io/val_1000.json --out /io/bert_v1
"""
from __future__ import annotations
import argparse
import json
import re
import os
from pathlib import Path

import random as _py_random
import numpy as _np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm


def set_seed(seed: int):
    """Reproducibility seed for multi-seed TTA / variance reduction."""
    _py_random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f">> seed = {seed}", flush=True)

from multi_task_bert import MultiTaskBert


# ── Official competition spec ───────────────────────────────────────
EVAL_FIELDS = {
    "promise_status":       ["Yes", "No"],
    "verification_timeline": ["already", "within_2_years", "between_2_and_5_years",
                              "more_than_5_years", "N/A"],
    "evidence_status":      ["Yes", "No", "N/A"],
    "evidence_quality":     ["Clear", "Not Clear", "Misleading", "N/A"],
}
FIELD_WEIGHTS = {
    "promise_status":        0.20,
    "evidence_status":       0.30,
    "verification_timeline": 0.15,
    "evidence_quality":      0.35,
}
LABEL2ID = {f: {l: i for i, l in enumerate(ls)} for f, ls in EVAL_FIELDS.items()}
ID2LABEL = {f: {i: l for i, l in enumerate(ls)} for f, ls in EVAL_FIELDS.items()}
NUM_LABELS = {f: len(ls) for f, ls in EVAL_FIELDS.items()}

# REAL LB metric: timeline & quality EXCLUDE N/A (it is an unscored, easy, common class for
# those fields); evidence keeps N/A; promise is 2-class. Checkpoint selection must optimize THIS,
# not the N/A-included macro (which inflates ~0.03-0.10 and mis-ranks epochs). See ensemble
# realignment (+0.006 held-out) — same fix applied to model selection here.
SCORED_LABELS = {
    "promise_status":        ["Yes", "No"],
    "verification_timeline": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years"],
    "evidence_status":       ["Yes", "No", "N/A"],
    "evidence_quality":      ["Clear", "Not Clear", "Misleading"],
}


# ── Data layer ──────────────────────────────────────────────────────
# ── Time-bucket token preprocessing ───────────────────────────────
# Extract year tokens, compute delta vs CURRENT_YEAR, prepend bucket token.
# These tokens get added to tokenizer vocab so each survives tokenization
# as a SINGLE id with its own learnable embedding (avoids 30+ subword fragments).
CURRENT_YEAR = 2025  # deployed v18 trained at 2025. The 2024 "fix" (better-aligned hint) tested
                     # WORSE — model over-anchors to the more-accurate deterministic bucket. Keep 2025.
_RE_AD_YEAR = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
_RE_AD_YEAR_CH = re.compile(r"(19\d{2}|20\d{2})\s*年")
_RE_ROC_YEAR = re.compile(r"民國\s*(\d{2,3})")

TIME_BUCKET_TOKENS = {
    "already":               "[T_already]",
    "within_2_years":        "[T_within2]",
    "between_2_and_5_years": "[T_2to5]",
    "more_than_5_years":     "[T_more5]",
    "N/A":                   "[T_NA]",
}
ADDED_TOKENS = list(TIME_BUCKET_TOKENS.values())


def _extract_years(text: str) -> list[int]:
    years = []
    for m in _RE_AD_YEAR.finditer(text):
        y = int(m.group(1))
        if 1990 <= y <= 2099: years.append(y)
    for m in _RE_AD_YEAR_CH.finditer(text):
        y = int(m.group(1))
        if 1990 <= y <= 2099: years.append(y)
    for m in _RE_ROC_YEAR.finditer(text):
        years.append(int(m.group(1)) + 1911)
    return years


def _bucket(delta: int) -> str:
    if delta <= 0:           return "already"
    if delta <= 2:           return "within_2_years"
    if delta <= 5:           return "between_2_and_5_years"
    return "more_than_5_years"


def add_time_bucket_token(text: str) -> str:
    """Prepend [T_xxx] bucket token. Idempotent on no-year-match."""
    if not text:
        return text
    years = _extract_years(text)
    if not years:
        return TIME_BUCKET_TOKENS["N/A"] + " " + text
    target = max(years)
    delta = target - CURRENT_YEAR
    bucket = _bucket(delta)
    return TIME_BUCKET_TOKENS[bucket] + " " + text


def _build_promise_aug_text(r: dict, base_text: str) -> str:
    """如果 promise_string 存在,拼成「[PROMISE] {ps} [EVIDENCE] {es} [TEXT] {data}」格式。

    用戶 06-07 洞察:dataset 可能由 2-stage 標註。給 BERT 顯式 promise + evidence
    span 信號,理想上 Timeline 提升 + Evidence 大幅提升(v34 GT spans 證實 +0.19)。
    若 promise_string 為空(Promise=No),只用 base_text(避免 leakage)。
    若 evidence_string 為空(Evidence=No),省略 [EVIDENCE] 段。
    """
    ps = (r.get('promise_string', '') or '').strip()
    es = (r.get('evidence_string', '') or '').strip()
    if ps and len(ps) >= 5:
        parts = [f"[PROMISE] {ps}"]
        if es and len(es) >= 5:
            parts.append(f"[EVIDENCE] {es}")
        parts.append(f"[TEXT] {base_text}")
        return " ".join(parts)
    return base_text


def _build_metadata_prefix(r: dict) -> str:
    """Prepend ESG type / page bin / company to text as metadata signal.

    From train analysis (06-07):
    - Page bin × Quality: page 100-150 has 63% Clear (highest)
    - Company bias: tcfhc 58% Clear vs fpc 30% NC
    - ESG type already shows preference (E→m5y, S→already, G→N/A)
    """
    esg = (r.get('esg_type', 'E') or 'E').split(';')[0].strip()
    page = int(r.get('page_number', 0))
    if page < 30:
        page_bin = 'P030'
    elif page < 60:
        page_bin = 'P3060'
    elif page < 100:
        page_bin = 'P60100'
    elif page < 150:
        page_bin = 'P100150'
    else:
        page_bin = 'P150'
    company = (r.get('company', 'UNK') or 'UNK').strip().lower()[:20]
    return f"[ESG={esg}] [{page_bin}] [CO={company}] "


SPAN_LABEL_MAP = {'O': 0, 'B-PROM': 1, 'I-PROM': 2, 'B-EVID': 3, 'I-EVID': 4}
NUM_SPAN_LABELS = len(SPAN_LABEL_MAP)


def _build_char_bio_tags(data: str, promise_str: str, evidence_str: str) -> list[int]:
    """For each char of data, assign BIO label (0=O, 1-2=PROM, 3-4=EVID)."""
    n = len(data)
    tags = [0] * n
    if promise_str:
        for part in promise_str.split('｜'):
            part = part.strip()
            if len(part) < 5: continue
            idx = data.find(part)
            if idx >= 0:
                tags[idx] = SPAN_LABEL_MAP['B-PROM']
                for i in range(idx + 1, min(idx + len(part), n)):
                    tags[i] = SPAN_LABEL_MAP['I-PROM']
    if evidence_str:
        for part in evidence_str.split('｜'):
            part = part.strip()
            if len(part) < 5: continue
            idx = data.find(part)
            if idx >= 0:
                if tags[idx] == 0:
                    tags[idx] = SPAN_LABEL_MAP['B-EVID']
                for i in range(idx + 1, min(idx + len(part), n)):
                    if tags[i] == 0:
                        tags[i] = SPAN_LABEL_MAP['I-EVID']
    return tags


class ESGDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_len: int = 512,
                 mask_ratio: float = 0.0, text_transform=None,
                 mask_token_id: int | None = None,
                 augment_train: bool = False,
                 text_field: str = "data",
                 use_metadata: bool = False,
                 use_span_head: bool = False):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len
        self.mask_ratio = float(mask_ratio)
        self.text_transform = text_transform
        self.mask_token_id = mask_token_id or tokenizer.mask_token_id
        self.augment_train = augment_train  # only mask in train mode
        self.text_field = text_field        # "data" (Chinese) or "data_en" (English)
        self.use_metadata = use_metadata    # prepend [ESG=][PAGE=][CO=]
        self.use_promise_aug = False        # set externally if needed
        self.use_span_head = use_span_head  # add BIO tag labels for joint training

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        text = r.get(self.text_field, r.get("data", ""))
        if self.use_metadata:
            text = _build_metadata_prefix(r) + text
        if self.use_promise_aug:
            text = _build_promise_aug_text(r, text)
        if self.text_transform is not None:
            text = self.text_transform(text)
        enc = self.tok(text, max_length=self.max_len,
                       padding="max_length", truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Random token masking augmentation (only during training).
        # Replace ~mask_ratio of non-special tokens with [MASK]. Avoids
        # [CLS]/[SEP]/[PAD]/added-vocab time tokens.
        if self.augment_train and self.mask_ratio > 0 and self.mask_token_id is not None:
            special = self.tok.all_special_ids + [self.tok.pad_token_id]
            # Eligible positions = real tokens, not special, mask=1
            eligible = (attention_mask == 1)
            for sid in special:
                eligible &= (input_ids != sid)
            n_elig = int(eligible.sum().item())
            n_mask = int(n_elig * self.mask_ratio)
            if n_mask > 0:
                elig_idx = torch.where(eligible)[0]
                pick = elig_idx[torch.randperm(len(elig_idx))[:n_mask]]
                input_ids = input_ids.clone()
                input_ids[pick] = self.mask_token_id

        labels = {}
        for f in EVAL_FIELDS:
            val = r.get(f, "N/A")
            labels[f] = LABEL2ID[f].get(val, LABEL2ID[f]["N/A"]) if "N/A" in LABEL2ID[f] else LABEL2ID[f][val]

        # Span labels for joint training (if enabled).
        # Use ORIGINAL data (not augmented) for BIO alignment.
        span_labels = None
        if self.use_span_head:
            orig_data = r.get("data", "")
            ps = (r.get("promise_string", "") or "").strip()
            es = (r.get("evidence_string", "") or "").strip()
            char_tags = _build_char_bio_tags(orig_data, ps, es)
            # Re-tokenize ORIGINAL data with offsets to get token-level tags
            enc2 = self.tok(orig_data, max_length=self.max_len, padding="max_length",
                            truncation=True, return_offsets_mapping=True, return_tensors="pt")
            offsets = enc2["offset_mapping"].squeeze(0).tolist()
            token_tags = []
            for start, end in offsets:
                if start == end:  # special / padding
                    token_tags.append(-100)
                elif end > len(char_tags):
                    token_tags.append(-100)
                else:
                    seg = char_tags[start:end]
                    token_tags.append(seg[0] if seg else 0)
            span_labels = torch.tensor(token_tags, dtype=torch.long)

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if span_labels is not None:
            out["span_labels"] = span_labels
        return out


def collate(batch):
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": {f: torch.tensor([b["labels"][f] for b in batch]) for f in EVAL_FIELDS},
    }
    if "span_labels" in batch[0]:
        out["span_labels"] = torch.stack([b["span_labels"] for b in batch])
    return out


# ── Loss ────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target, alpha=None):
        """alpha (optional) is a 1D tensor of per-class weights, same size as #classes."""
        ce = F.cross_entropy(logits, target, reduction="none", weight=alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def _unpack(out):
    """Split model output into (main, verify).

    Model returns either:
      - dict of {field: logits}  (old, no verify/span)
      - {'main': ..., 'verify': ..., 'span_logits': ...}  (new, joint/verify cases)
    """
    if isinstance(out, dict) and "main" in out:
        return out["main"], out.get("verify")
    return out, None


def _unpack3(out):
    """Return (main, verify, span_logits) from model output."""
    if isinstance(out, dict) and "main" in out:
        return out["main"], out.get("verify"), out.get("span_logits")
    return out, None, None


# Field weights for loss = official competition ratios but SCALED so total = 4.0
# (matching the old equal-weight regime where each of 4 fields contributed 1.0).
# This preserves effective learning rate; without scaling the total loss shrinks 4x.
# Ratios: P:T:E:Q = 0.20:0.15:0.30:0.35  →  scaled to 0.80:0.60:1.20:1.40 (sum=4.0)
FIELD_LOSS_WEIGHTS = {
    "promise_status":        0.80,
    "verification_timeline": 0.60,
    "evidence_status":       1.20,
    "evidence_quality":      1.40,
}

# Class weights — minority classes get more gradient (their recall is weakest:
# promise No 0.59, evidence No 0.43, Not Clear 0.35, Misleading 0.00).
# Orders match EVAL_FIELDS. These module-level defaults are overridden by CLI
# (--promise_alpha/--evidence_alpha/--quality_alpha) when use_class_weights=1.
PROMISE_CLASS_ALPHA = [1.0, 1.0]            # Yes, No
EVIDENCE_CLASS_ALPHA = [1.0, 1.0, 1.0]      # Yes, No, N/A
QUALITY_CLASS_ALPHA = [1.0, 3.0, 5.0, 1.0]  # Clear, Not Clear, Misleading, N/A


# ── Train / eval ────────────────────────────────────────────────────
def _compute_main_loss(main, labs, focal, focal_q, wP, wT, wE, wQ, quality_alpha,
                       promise_alpha=None, evidence_alpha=None):
    """Compute the multi-task main loss (no verification, no R-Drop)."""
    loss = wP * focal(main["promise_status"], labs["promise_status"],
                      alpha=promise_alpha)
    yes_mask = (labs["promise_status"] == LABEL2ID["promise_status"]["Yes"])
    if yes_mask.any():
        loss = loss + wT * focal(main["verification_timeline"][yes_mask],
                                 labs["verification_timeline"][yes_mask])
        loss = loss + wE * focal(main["evidence_status"][yes_mask],
                                 labs["evidence_status"][yes_mask],
                                 alpha=evidence_alpha)
        loss = loss + wQ * focal_q(main["evidence_quality"][yes_mask],
                                   labs["evidence_quality"][yes_mask],
                                   alpha=quality_alpha)
    return loss, yes_mask


def _kl_sym(logits1, logits2):
    """Symmetric KL divergence between two logit sets — used by R-Drop."""
    p = F.log_softmax(logits1, dim=-1)
    q = F.log_softmax(logits2, dim=-1)
    return 0.5 * (F.kl_div(p, q, reduction="batchmean", log_target=True) +
                  F.kl_div(q, p, reduction="batchmean", log_target=True))


class FGM:
    """Fast Gradient Method (Goodfellow et al. 2014).

    Add adversarial perturbation to word embedding during training to
    improve robustness. +0.5-1.2% F1 reported on multiple Chinese NLP tasks.
    """
    def __init__(self, model, epsilon=1.0, emb_name='word_embeddings'):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                if param.grad is None:
                    continue
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


def train_epoch(model, loader, optim, sched, device, focal,
                verify_weight=0.0,
                use_field_weights=False,
                use_class_weights=False,
                use_amp=False,
                grad_accum=1,
                scaler=None,
                focal_q=None,
                rdrop_alpha=0.0,
                fgm=None,
                span_loss_weight=0.0):
    # focal_q (optional) — separate FocalLoss for Quality. Falls back to focal.
    if focal_q is None:
        focal_q = focal
    model.train()
    total = 0.0
    optim.zero_grad()
    autocast = torch.amp.autocast(device_type="cuda", enabled=use_amp)
    for step, batch in enumerate(tqdm(loader, desc="train")):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labs = {f: batch["labels"][f].to(device) for f in EVAL_FIELDS}

        with autocast:
            # Pre-compute per-field class alphas if enabled
            quality_alpha = promise_alpha = evidence_alpha = None
            if use_class_weights:
                quality_alpha = torch.tensor(QUALITY_CLASS_ALPHA,
                                              device=ids.device, dtype=torch.float)
                promise_alpha = torch.tensor(PROMISE_CLASS_ALPHA,
                                             device=ids.device, dtype=torch.float)
                evidence_alpha = torch.tensor(EVIDENCE_CLASS_ALPHA,
                                              device=ids.device, dtype=torch.float)

            # Field weight (use 1.0 each if disabled, official weights if enabled)
            if use_field_weights:
                wP = FIELD_LOSS_WEIGHTS["promise_status"]
                wT = FIELD_LOSS_WEIGHTS["verification_timeline"]
                wE = FIELD_LOSS_WEIGHTS["evidence_status"]
                wQ = FIELD_LOSS_WEIGHTS["evidence_quality"]
            else:
                wP = wT = wE = wQ = 1.0

            # First forward
            out1 = model(ids, mask)
            main1, verify, span_logits = _unpack3(out1)
            loss, yes_mask = _compute_main_loss(main1, labs, focal, focal_q,
                                                wP, wT, wE, wQ, quality_alpha,
                                                promise_alpha, evidence_alpha)
            # Add token-level span loss for joint training (v39+)
            if span_logits is not None and "span_labels" in batch:
                span_labels = batch["span_labels"].to(device)
                span_loss = F.cross_entropy(
                    span_logits.reshape(-1, span_logits.size(-1)),
                    span_labels.reshape(-1), ignore_index=-100,
                )
                loss = loss + span_loss_weight * span_loss

            # R-Drop: second forward + KL between two outputs (Liang et al. 2021)
            if rdrop_alpha > 0:
                out2 = model(ids, mask)
                main2, _ = _unpack(out2)
                loss2, _ = _compute_main_loss(main2, labs, focal, focal_q,
                                              wP, wT, wE, wQ, quality_alpha,
                                              promise_alpha, evidence_alpha)
                loss = 0.5 * (loss + loss2)
                # KL on every task (apply yes_mask for detail fields)
                kl = _kl_sym(main1["promise_status"], main2["promise_status"])
                if yes_mask.any():
                    for f in ("verification_timeline", "evidence_status", "evidence_quality"):
                        kl = kl + _kl_sym(main1[f][yes_mask], main2[f][yes_mask])
                loss = loss + rdrop_alpha * kl

            # ── Verification heads: BCE on "did head_A's argmax match GT?" ──
            main = main1  # alias for verify-head compatibility
            if verify is not None and verify_weight > 0:
                v_loss = 0.0
                for f in EVAL_FIELDS:
                    with torch.no_grad():
                        pred_idx = main[f].argmax(dim=-1)
                        is_correct = (pred_idx == labs[f]).float()
                    if f == "promise_status":
                        sample_mask = torch.ones_like(is_correct, dtype=torch.bool)
                    else:
                        sample_mask = yes_mask
                    if sample_mask.any():
                        v_loss = v_loss + F.binary_cross_entropy_with_logits(
                            verify[f][sample_mask], is_correct[sample_mask]
                        )
                loss = loss + verify_weight * v_loss

            # Scale loss for grad accumulation
            loss = loss / grad_accum

        # Backward (with AMP scaler if enabled)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # FGM adversarial perturbation + 2nd backward
        if fgm is not None:
            fgm.attack()
            with autocast:
                out_adv = model(ids, mask)
                main_adv, _ = _unpack(out_adv)
                loss_adv, _ = _compute_main_loss(main_adv, labs, focal, focal_q,
                                                  wP, wT, wE, wQ, quality_alpha,
                                                  promise_alpha, evidence_alpha)
                loss_adv = loss_adv / grad_accum
            if use_amp and scaler is not None:
                scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()
            fgm.restore()

        # Optimizer step every grad_accum batches
        if (step + 1) % grad_accum == 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            sched.step()
            optim.zero_grad()
        total += loss.item() * grad_accum
    return total / len(loader)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model(ids, mask)
        main, verify = _unpack(out)
        bs = ids.size(0)
        for i in range(bs):
            row = {f: ID2LABEL[f][int(main[f][i].argmax().item())] for f in EVAL_FIELDS}
            # Hierarchical post-process
            if row["promise_status"] == "No":
                row["verification_timeline"] = "N/A"
                row["evidence_status"] = "N/A"
                row["evidence_quality"] = "N/A"
            elif row["evidence_status"] in ("No", "N/A"):
                row["evidence_quality"] = "N/A"
            preds.append(row)
    return preds


def weighted_macro_f1(gts: list[dict], preds: list[dict]) -> tuple[float, dict]:
    out = {}
    weighted = 0.0
    for f in EVAL_FIELDS:
        y = [g.get(f, "N/A") for g in gts]
        p = [pr[f] for pr in preds]
        macro = f1_score(y, p, labels=SCORED_LABELS[f], average="macro", zero_division=0)
        out[f] = macro
        weighted += macro * FIELD_WEIGHTS[f]
    return weighted, out


# ── Main ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.environ.get("TRAIN_FILE", "/io/train_1000.json"))
    ap.add_argument("--val",   default=os.environ.get("VAL_FILE",   "/io/val_1000.json"))
    ap.add_argument("--out",   default=os.environ.get("OUT_DIR",    "/io/bert_v1"))
    ap.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "hfl/chinese-roberta-wwm-ext"))
    # Defaults: focal loss + per-task class-weighting recipe:
    #   max_len 384, lr 3e-5, cosine schedule, AMP fp16, grad_accum=2.
    ap.add_argument("--max_len", type=int, default=int(os.environ.get("MAX_LEN", "384")))
    ap.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("LR", "3e-5")))
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "8")))
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int,
                    default=int(os.environ.get("GRAD_ACCUM", "2")),
                    help="Accumulate gradients over N batches before optimizer step "
                         "(effective batch = batch_size * grad_accum)")
    ap.add_argument("--use_amp", type=int,
                    default=int(os.environ.get("USE_AMP", "1")),
                    help="Mixed-precision fp16 training (1=enable, 0=disable)")
    ap.add_argument("--scheduler", type=str,
                    default=os.environ.get("SCHEDULER", "cosine"),
                    choices=["linear", "cosine"],
                    help="LR schedule (default cosine, was linear pre-Phase1)")
    ap.add_argument("--pooling", type=str,
                    default=os.environ.get("POOLING", "cls_mean"),
                    choices=["cls", "mean", "cls_mean"],
                    help="Pooling mode (default cls_mean = CLS||attn-mean concat)")
    ap.add_argument("--dropout", type=float,
                    default=float(os.environ.get("DROPOUT", "0.1")),
                    help="Per-task dropout rate (default 0.1)")
    ap.add_argument("--quality_focal_gamma", type=float,
                    default=float(os.environ.get("QUALITY_FOCAL_GAMMA", "2.0")),
                    help="Focal gamma for Quality field only (default 3.0). "
                         "Other fields keep gamma=2.0.")
    ap.add_argument("--text_field", default=os.environ.get("TEXT_FIELD", "data"),
                    help="Field in JSON containing the input text. "
                         "Use 'data_en' to train on English translations.")
    ap.add_argument("--use_metadata", type=int,
                    default=int(os.environ.get("USE_METADATA", "0")),
                    help="Prepend [ESG=][PAGE bin=][COMPANY=] as metadata prefix")
    ap.add_argument("--fgm_epsilon", type=float,
                    default=float(os.environ.get("FGM_EPSILON", "0.0")),
                    help="FGM adversarial training epsilon (Goodfellow 2014). "
                         "Try 0.5-1.0. 0 = disabled.")
    ap.add_argument("--use_promise_aug", type=int,
                    default=int(os.environ.get("USE_PROMISE_AUG", "0")),
                    help="Augment input with [PROMISE] {promise_string} [SEP] [TEXT] prefix. "
                         "Helps Timeline classification via 2-stage signal.")
    ap.add_argument("--use_span_head", type=int,
                    default=int(os.environ.get("USE_SPAN_HEAD", "0")),
                    help="Enable joint training with token-level BIO span head. "
                         "Model outputs main + span_logits; loss = main + span_weight * span_loss.")
    ap.add_argument("--span_loss_weight", type=float,
                    default=float(os.environ.get("SPAN_LOSS_WEIGHT", "0.5")),
                    help="Weight for token-level span classification loss in joint training.")
    # Phase A new flags:
    ap.add_argument("--rdrop_alpha", type=float,
                    default=float(os.environ.get("RDROP_ALPHA", "0")),
                    help="R-Drop KL weight. Run forward 2x with different dropout, "
                         "add alpha * sym_KL(p1, p2). Try 0.5 (Liang et al.).")
    ap.add_argument("--msd_k", type=int,
                    default=int(os.environ.get("MSD_K", "1")),
                    help="Multi-Sample Dropout K — average logits over K dropout "
                         "passes during training. Try 5 (Inoue 2019).")
    ap.add_argument("--mask_ratio", type=float,
                    default=float(os.environ.get("MASK_RATIO", "0")),
                    help="Random token mask ratio during training. Try 0.10.")
    ap.add_argument("--resample_t4", type=float,
                    default=float(os.environ.get("RESAMPLE_T4", "0")),
                    help="If >0, use WeightedRandomSampler with weight = 1/count^alpha. "
                         "alpha=1 = inverse freq, 0.5 = sqrt-inv-freq.")
    ap.add_argument("--text_transform", default=os.environ.get("TEXT_TRANSFORM", ""),
                    choices=["", "time_bucket_token"],
                    help="Optional text-level transform; 'time_bucket_token' prepends "
                         "[T_xxx] time-bucket token prefix.")
    ap.add_argument("--with_verification", type=int,
                    default=int(os.environ.get("WITH_VERIFICATION", "0")),
                    help="Add 4 binary verification heads (1=enable, 0=disable)")
    ap.add_argument("--verify_weight", type=float,
                    default=float(os.environ.get("VERIFY_WEIGHT", "0.3")),
                    help="Weight on verification loss in joint training")
    ap.add_argument("--use_field_weights", type=int,
                    default=int(os.environ.get("USE_FIELD_WEIGHTS", "0")),
                    help="Weight per-field loss by official competition weights "
                         "(Quality 0.35 > Evidence 0.30 > Promise 0.20 > Timeline 0.15)")
    ap.add_argument("--use_class_weights", type=int,
                    default=int(os.environ.get("USE_CLASS_WEIGHTS", "0")),
                    help="Apply class-weighted focal loss on minority classes")
    ap.add_argument("--promise_alpha", default=os.environ.get("PROMISE_ALPHA") or None,
                    help="comma class weights Yes,No e.g. '1,3' (needs use_class_weights)")
    ap.add_argument("--evidence_alpha", default=os.environ.get("EVIDENCE_ALPHA") or None,
                    help="comma class weights Yes,No,N/A e.g. '1,3,1'")
    ap.add_argument("--quality_alpha", default=os.environ.get("QUALITY_ALPHA") or None,
                    help="comma class weights Clear,NotClear,Misleading,N/A e.g. '1,3,5,1'")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")),
                    help="Random seed for variance reduction / multi-seed TTA")
    args = ap.parse_args()

    # CLI overrides for per-field class alphas (only used when use_class_weights=1)
    global PROMISE_CLASS_ALPHA, EVIDENCE_CLASS_ALPHA, QUALITY_CLASS_ALPHA
    if args.promise_alpha:
        PROMISE_CLASS_ALPHA = [float(x) for x in args.promise_alpha.split(',')]
    if args.evidence_alpha:
        EVIDENCE_CLASS_ALPHA = [float(x) for x in args.evidence_alpha.split(',')]
    if args.quality_alpha:
        QUALITY_CLASS_ALPHA = [float(x) for x in args.quality_alpha.split(',')]

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> device={device}  model={args.model_name}  bs={args.batch_size}  epochs={args.epochs}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.train, "r", encoding="utf-8") as f:
        train_rows = json.load(f)
    with open(args.val, "r", encoding="utf-8") as f:
        val_rows = json.load(f)
    print(f">> train={len(train_rows)}  val={len(val_rows)}")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    # Time-token preprocessing: register new vocab tokens
    text_tf = None
    if args.text_transform == "time_bucket_token":
        added = tok.add_tokens(ADDED_TOKENS, special_tokens=True)
        print(f">> text_transform=time_bucket_token  added {added} new vocab tokens: {ADDED_TOKENS}")
        text_tf = add_time_bucket_token

    train_ds = ESGDataset(train_rows, tok, max_len=args.max_len,
                          mask_ratio=args.mask_ratio, text_transform=text_tf,
                          augment_train=True, text_field=args.text_field,
                          use_metadata=bool(args.use_metadata),
                          use_span_head=bool(args.use_span_head))
    val_ds = ESGDataset(val_rows, tok, max_len=args.max_len,
                        mask_ratio=0.0, text_transform=text_tf,
                        augment_train=False, text_field=args.text_field,
                        use_metadata=bool(args.use_metadata),
                        use_span_head=bool(args.use_span_head))
    train_ds.use_promise_aug = bool(args.use_promise_aug)
    val_ds.use_promise_aug = bool(args.use_promise_aug)
    print(f">> text_field={args.text_field}  use_metadata={args.use_metadata}  use_promise_aug={args.use_promise_aug}")

    # WeightedRandomSampler for T4 (Quality) class balancing
    train_sampler = None
    if args.resample_t4 > 0:
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler
        labels = [r.get("evidence_quality", "N/A") for r in train_rows]
        counts = Counter(labels)
        alpha = args.resample_t4
        weights = []
        for lab in labels:
            c = max(counts[lab], 1)
            weights.append(1.0 / (c ** alpha))
        train_sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                               replacement=True)
        print(f">> resample_t4 alpha={alpha}  class counts={dict(counts)}")
        print(f">> sample weights range: [{min(weights):.5f}, {max(weights):.5f}]")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = MultiTaskBert(
        model_name=args.model_name, num_labels=NUM_LABELS,
        with_verification=bool(args.with_verification),
        pooling=args.pooling,
        dropout=args.dropout,
        msd_k=args.msd_k,
        with_span_head=bool(args.use_span_head),
        num_span_labels=NUM_SPAN_LABELS,
    ).to(device)
    # If we added new tokens to tokenizer (time-token), resize encoder embeddings
    if args.text_transform == "time_bucket_token":
        new_size = len(tok)
        model.resize_token_embeddings(new_size)
        print(f">> resized encoder embeddings to {new_size} tokens (after time-token add)")
    print(f">> with_verification={bool(args.with_verification)}  "
          f"verify_weight={args.verify_weight}")
    print(f">> use_field_weights={bool(args.use_field_weights)}  "
          f"use_class_weights={bool(args.use_class_weights)}")
    print(f">> scheduler={args.scheduler}  use_amp={bool(args.use_amp)}  "
          f"grad_accum={args.grad_accum}  effective_bs={args.batch_size * args.grad_accum}")
    print(f">> rdrop_alpha={args.rdrop_alpha}  msd_k={args.msd_k}  "
          f"mask_ratio={args.mask_ratio}  resample_t4={args.resample_t4}  "
          f"text_transform={args.text_transform or 'none'}")
    if args.use_field_weights:
        print(f">>   field weights: {FIELD_LOSS_WEIGHTS}")
    if args.use_class_weights:
        print(f">>   class alphas: promise(Yes,No)={PROMISE_CLASS_ALPHA}  "
              f"evidence(Yes,No,N/A)={EVIDENCE_CLASS_ALPHA}  "
              f"quality(Clear,NC,Mis,N/A)={QUALITY_CLASS_ALPHA}")
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # Optimizer steps = batches/grad_accum * epochs
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    if args.scheduler == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    else:
        sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)
    focal = FocalLoss(gamma=2.0)
    focal_q = FocalLoss(gamma=args.quality_focal_gamma) if args.quality_focal_gamma != 2.0 else focal
    print(f">> focal gamma: P/T/E=2.0  Q={args.quality_focal_gamma}")
    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None
    fgm = None
    if args.fgm_epsilon > 0:
        fgm = FGM(model, epsilon=args.fgm_epsilon)
        print(f">> FGM 對抗訓練 enabled, epsilon={args.fgm_epsilon}")

    best_f1 = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optim, sched, device, focal,
                           verify_weight=args.verify_weight if args.with_verification else 0.0,
                           use_field_weights=bool(args.use_field_weights),
                           use_class_weights=bool(args.use_class_weights),
                           use_amp=bool(args.use_amp),
                           grad_accum=args.grad_accum,
                           scaler=scaler,
                           focal_q=focal_q,
                           rdrop_alpha=args.rdrop_alpha,
                           fgm=fgm,
                           span_loss_weight=args.span_loss_weight if args.use_span_head else 0.0)
        val_preds = predict(model, val_loader, device)
        weighted, per_field = weighted_macro_f1(val_rows, val_preds)
        print(f"epoch {epoch} loss={loss:.4f} weighted_F1={weighted:.4f}  "
              + " ".join(f"{k[:4]}={v:.3f}" for k, v in per_field.items()))
        history.append({"epoch": epoch, "loss": loss, "weighted_f1": weighted, "per_field": per_field})

        if weighted > best_f1:
            best_f1 = weighted
            torch.save(model.state_dict(), out_dir / "best.pt")
            with (out_dir / "val_predictions.json").open("w", encoding="utf-8") as f:
                json.dump([{"id": v.get("id"), **p} for v, p in zip(val_rows, val_preds)],
                          f, ensure_ascii=False, indent=2)
            print(f"  ⭐ new best {best_f1:.4f} saved → {out_dir}/best.pt")

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump({"best_f1": best_f1, "history": history, "args": vars(args)},
                  f, ensure_ascii=False, indent=2)
    print(f"\n>> done. best weighted F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
