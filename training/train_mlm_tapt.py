# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yayou (SHIH Ya-You) and Eason (WANG Yi-Hsin) — AI CUP 2026 / TEAM_10049
"""TAPT — Task-Adaptive Pre-Training (Gururangan et al., ACL 2020).
Continue masked-LM pretraining of a backbone on the ESG task corpus (train+val+test text,
unlabeled), producing a domain-adapted backbone for downstream fine-tuning. Runs in esg-trainer.

  docker run ... -e MODEL_NAME=hfl/chinese-macbert-base -e OUT_DIR=/io/tapt_macbert \
                 -e CORPUS=/io/tapt_corpus.txt --entrypoint python esg-trainer:latest /app/train_mlm_tapt.py
"""
import argparse, os, math, torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)


class LineDS(Dataset):
    def __init__(self, path, tok, max_len, limit=0):
        lines = [l.strip() for l in open(path, encoding='utf-8') if l.strip()]
        if limit:
            lines = lines[:limit]
        self.enc = [tok(l, truncation=True, max_length=max_len) for l in lines]

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        return {'input_ids': self.enc[i]['input_ids'],
                'attention_mask': self.enc[i]['attention_mask']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.environ.get('MODEL_NAME', 'hfl/chinese-macbert-base'))
    ap.add_argument('--corpus', default=os.environ.get('CORPUS', '/io/tapt_corpus.txt'))
    ap.add_argument('--out', default=os.environ.get('OUT_DIR', '/io/tapt_out'))
    ap.add_argument('--epochs', type=float, default=float(os.environ.get('EPOCHS', '10')))
    ap.add_argument('--max_len', type=int, default=int(os.environ.get('MAX_LEN', '384')))
    ap.add_argument('--batch_size', type=int, default=int(os.environ.get('BATCH_SIZE', '16')))
    ap.add_argument('--lr', type=float, default=float(os.environ.get('LR', '5e-5')))
    ap.add_argument('--mlm_prob', type=float, default=float(os.environ.get('MLM_PROB', '0.15')))
    ap.add_argument('--limit', type=int, default=int(os.environ.get('LIMIT', '0')), help='smoke: only first N lines')
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForMaskedLM.from_pretrained(a.model)
    ds = LineDS(a.corpus, tok, a.max_len, a.limit)
    print(f">> TAPT {a.model}: {len(ds)} lines, {a.epochs} ep, max_len={a.max_len}, mlm_prob={a.mlm_prob}", flush=True)
    coll = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=a.mlm_prob)

    targs = TrainingArguments(
        output_dir=a.out, overwrite_output_dir=True,
        num_train_epochs=a.epochs, per_device_train_batch_size=a.batch_size,
        learning_rate=a.lr, warmup_ratio=0.06, weight_decay=0.01,
        fp16=torch.cuda.is_available(), logging_steps=50,
        save_strategy='no', report_to=[],
    )
    tr = Trainer(model=model, args=targs, train_dataset=ds, data_collator=coll)
    out = tr.train()
    loss = out.training_loss
    print(f">> final train loss={loss:.4f}  perplexity={math.exp(min(loss, 20)):.2f}", flush=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f">> saved TAPT backbone -> {a.out}", flush=True)


if __name__ == '__main__':
    main()
