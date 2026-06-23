"""Task-Adaptive Pretraining (TAPT, Gururangan et al. ACL 2020).

Continue MLM on the task's *unlabeled* text (train+val+test `data` fields) before
supervised fine-tuning. No labels are used — only the raw text — so including the
test text is the standard transductive TAPT setting.

Output: a HF encoder dir at models/ckip-tapt/ that bert_ckip_tapt.yaml points to.
Early-stops on held-out MLM loss (10% of the corpus, seed=42).
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
MLM_PROB = 0.15
SEED = 42

_p = argparse.ArgumentParser()
_p.add_argument("--pretrained", default="ckiplab/bert-base-chinese")
_p.add_argument("--corpus", default=None,
                help="external corpus .txt (one paragraph/line); if set, used instead of the task JSONs")
_p.add_argument("--out", default=str(ROOT / "models/ckip-tapt"))
_p.add_argument("--lr", type=float, default=5e-5)
_p.add_argument("--batch", type=int, default=16)
_p.add_argument("--max-len", type=int, default=256)
_p.add_argument("--max-epochs", type=int, default=15)
_p.add_argument("--patience", type=int, default=3)
_p.add_argument("--warmup", type=float, default=0.0, help="warmup ratio")
_p.add_argument("--holdout-frac", type=float, default=0.1,
                help="held-out fraction for MLM-loss early-stop; 0 = train on ALL, fixed epochs, save final")
_p.add_argument("--fp16", action="store_true")
_p.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (cut activation memory for large models)")
_ARGS = _p.parse_args()
PRETRAINED = _ARGS.pretrained
OUT = Path(_ARGS.out)
LR = _ARGS.lr
BATCH = _ARGS.batch
MAXLEN = _ARGS.max_len
MAX_EPOCHS = _ARGS.max_epochs
PATIENCE = _ARGS.patience
WARMUP = _ARGS.warmup
HOLDOUT_FRAC = _ARGS.holdout_frac
FP16 = _ARGS.fp16


def load_texts():
    if _ARGS.corpus:
        path = Path(_ARGS.corpus)
        texts = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"corpus file: {path} ({len(texts)} lines)", flush=True)
        return texts
    files = [
        "data/raw/vpesg_4k_train_1000.json",
        "data/raw/vpesg4k_val_1000.json",
        "data/raw/vpesg4k_test_2000.json",
    ]
    texts = []
    for f in files:
        for r in json.loads((ROOT / f).read_text()):
            t = (r.get("data") or "").strip()
            if t:
                texts.append(t)
    return texts


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(PRETRAINED)

    texts = load_texts()
    random.shuffle(texts)
    holdout = HOLDOUT_FRAC > 0
    if holdout:
        n_val = max(1, int(len(texts) * HOLDOUT_FRAC))
        val_texts, train_texts = texts[:n_val], texts[n_val:]
    else:
        val_texts, train_texts = [], texts
    print(f"corpus: {len(texts)} | train {len(train_texts)} | held-out {len(val_texts)} "
          f"| max_len={MAXLEN} lr={LR} warmup={WARMUP} fp16={FP16} "
          f"| mode={'early-stop' if holdout else 'fixed-'+str(MAX_EPOCHS)+'ep'}", flush=True)

    def encode(batch_texts):
        return [tok(t, truncation=True, max_length=MAXLEN) for t in batch_texts]

    train_enc = encode(train_texts)
    val_enc = encode(val_texts)
    collator = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=MLM_PROB)

    model = AutoModelForMaskedLM.from_pretrained(PRETRAINED).to(dev)
    if _ARGS.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print("gradient checkpointing ON", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_enc) / BATCH)
    total_steps = steps_per_epoch * MAX_EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=FP16 and dev.type == "cuda")

    def run_eval():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_enc), BATCH):
                b = collator(val_enc[i:i + BATCH])
                b = {k: v.to(dev) for k, v in b.items()}
                tot += model(**b).loss.item() * b["input_ids"].size(0)
                n += b["input_ids"].size(0)
        return tot / n

    best = math.inf
    noimp = 0
    OUT.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        random.Random(epoch).shuffle(train_enc)
        run_loss, steps = 0.0, 0
        for i in range(0, len(train_enc), BATCH):
            b = collator(train_enc[i:i + BATCH])
            b = {k: v.to(dev) for k, v in b.items()}
            opt.zero_grad()
            with torch.autocast(device_type=dev.type, dtype=torch.float16,
                                enabled=FP16 and dev.type == "cuda"):
                loss = model(**b).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            steps += 1
        if holdout:
            vloss = run_eval()
            ppl = math.exp(vloss) if vloss < 20 else float("inf")
            print(f"epoch{epoch}: train_loss={run_loss/steps:.4f} val_mlm_loss={vloss:.4f} ppl={ppl:.2f}", flush=True)
            if vloss < best - 1e-4:
                best = vloss; noimp = 0
                model.save_pretrained(OUT); tok.save_pretrained(OUT)
            else:
                noimp += 1
                if noimp >= PATIENCE:
                    print(f"early stop at epoch {epoch} (best val_mlm_loss={best:.4f})", flush=True)
                    break
        else:
            tl = run_loss / steps
            print(f"epoch{epoch}: train_loss={tl:.4f} train_ppl={math.exp(min(tl,20)):.2f}", flush=True)
            best = tl

    if not holdout:  # fixed-epoch mode: save final model
        model.save_pretrained(OUT); tok.save_pretrained(OUT)
    print(f"TAPT DONE -> {OUT} | best={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
