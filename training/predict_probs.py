# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yayou (SHIH Ya-You) and Eason (WANG Yi-Hsin) — AI CUP 2026 / TEAM_10049
"""Predict an arbitrary data file with a trained checkpoint, output the ensemble-pipeline
format (probs_<field> softmax in EVAL_FIELDS order + pred_/conf_/gt_/id). Runs in esg-trainer."""
import sys, json, argparse, torch
import torch.nn.functional as F
sys.path.insert(0, '/app')
import train_multitask as T
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True)     # /io/bert_xxx_tv
ap.add_argument('--data', required=True)    # file to predict (val_200 or test_2000)
ap.add_argument('--out', required=True)
a = ap.parse_args()
A = argparse.Namespace(**json.load(open(f'{a.dir}/history.json'))['args'])
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(A.model_name)
model = T.MultiTaskBert(
    model_name=A.model_name, num_labels=T.NUM_LABELS,
    with_verification=bool(getattr(A, 'with_verification', 0)),
    pooling=getattr(A, 'pooling', 'cls_mean'), dropout=getattr(A, 'dropout', 0.1),
    msd_k=getattr(A, 'msd_k', 1), with_span_head=bool(getattr(A, 'use_span_head', 0)),
    num_span_labels=T.NUM_SPAN_LABELS).to(dev)
text_tf = None
if getattr(A, 'text_transform', '') == 'time_bucket_token':
    tok.add_tokens(T.ADDED_TOKENS)              # add the [T_xxx] tokens (match training) BEFORE resize
    model.resize_token_embeddings(len(tok))      # now base+5 -> matches the v18 checkpoint
    text_tf = T.add_time_bucket_token            # prepend the bucket token to input text (match training)
model.load_state_dict(torch.load(f'{a.dir}/best.pt', map_location=dev))
model.eval()
rows = json.load(open(a.data))
# predict-only: ESGDataset reads label fields for its training collate; TEST rows have absent/invalid
# labels (e.g. promise_status 'N/A' which isn't in {Yes,No}) -> KeyError. Sanitize to valid dummies
# (labels are UNUSED for prediction; none of the 12 K-fold models use_promise_aug so input is label-free).
_VALID = {f: set(T.LABEL2ID[f].keys()) for f in T.EVAL_FIELDS}
_DUM = {'promise_status': 'Yes', 'verification_timeline': 'already', 'evidence_status': 'Yes', 'evidence_quality': 'Clear'}
for _r in rows:
    for _f in T.EVAL_FIELDS:
        if _r.get(_f) not in _VALID[_f]:
            _r[_f] = _DUM[_f]
ds = T.ESGDataset(rows, tok, max_len=A.max_len, mask_ratio=0.0, text_transform=text_tf,
                  augment_train=False, text_field=getattr(A, 'text_field', 'data'),
                  use_metadata=bool(getattr(A, 'use_metadata', 0)),
                  use_span_head=bool(getattr(A, 'use_span_head', 0)))
ds.use_promise_aug = bool(getattr(A, 'use_promise_aug', 0))
loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=T.collate)
out = []; idx = 0
with torch.no_grad():
    for batch in loader:
        ids = batch['input_ids'].to(dev); mask = batch['attention_mask'].to(dev)
        o = model(ids, mask); main, _ = T._unpack(o)
        for i in range(ids.size(0)):
            rec = {'id': rows[idx].get('id')}
            for f in T.EVAL_FIELDS:
                p = F.softmax(main[f][i], dim=-1).cpu().tolist()
                rec['probs_' + f] = p
                rec['pred_' + f] = T.ID2LABEL[f][int(main[f][i].argmax())]
                rec['conf_' + f] = max(p)
                rec['gt_' + f] = rows[idx].get(f)
            out.append(rec); idx += 1
json.dump(out, open(a.out, 'w'), ensure_ascii=False)
print(f"WROTE {len(out)} -> {a.out}")
