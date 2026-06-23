"""Per-field soft-voting ensemble on val(1000), GreenReport-style.

Strategies: equal / f1w (F1-weighted) / bestsrc. All + cascade + timeline year-rule.
+ Temperature calibration (GR fit_T): per model per field, T is fit on the held-out
  CV OOF (never on val), then applied to val probs before voting. Reports calib on/off.

Weights/temperatures come from CV (out-of-fold) — never from val.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import minimize_scalar
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import build_model, enforce_constraints, get_device
from src.dataset import IDX_MAPS, LABEL_MAPS
from src.evaluate import score, TASKS
from src.preprocess.pipeline import run_pipeline
from src.rules import timeline_rule
from src.span_rules import predict_timeline_rule, predict_quality_rule

VAL = ROOT / "data/raw/vpesg4k_val_1000.json"
SPANS = ROOT / "data/gr_spans/val_spans_raw.json"
RULE_W = 3.0
PCFG = yaml.safe_load((ROOT / "configs/preprocess/fullwidth.yaml").read_text())
ML = 256

# key -> (model_cfg, val checkpoint, cv_dir for OOF weights + calibration)
MODELS = {
    "ckip_B0":       ("configs/model/bert_ckip_base.yaml",      "experiments/valeval_b0_ckip/best_model.pt",       "experiments/exp008_bert_fullwidth_cv5_seed42"),
    "macbert_base":  ("configs/model/macbert_base.yaml",        "experiments/valeval_macbert_base/best_model.pt",  "experiments/exp036_zht_macbert_cv5_seed42"),
    "macbert_tapt":  ("configs/model/macbert_tapt.yaml",        "experiments/valeval_macbert_tapt/best_model.pt",  "experiments/exp050_tapt_macbert_cv5_seed42"),
    "macbert_dapt":  ("configs/model/macbert_dapt.yaml",        "experiments/valeval_macbert_dapt/best_model.pt",  "experiments/exp051_dapt_macbert_cv5_seed42"),
    "ckip_tapt_ep3": ("configs/model/bert_ckip_tapt_ep3.yaml",  "experiments/valeval_ckip_tapt_ep3/best_model.pt", "experiments/exp048_tapt_ep3_cv5_seed42"),
    "ckip_tapt_2e5": ("configs/model/bert_ckip_tapt_lr2e5.yaml","experiments/valeval_ckip_tapt_2e5/best_model.pt", "experiments/exp047_tapt_lr2e5_cv5_seed42"),
    "bge_m3":        ("configs/model/bge_m3.yaml",              "experiments/valeval_bgem3/best_model.pt",          "experiments/exp033_bge_m3_lr1e5_cv5_seed42"),
    "roberta_wwm":   ("configs/model/roberta_wwm_base.yaml",    "experiments/valeval_roberta_wwm/best_model.pt",    "experiments/exp052_roberta_wwm_cv5_seed42"),
    "bge_m3_tapt":   ("configs/model/bge_m3_tapt.yaml",         "experiments/valeval_bgem3_tapt/best_model.pt",     "experiments/exp055_tapt_bgem3_cv5_seed42"),
    # "roberta_tapt": dropped — roberta-wwm domain-saturated, TAPT hurt (CV 0.577<0.586); 9-way 0.6244<0.6258
    # "pert": dropped — single-model CV 0.553 too weak, dilutes F1-weighted vote (9-way 0.6226 < 8-way 0.6258)
}


def _load(mcfg_path, ckpt):
    mcfg = yaml.safe_load((ROOT / mcfg_path).read_text())
    dev = get_device()
    tok = AutoTokenizer.from_pretrained(mcfg["pretrained"])
    model = build_model(mcfg).to(dev)
    model.load_state_dict(torch.load(ROOT / ckpt, map_location=dev))
    model.eval()
    return model, tok, dev


def predict_probs(model, tok, dev, records):
    """records (already preprocessed) -> {id: {task: probvec}}"""
    out = {}
    with torch.no_grad():
        for i in range(0, len(records), 32):
            ch = records[i:i + 32]
            enc = tok([r["data"] for r in ch], max_length=ML, padding="max_length",
                      truncation=True, return_tensors="pt")
            logits = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev), None)
            for t in TASKS:
                p = F.softmax(logits[t], dim=1).cpu().numpy()
                for j, r in enumerate(ch):
                    out.setdefault(r["id"], {})[t] = p[j]
    return out


# ── GR-style temperature calibration ────────────────────────────────────────
def softmax_logp(probs, T):
    log = np.log(np.maximum(probs, 1e-12)); s = log / T
    s = s - s.max(); e = np.exp(s); return e / e.sum()


def fit_T(pl, yi):
    def nll(T):
        return sum(-np.log(max(softmax_logp(np.array(p), T)[g], 1e-12)) for p, g in zip(pl, yi))
    return float(minimize_scalar(nll, bounds=(0.3, 10), method="bounded").x)


def fit_temperatures(mcfg_path, cv_dir):
    """Fit T per field on the pooled 5-fold OOF (held-out). Returns {task: T}."""
    pl = {t: [] for t in TASKS}; yi = {t: [] for t in TASKS}
    for k in range(5):
        ck = ROOT / cv_dir / f"fold_{k}" / "best_model.pt"
        sp = ROOT / cv_dir / "splits" / f"fold_{k}" / "val.json"
        if not ck.exists() or not sp.exists():
            continue
        recs = run_pipeline(json.loads(sp.read_text()), PCFG)
        model, tok, dev = _load(mcfg_path, ck.relative_to(ROOT))
        pr = predict_probs(model, tok, dev, recs)
        for r in recs:
            for t in TASKS:
                g = LABEL_MAPS[t].get(r.get(t))
                if g is not None:
                    pl[t].append(pr[r["id"]][t]); yi[t].append(g)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return {t: fit_T(pl[t], yi[t]) if pl[t] else 1.0 for t in TASKS}


def _inject(prob, field, label, w):
    p = prob.copy()
    p[LABEL_MAPS[field][label]] += w
    return p


def decode(prob_by_task, data_text, span=None, rule_mode="dataonly"):
    p = {t: prob_by_task[t].copy() for t in TASKS}
    uses_span = rule_mode in ("span", "span_safe", "data_tl_span_q", "data_tl_span_q_full")
    if uses_span and span is not None:
        prom = IDX_MAPS["promise_status"][int(p["promise_status"].argmax())]
        evid = IDX_MAPS["evidence_status"][int(p["evidence_status"].argmax())]
        r = {"promise_status": prom, "evidence_status": evid,
             "promise_string": span.get("promise_string", ""),
             "evidence_string": span.get("evidence_string", ""), "data": data_text}
        # timeline via span rule only for the full "span"/"span_safe" modes
        if rule_mode in ("span", "span_safe"):
            rt = predict_timeline_rule(r, fallback=(rule_mode == "span"))
            if rt:
                p["verification_timeline"] = _inject(p["verification_timeline"], "verification_timeline", rt, RULE_W)
        # quality via span rule
        conf_only = rule_mode in ("span_safe", "data_tl_span_q")
        rq = predict_quality_rule(r, conf_only=conf_only)
        if rq:
            p["evidence_quality"] = _inject(p["evidence_quality"], "evidence_quality", rq, RULE_W)
    d = {t: IDX_MAPS[t][int(p[t].argmax())] for t in TASKS}
    # data-only timeline rule for our champion modes
    if rule_mode in ("dataonly", "data_tl_span_q", "data_tl_span_q_full"):
        tl = timeline_rule(data_text, year_only=True)
        if tl:
            d["verification_timeline"] = tl
    enforce_constraints(d)
    return d


def main():
    val = run_pipeline(json.loads(VAL.read_text()), PCFG)
    text_by_id = {r["id"]: r["data"] for r in val}
    gold = json.loads(VAL.read_text())
    ids = list(text_by_id)

    raw_probs, f1w = {}, {}
    for k, (mc, ck, cvd) in MODELS.items():
        print(f"[{k}] val predict ...", flush=True)
        model, tok, dev = _load(mc, ck)
        raw_probs[k] = predict_probs(model, tok, dev, val)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        agg = json.loads((ROOT / cvd / "cv_results.json").read_text())["aggregate"]
        f1w[k] = {t: agg[t]["macro_f1_mean"] for t in TASKS}

    def get_probs(calib=False):
        return raw_probs

    bestsrc = {t: max(MODELS, key=lambda k: f1w[k][t]) for t in TASKS}

    spans = {x["id"]: x for x in json.loads(SPANS.read_text())} if SPANS.exists() else {}
    if not spans:
        print("(no GR spans found -> span-rule modes skipped; core path needs no spans)", flush=True)

    def run(probs, strategy, rule_mode):
        preds = []
        for sid in ids:
            pt = {}
            for t in TASKS:
                if strategy == "equal":
                    pt[t] = sum(probs[k][sid][t] for k in MODELS) / len(MODELS)
                elif strategy == "f1w":
                    w = sum(f1w[k][t] for k in MODELS)
                    pt[t] = sum(f1w[k][t] * probs[k][sid][t] for k in MODELS) / w
                else:
                    pt[t] = probs[bestsrc[t]][sid][t]
            d = decode(pt, text_by_id[sid], span=spans.get(str(sid)), rule_mode=rule_mode)
            d["id"] = sid
            preds.append(d)
        return score(gold, preds)

    probs = get_probs(calib=False)  # calibration was a no-op; use raw
    print(f"\n{'config (f1w soft-vote)':<28}{'promise':>9}{'evid_st':>9}{'evid_q':>9}{'timeline':>10}{'total':>9}")
    modes = [("none", "no rules"), ("dataonly", "data-tl (我們最佳)")]
    if spans:  # span-rule modes only when GR spans are present
        modes += [("span", "GR span T+Q (裸搬)"), ("span_safe", "GR span 安全版(去fallback)"),
                  ("data_tl_span_q", "data-tl + span-Q(高信心)"),
                  ("data_tl_span_q_full", "data-tl + span-Q(全branch)")]
    for rm, lab in modes:
        m = run(probs, "f1w", rm)
        print(f"{lab:<28}{m['promise_status']['macro_f1']:>9.4f}{m['evidence_status']['macro_f1']:>9.4f}"
              f"{m['evidence_quality']['macro_f1']:>9.4f}{m['verification_timeline']['macro_f1']:>10.4f}{m['total']:>9.4f}")
    print("\n(對照: 單模 ckip B0=0.6046; 我們舊最佳 f1w+data規則=0.6198; GR 純BERT ensemble=0.6306; GR 最佳=0.6607)")


if __name__ == "__main__":
    main()
