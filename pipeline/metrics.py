# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yayou (SHIH Ya-You) and Eason (WANG Yi-Hsin) — AI CUP 2026 / TEAM_10049
"""Macro-F1 + weighted score using competition weights.

Re-exports src.metrics.evaluate_hybrid and adds a thin wrapper that takes our
Prediction dataclass directly.
"""
from __future__ import annotations
from sklearn.metrics import f1_score, classification_report

from src.metrics import evaluate_hybrid  # noqa: F401
from pipeline.data.schema import Prediction, ESGSample, FIELD_WEIGHTS
from pipeline.data.normalize import normalize


def score(predictions: list[Prediction], gts: list[ESGSample]) -> dict:
    """Compute weighted macro-F1 + per-field detail."""
    by_id = {p.id: p for p in predictions}
    rows = [(by_id[g.id], g) for g in gts if g.id in by_id]
    out = {"per_field": {}, "weighted": 0.0, "n": len(rows)}
    for field in FIELD_WEIGHTS:
        y_true = [normalize(getattr(g, field)) for _, g in rows]
        y_pred = [normalize(getattr(p, f"pred_{field}")) for p, _ in rows]
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        out["per_field"][field] = {
            "macro_f1": f1,
            "weight": FIELD_WEIGHTS[field],
            "classification_report": classification_report(
                y_true, y_pred, zero_division=0, output_dict=True
            ),
        }
        out["weighted"] += f1 * FIELD_WEIGHTS[field]
    return out


def pretty_print(report: dict) -> None:
    print(f"\n{'='*60}\nMacro-F1 Report (n={report['n']})\n{'='*60}")
    for field, info in report["per_field"].items():
        print(f"  {field:<25}  F1={info['macro_f1']:.4f}  (w={info['weight']})")
    print(f"  {'WEIGHTED TOTAL':<25}  {report['weighted']:.4f}")
