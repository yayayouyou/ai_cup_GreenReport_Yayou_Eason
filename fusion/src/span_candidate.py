"""BIO span candidate decoding and recall metrics for exp027A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from src.dataset import SPAN_LABEL_MAP
from src.preprocess.span_align import find_spans, normalize_fullwidth


@dataclass(frozen=True)
class SpanCandidate:
    text: str
    start: int
    end: int
    score: float
    rank: int = 0

    def to_dict(self, rank: int | None = None) -> dict:
        return {
            "rank": self.rank if rank is None else rank,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "score": round(float(self.score), 6),
        }


def _offset_tuple(offset) -> tuple[int, int]:
    if isinstance(offset, torch.Tensor):
        offset = offset.detach().cpu().tolist()
    return int(offset[0]), int(offset[1])


def _rank_candidates(candidates: list[SpanCandidate], top_k: int) -> list[SpanCandidate]:
    ordered = sorted(candidates, key=lambda c: (-c.score, c.start, c.end, c.text))[:top_k]
    return [
        SpanCandidate(
            text=candidate.text,
            start=candidate.start,
            end=candidate.end,
            score=candidate.score,
            rank=rank,
        )
        for rank, candidate in enumerate(ordered, start=1)
    ]


def decode_bio_candidates(
    logits: torch.Tensor,
    offsets: Sequence,
    text: str,
    top_k: int = 5,
) -> list[SpanCandidate]:
    """Decode BIO token logits into ranked character-span candidates.

    ``offsets`` and returned character positions refer to ``normalize_fullwidth(text)``.
    Special/padding offsets ``(0, 0)`` are ignored. A stray ``I`` starts a new span.
    """
    if top_k <= 0:
        return []

    norm_text = normalize_fullwidth(text)
    probs = torch.softmax(logits.detach().cpu(), dim=-1)
    pred_ids = probs.argmax(dim=-1).tolist()

    candidates: list[SpanCandidate] = []
    current_start: int | None = None
    current_end: int | None = None
    current_scores: list[float] = []

    def close_current() -> None:
        nonlocal current_start, current_end, current_scores
        if current_start is not None and current_end is not None and current_end > current_start:
            score = sum(current_scores) / max(len(current_scores), 1)
            candidates.append(SpanCandidate(
                text=norm_text[current_start:current_end],
                start=current_start,
                end=current_end,
                score=score,
            ))
        current_start = None
        current_end = None
        current_scores = []

    for idx, label_id in enumerate(pred_ids):
        start, end = _offset_tuple(offsets[idx])
        if start == end == 0:
            close_current()
            continue

        if label_id == SPAN_LABEL_MAP["B"]:
            close_current()
            current_start = start
            current_end = end
            current_scores = [float(1.0 - probs[idx, SPAN_LABEL_MAP["O"]].item())]
        elif label_id == SPAN_LABEL_MAP["I"]:
            if current_start is None:
                current_start = start
                current_scores = []
            current_end = end
            current_scores.append(float(1.0 - probs[idx, SPAN_LABEL_MAP["O"]].item()))
        else:
            close_current()

    close_current()
    return _rank_candidates(candidates, top_k)


def candidate_dicts(candidates: Iterable[SpanCandidate]) -> list[dict]:
    return [candidate.to_dict() for candidate in candidates]


def build_candidate_record(
    record: dict,
    promise_logits: torch.Tensor,
    evidence_logits: torch.Tensor,
    offsets: Sequence,
    top_k: int = 5,
) -> dict:
    return {
        "id": record["id"],
        "promise_candidates": candidate_dicts(
            decode_bio_candidates(promise_logits, offsets, record["data"], top_k=top_k)
        ),
        "evidence_candidates": candidate_dicts(
            decode_bio_candidates(evidence_logits, offsets, record["data"], top_k=top_k)
        ),
    }


def gold_subspans(record: dict, kind: str) -> list[tuple[int, int]]:
    if kind == "promise":
        status_field = "promise_status"
        string_field = "promise_string"
    elif kind == "evidence":
        status_field = "evidence_status"
        string_field = "evidence_string"
    else:
        raise ValueError(f"Unknown span kind: {kind}")

    if record.get(status_field) != "Yes":
        return []
    return find_spans(record.get("data", ""), record.get(string_field) or "")


def is_gold_hit(
    candidate: SpanCandidate | dict,
    gold_span: tuple[int, int],
    threshold: float = 0.8,
) -> bool:
    if isinstance(candidate, dict):
        cand_start = int(candidate["start"])
        cand_end = int(candidate["end"])
    else:
        cand_start = candidate.start
        cand_end = candidate.end

    gold_start, gold_end = gold_span
    gold_len = max(gold_end - gold_start, 0)
    if gold_len == 0:
        return False
    overlap = max(0, min(cand_end, gold_end) - max(cand_start, gold_start))
    return (overlap / gold_len) >= threshold


def _kind_metrics(
    records: list[dict],
    candidate_by_id: dict,
    kind: str,
    ks: tuple[int, ...],
    threshold: float,
) -> dict:
    field = f"{kind}_candidates"
    total_gold = 0
    records_with_gold = 0
    subspan_hits = {k: 0 for k in ks}
    record_any_hits = {k: 0 for k in ks}
    record_all_hits = {k: 0 for k in ks}

    for record in records:
        golds = gold_subspans(record, kind)
        if not golds:
            continue
        records_with_gold += 1
        total_gold += len(golds)
        candidates = candidate_by_id.get(record["id"], {}).get(field, [])

        for k in ks:
            top_candidates = candidates[:k]
            hits = [
                any(is_gold_hit(candidate, gold, threshold=threshold) for candidate in top_candidates)
                for gold in golds
            ]
            subspan_hits[k] += sum(1 for hit in hits if hit)
            if any(hits):
                record_any_hits[k] += 1
            if all(hits):
                record_all_hits[k] += 1

    metrics = {
        "records_with_gold": records_with_gold,
        "gold_subspans": total_gold,
    }
    for k in ks:
        metrics[f"subspan_recall@{k}"] = round(subspan_hits[k] / total_gold, 4) if total_gold else 0.0
        metrics[f"record_any_recall@{k}"] = round(record_any_hits[k] / records_with_gold, 4) if records_with_gold else 0.0
        metrics[f"record_all_recall@{k}"] = round(record_all_hits[k] / records_with_gold, 4) if records_with_gold else 0.0
    return metrics


def evaluate_candidate_records(
    records: list[dict],
    candidate_records: list[dict],
    ks: tuple[int, ...] = (1, 3, 5),
    threshold: float = 0.8,
) -> dict:
    candidate_by_id = {candidate_record["id"]: candidate_record for candidate_record in candidate_records}
    promise_metrics = _kind_metrics(records, candidate_by_id, "promise", ks, threshold)
    evidence_metrics = _kind_metrics(records, candidate_by_id, "evidence", ks, threshold)

    metrics = {
        "threshold": threshold,
        "promise": promise_metrics,
        "evidence": evidence_metrics,
    }
    for k in ks:
        metrics[f"mean_subspan_recall@{k}"] = round(
            (promise_metrics[f"subspan_recall@{k}"] + evidence_metrics[f"subspan_recall@{k}"]) / 2,
            4,
        )
    return metrics
