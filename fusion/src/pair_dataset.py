"""Pair verifier datasets and official-label mapping utilities."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset

from src.preprocess.span_align import _MULTI_SPAN_DELIM, normalize_fullwidth


PAIR_LABEL_MAP = {
    "support_clear": 0,
    "support_weak": 1,
    "no_support": 2,
}
PAIR_IDX_MAP = {idx: label for label, idx in PAIR_LABEL_MAP.items()}
PAIR_LABELS = [PAIR_IDX_MAP[idx] for idx in range(len(PAIR_IDX_MAP))]

PAIR_TO_OFFICIAL = {
    "support_clear": ("Yes", "Clear"),
    "support_weak": ("Yes", "Not Clear"),
    "no_support": ("No", "N/A"),
}


@dataclass(frozen=True)
class PairExample:
    id: str
    record_id: int | str
    promise_record_id: int | str
    evidence_record_id: int | str
    promise_text: str
    evidence_text: str
    label: str | None
    label_id: int
    is_synthetic_negative: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "record_id": self.record_id,
            "promise_record_id": self.promise_record_id,
            "evidence_record_id": self.evidence_record_id,
            "promise_text": self.promise_text,
            "evidence_text": self.evidence_text,
            "label": self.label,
            "label_id": self.label_id,
            "is_synthetic_negative": self.is_synthetic_negative,
        }


def normalize_pair_text(text: str | None) -> str:
    """Normalize fullwidth Latin/digits and join multi-span annotations."""
    if not text:
        return ""
    parts = [
        part.strip()
        for part in normalize_fullwidth(text).split(_MULTI_SPAN_DELIM)
        if part.strip()
    ]
    return " ".join(parts)


def positive_pair_label(record: dict) -> str | None:
    if record.get("promise_status") != "Yes":
        return None
    if record.get("evidence_status") != "Yes":
        return None
    quality = record.get("evidence_quality")
    if quality == "Clear":
        return "support_clear"
    if quality == "Not Clear":
        return "support_weak"
    return None


def _positive_pair(record: dict, label: str) -> PairExample:
    rid = record["id"]
    return PairExample(
        id=f"{rid}:pos",
        record_id=rid,
        promise_record_id=rid,
        evidence_record_id=rid,
        promise_text=normalize_pair_text(record.get("promise_string")),
        evidence_text=normalize_pair_text(record.get("evidence_string")),
        label=label,
        label_id=PAIR_LABEL_MAP[label],
        is_synthetic_negative=False,
    )


def _data_direct_pair(record: dict, label: str) -> PairExample:
    rid = record["id"]
    return PairExample(
        id=f"{rid}:data",
        record_id=rid,
        promise_record_id=rid,
        evidence_record_id=rid,
        promise_text=normalize_fullwidth(record.get("data") or ""),
        evidence_text="",
        label=label,
        label_id=PAIR_LABEL_MAP[label],
        is_synthetic_negative=False,
    )


def build_positive_pair_examples(records: Iterable[dict]) -> list[PairExample]:
    examples: list[PairExample] = []
    for record in records:
        label = positive_pair_label(record)
        if label is None:
            continue
        pair = _positive_pair(record, label)
        if pair.promise_text and pair.evidence_text:
            examples.append(pair)
    return examples


def data_direct_pair_label(record: dict) -> str | None:
    if record.get("promise_status") != "Yes":
        return None
    if record.get("evidence_status") == "No":
        return "no_support"
    if record.get("evidence_status") != "Yes":
        return None
    return positive_pair_label(record)


def build_data_direct_pair_examples(records: Iterable[dict]) -> list[PairExample]:
    """Build data-only 3-way examples.

    The model input is the full ``data`` field only. Misleading records are
    excluded from the 3-way training target, matching exp026.
    """
    examples: list[PairExample] = []
    for record in records:
        label = data_direct_pair_label(record)
        if label is None:
            continue
        pair = _data_direct_pair(record, label)
        if pair.promise_text:
            examples.append(pair)
    return examples


def build_pair_examples(
    records: list[dict],
    negative_ratio: int = 1,
    seed: int = 42,
) -> list[PairExample]:
    """Build 3-way pair examples with cross-record synthetic negatives.

    Misleading records are intentionally excluded because exp026 trains a
    3-way verifier. The official scorer still keeps the Misleading class.
    """
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be >= 0")

    positives = build_positive_pair_examples(records)
    if negative_ratio == 0 or not positives:
        return positives
    if len(positives) < 2:
        raise ValueError("At least two positive pairs are required to create cross-record negatives")

    rng = random.Random(seed)
    examples = list(positives)
    evidence_pool = list(positives)

    for pos in positives:
        candidates = [
            candidate for candidate in evidence_pool
            if candidate.evidence_record_id != pos.promise_record_id
        ]
        if not candidates:
            raise ValueError(f"No negative evidence candidate for record id={pos.promise_record_id}")
        for neg_idx in range(negative_ratio):
            evidence = rng.choice(candidates)
            examples.append(PairExample(
                id=f"{pos.record_id}:neg:{neg_idx}",
                record_id=pos.record_id,
                promise_record_id=pos.promise_record_id,
                evidence_record_id=evidence.evidence_record_id,
                promise_text=pos.promise_text,
                evidence_text=evidence.evidence_text,
                label="no_support",
                label_id=PAIR_LABEL_MAP["no_support"],
                is_synthetic_negative=True,
            ))

    return examples


def build_official_data_direct_examples(records: Iterable[dict]) -> list[PairExample]:
    examples: list[PairExample] = []
    for record in records:
        if record.get("promise_status") != "Yes":
            continue
        label = data_direct_pair_label(record)
        rid = record["id"]
        text = normalize_fullwidth(record.get("data") or "")
        if not text:
            continue
        examples.append(PairExample(
            id=f"{rid}:official_data",
            record_id=rid,
            promise_record_id=rid,
            evidence_record_id=rid,
            promise_text=text,
            evidence_text="",
            label=label,
            label_id=PAIR_LABEL_MAP[label] if label is not None else -1,
            is_synthetic_negative=False,
        ))
    return examples


def build_official_pair_examples(records: Iterable[dict]) -> list[PairExample]:
    """Build gold promise/evidence inference pairs for official val records."""
    examples: list[PairExample] = []
    for record in records:
        if record.get("promise_status") != "Yes" or record.get("evidence_status") != "Yes":
            continue
        rid = record["id"]
        label = positive_pair_label(record)
        promise_text = normalize_pair_text(record.get("promise_string"))
        evidence_text = normalize_pair_text(record.get("evidence_string"))
        if not promise_text or not evidence_text:
            continue
        examples.append(PairExample(
            id=f"{rid}:official",
            record_id=rid,
            promise_record_id=rid,
            evidence_record_id=rid,
            promise_text=promise_text,
            evidence_text=evidence_text,
            label=label,
            label_id=PAIR_LABEL_MAP[label] if label is not None else -1,
            is_synthetic_negative=False,
        ))
    return examples


def pair_label_to_official(label: str) -> tuple[str, str]:
    if label not in PAIR_TO_OFFICIAL:
        raise ValueError(f"Unknown pair label: {label}")
    return PAIR_TO_OFFICIAL[label]


def make_official_predictions(
    records: list[dict],
    pair_predictions: dict[int | str, str],
) -> list[dict]:
    """Map pair predictions back to official labels.

    promise_status and verification_timeline are gold pass-through fields for
    this oracle experiment. Misleading is never emitted by this 3-way verifier.
    """
    predictions: list[dict] = []
    for record in records:
        pred = {
            "id": record["id"],
            "promise_status": record["promise_status"],
            "verification_timeline": record["verification_timeline"],
        }

        if record.get("promise_status") == "No":
            pred["evidence_status"] = "N/A"
            pred["evidence_quality"] = "N/A"
        elif record.get("evidence_status") == "No":
            pred["evidence_status"] = "No"
            pred["evidence_quality"] = "N/A"
        elif record.get("evidence_status") == "N/A":
            pred["evidence_status"] = "N/A"
            pred["evidence_quality"] = "N/A"
        else:
            pair_label = pair_predictions.get(record["id"], "no_support")
            evidence_status, evidence_quality = pair_label_to_official(pair_label)
            pred["evidence_status"] = evidence_status
            pred["evidence_quality"] = evidence_quality

        predictions.append(pred)

    return predictions


def make_official_data_direct_predictions(
    records: list[dict],
    pair_predictions: dict[int | str, str],
) -> list[dict]:
    """Map data-direct pair predictions back to official labels.

    Only promise_status and verification_timeline are gold pass-through. For
    promise_status=Yes records, evidence_status/evidence_quality come from the
    data-only classifier, including gold evidence_status=No records.
    """
    predictions: list[dict] = []
    for record in records:
        pred = {
            "id": record["id"],
            "promise_status": record["promise_status"],
            "verification_timeline": record["verification_timeline"],
        }

        if record.get("promise_status") == "No":
            pred["evidence_status"] = "N/A"
            pred["evidence_quality"] = "N/A"
        else:
            pair_label = pair_predictions.get(record["id"], "no_support")
            evidence_status, evidence_quality = pair_label_to_official(pair_label)
            pred["evidence_status"] = evidence_status
            pred["evidence_quality"] = evidence_quality

        predictions.append(pred)

    return predictions


def pair_label_counts(examples: Iterable[PairExample]) -> dict[str, int]:
    counts = Counter(example.label for example in examples)
    return {label: counts.get(label, 0) for label in PAIR_LABELS}


class PairVerifierDataset(Dataset):
    def __init__(
        self,
        examples: list[PairExample],
        tokenizer,
        max_length: int = 256,
        has_labels: bool = True,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        tok_kwargs = dict(
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if example.evidence_text:
            tok_kwargs["text_pair"] = example.evidence_text
        enc = self.tokenizer(example.promise_text, **tok_kwargs)
        item = {
            "id": example.id,
            "record_id": example.record_id,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        if self.has_labels:
            item["label"] = torch.tensor(example.label_id, dtype=torch.long)
        return item
