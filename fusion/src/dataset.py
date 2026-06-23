"""ESG dataset for BERT fine-tuning."""

import torch
from torch.utils.data import Dataset

LABEL_MAPS = {
    "promise_status": {"Yes": 0, "No": 1},
    "evidence_status": {"Yes": 0, "No": 1, "N/A": 2},
    "evidence_quality": {"Clear": 0, "Not Clear": 1, "Misleading": 2, "N/A": 3},
    "verification_timeline": {
        "already": 0, "within_2_years": 1,
        "between_2_and_5_years": 2, "more_than_5_years": 3, "N/A": 4,
    },
}

IDX_MAPS = {task: {v: k for k, v in lmap.items()} for task, lmap in LABEL_MAPS.items()}

TASKS = list(LABEL_MAPS.keys())

# Merged evidence head (tasks 2+3 fused). Misleading is excluded by design.
MERGED_EVIDENCE_MAP = {"clear": 0, "weak": 1, "no_support": 2}
MERGED_EVIDENCE_IDX = {v: k for k, v in MERGED_EVIDENCE_MAP.items()}
MERGED_IGNORE_INDEX = -100  # CrossEntropyLoss default ignore_index -> gates the loss


def merged_evidence_label(record: dict) -> int:
    """Map (evidence_status, evidence_quality) -> merged 3-class label.

    Returns MERGED_IGNORE_INDEX for records the evidence head must not learn:
    - promise_status != Yes (evidence is N/A by rule, recovered at inference)
    - evidence_quality == Misleading (single sample, excluded by design)
    """
    if record.get("promise_status") != "Yes":
        return MERGED_IGNORE_INDEX
    if record.get("evidence_status") == "No":
        return MERGED_EVIDENCE_MAP["no_support"]
    eq = record.get("evidence_quality")
    if eq == "Clear":
        return MERGED_EVIDENCE_MAP["clear"]
    if eq == "Not Clear":
        return MERGED_EVIDENCE_MAP["weak"]
    return MERGED_IGNORE_INDEX  # Misleading / anomalies

SPAN_IGNORE_INDEX = -100
SPAN_LABEL_MAP = {"O": 0, "B": 1, "I": 2}
SPAN_IDX_MAP = {v: k for k, v in SPAN_LABEL_MAP.items()}


class ESGDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int = 256,
        has_labels: bool = True,
        return_spans: bool = False,
        use_hints: bool = False,
        return_qa_spans: bool = False,
        span_label_scheme: str = "binary",
        merged_evidence: bool = False,
        evidence_na_as_no: bool = False,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels
        self.return_spans = return_spans
        self.use_hints = use_hints
        self.return_qa_spans = return_qa_spans
        self.merged_evidence = merged_evidence
        self.evidence_na_as_no = evidence_na_as_no
        if span_label_scheme not in {"binary", "bio"}:
            raise ValueError(f"Unknown span_label_scheme: {span_label_scheme}")
        self.span_label_scheme = span_label_scheme

    def __len__(self):
        return len(self.records)

    def _build_hint_text(self, record) -> tuple[str, str | None]:
        """Return (text_a, text_b) for tokenizer. text_b is data; text_a is the hint."""
        from src.preprocess.span_align import normalize_fullwidth

        hints = []
        if record.get("promise_status") == "Yes":
            ps = normalize_fullwidth(record.get("promise_string") or "")
            ps = " ".join(p.strip() for p in ps.split("｜") if p.strip())
            if ps:
                hints.append(ps)
        if record.get("evidence_status") == "Yes":
            es = normalize_fullwidth(record.get("evidence_string") or "")
            es = " ".join(p.strip() for p in es.split("｜") if p.strip())
            if es:
                hints.append(es)

        if hints:
            return " ".join(hints), record["data"]
        return record["data"], None

    def __getitem__(self, idx):
        record = self.records[idx]
        tok_kwargs = dict(
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if self.return_spans:
            tok_kwargs["return_offsets_mapping"] = True

        if self.use_hints:
            text_a, text_b = self._build_hint_text(record)
            enc = self.tokenizer(text_a, text_pair=text_b, **tok_kwargs)
        else:
            enc = self.tokenizer(record["data"], **tok_kwargs)
        item = {
            "id":             record["id"],
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        if self.has_labels:
            for task, lmap in LABEL_MAPS.items():
                label = record[task]
                if task == "evidence_status" and self.evidence_na_as_no and label == "N/A":
                    label = "No"  # fold N/A into No for training; N/A recovered at inference via promise gate
                item[task] = torch.tensor(lmap[label], dtype=torch.long)
            if self.merged_evidence:
                item["merged_evidence"] = torch.tensor(
                    merged_evidence_label(record), dtype=torch.long
                )

        if self.return_spans and self.has_labels:
            offsets = enc["offset_mapping"].squeeze(0)
            item["promise_token_labels"]  = self._token_labels(
                record, "promise_status",  "promise_string",  offsets, self.span_label_scheme
            )
            item["evidence_token_labels"] = self._token_labels(
                record, "evidence_status", "evidence_string", offsets, self.span_label_scheme
            )

        if self.return_qa_spans and self.has_labels:
            if not self.return_spans:
                enc_off = self.tokenizer(
                    record["data"],
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                )
                offsets = enc_off["offset_mapping"].squeeze(0)
            item["promise_start"], item["promise_end"] = self._qa_span_labels(
                record, "promise_status", "promise_string", offsets
            )
            item["evidence_start"], item["evidence_end"] = self._qa_span_labels(
                record, "evidence_status", "evidence_string", offsets
            )

        return item

    @staticmethod
    def _is_real_token(tok_start: int, tok_end: int) -> bool:
        return not (tok_start == tok_end == 0)

    def _token_labels(self, record, status_field, string_field, offsets, scheme: str = "binary"):
        # Lazy import to avoid importing transformers stuff at module load time when unused
        from src.preprocess.span_align import find_spans

        labels = torch.full((self.max_length,), SPAN_IGNORE_INDEX, dtype=torch.long)

        if scheme == "bio" and record.get(status_field) != "Yes":
            for tok_idx, (start, end) in enumerate(offsets.tolist()):
                if self._is_real_token(start, end):
                    labels[tok_idx] = SPAN_LABEL_MAP["O"]
            return labels

        if scheme == "binary" and record.get(status_field) != "Yes":
            return labels  # only Yes samples contribute binary span supervision

        target = record.get(string_field) or ""
        spans  = find_spans(record["data"], target)
        if not spans:
            return labels  # alignment failed (annotation noise) → skip span loss

        if scheme == "bio":
            for tok_idx, (start, end) in enumerate(offsets.tolist()):
                if self._is_real_token(start, end):
                    labels[tok_idx] = SPAN_LABEL_MAP["O"]

            for span_start, span_end in sorted(spans):
                token_indices = [
                    tok_idx
                    for tok_idx, (start, end) in enumerate(offsets.tolist())
                    if self._is_real_token(start, end) and span_start < end and start < span_end
                ]
                if not token_indices:
                    continue
                labels[token_indices[0]] = SPAN_LABEL_MAP["B"]
                for tok_idx in token_indices[1:]:
                    labels[tok_idx] = SPAN_LABEL_MAP["I"]
            return labels

        for tok_idx, (start, end) in enumerate(offsets.tolist()):
            if not self._is_real_token(start, end):
                continue  # special token / padding → keep -100
            in_span = any(s < end and start < e for s, e in spans)
            labels[tok_idx] = 1 if in_span else 0

        return labels

    def _span_to_token_indices(self, offsets, char_start: int, char_end: int):
        start_tok = SPAN_IGNORE_INDEX
        end_tok   = SPAN_IGNORE_INDEX
        for i, (tok_start, tok_end) in enumerate(offsets.tolist()):
            if tok_start == tok_end == 0:
                continue
            if start_tok == SPAN_IGNORE_INDEX and tok_start >= char_start:
                start_tok = i
            if tok_end <= char_end:
                end_tok = i
        if start_tok == SPAN_IGNORE_INDEX or end_tok == SPAN_IGNORE_INDEX or start_tok > end_tok:
            return SPAN_IGNORE_INDEX, SPAN_IGNORE_INDEX
        return start_tok, end_tok

    def _qa_span_labels(self, record, status_field, string_field, offsets):
        from src.preprocess.span_align import find_spans
        ign = SPAN_IGNORE_INDEX
        if record.get(status_field) != "Yes":
            return torch.tensor(ign, dtype=torch.long), torch.tensor(ign, dtype=torch.long)
        target = record.get(string_field) or ""
        spans  = find_spans(record["data"], target)
        if not spans:
            return torch.tensor(ign, dtype=torch.long), torch.tensor(ign, dtype=torch.long)
        char_start, char_end = spans[0]  # use first span
        s, e = self._span_to_token_indices(offsets, char_start, char_end)
        return torch.tensor(s, dtype=torch.long), torch.tensor(e, dtype=torch.long)
