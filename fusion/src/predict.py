"""
Unified prediction interface.
Each model class implements predict(records) -> list[dict].
Usage:
    python src/predict.py --config configs/experiment/exp001.yaml
"""

import json
import shutil
import argparse
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import yaml


# ── Base interface ────────────────────────────────────────────────────────────

class BaseModel(ABC):
    def __init__(self, model_cfg: dict):
        self.cfg = model_cfg

    @abstractmethod
    def predict(self, records: list[dict]) -> list[dict]:
        """
        Args:
            records: list of input dicts (must contain at least 'id' and 'data')
        Returns:
            list of dicts with 'id' + all 6 output fields:
                promise_status, promise_string,
                verification_timeline,
                evidence_status, evidence_string, evidence_quality
        """
        ...


# ── Model registry ────────────────────────────────────────────────────────────

def load_model(model_cfg: dict) -> BaseModel:
    model_type = model_cfg["type"]
    if model_type == "llm_api":
        from src.models.llm_api import LLMApiModel
        return LLMApiModel(model_cfg)
    if model_type == "ollama":
        from src.models.ollama_model import OllamaModel
        return OllamaModel(model_cfg)
    raise ValueError(f"Unknown model type: {model_type}")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def load_data(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_preprocess(records: list[dict], preprocess_cfg: dict) -> list[dict]:
    from src.preprocess.pipeline import run_pipeline
    return run_pipeline(records, preprocess_cfg)


def save_experiment(output_dir: Path, exp_cfg: dict, predictions: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "config.yaml").write_text(
        yaml.dump(exp_cfg, allow_unicode=True), encoding="utf-8"
    )


def load_partial(partial_jsonl: Path):
    """Load already-completed predictions from a partial JSONL file."""
    results = []
    for line in partial_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            results.append(json.loads(line))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True,      help="Experiment config YAML")
    parser.add_argument("--dry-run",   action="store_true", help="Load config only, skip inference")
    parser.add_argument("--no-resume", action="store_true", help="Discard partial results and start fresh")
    args = parser.parse_args()

    exp_cfg   = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pre_cfg   = yaml.safe_load(Path(exp_cfg["preprocess"]).read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(Path(exp_cfg["model"]).read_text(encoding="utf-8"))

    # Inject prompt text into model_cfg so the model class stays prompt-agnostic
    if "prompt" in exp_cfg:
        prompt_path = Path(exp_cfg["prompt"])
        model_cfg["system_prompt"] = prompt_path.read_text(encoding="utf-8")
        model_cfg["prompt_file"]   = str(prompt_path)   # for snapshot

    # Partial results live in a fixed .partial/ subdir so re-runs auto-resume
    partial_dir  = Path(exp_cfg["output_dir"]) / ".partial"
    partial_jsonl = partial_dir / "predictions.jsonl"

    print(f"Experiment : {exp_cfg['name']}")
    print(f"Preprocess : {pre_cfg['name']}")
    print(f"Model      : {model_cfg['name']}")
    print(f"Prompt     : {exp_cfg.get('prompt', '(embedded)')}")

    if args.dry_run:
        print("Dry run — exiting.")
        return

    data_cfg  = exp_cfg["data"]
    data_path = data_cfg.get("val") or data_cfg.get("train")
    records   = load_data(data_path)
    records   = run_preprocess(records, pre_cfg)

    # ── Resume handling ───────────────────────────────────────────────────────
    existing_results = []
    done_ids = set()

    if partial_jsonl.exists() and not args.no_resume:
        existing_results = load_partial(partial_jsonl)
        done_ids = {r["id"] for r in existing_results}
        print(f"Resuming   : {len(done_ids)} / {len(records)} already done")
    elif partial_jsonl.exists() and args.no_resume:
        shutil.rmtree(partial_dir)
        print("Starting fresh (--no-resume)")

    partial_dir.mkdir(parents=True, exist_ok=True)

    remaining = [r for r in records if r["id"] not in done_ids]
    print(f"Remaining  : {len(remaining)} records to predict")
    print()

    # ── Inference — write each result immediately ─────────────────────────────
    if remaining:
        model = load_model(model_cfg)
        with open(partial_jsonl, "a", encoding="utf-8") as f:
            for result in model.predict_stream(remaining):
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                existing_results.append(result)

    # ── Finalise — move partial to timestamped run dir ────────────────────────
    all_results = sorted(existing_results, key=lambda r: r["id"])
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir  = Path(exp_cfg["output_dir"]) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "predictions.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "config.yaml").write_text(
        yaml.dump(exp_cfg, allow_unicode=True), encoding="utf-8"
    )
    shutil.rmtree(partial_dir)

    print(f"\nSaved {len(all_results)} predictions → {output_dir}/predictions.json")

    # Auto-evaluate against ground truth when labels are present
    if records and all(k in records[0] for k in ("promise_status", "evidence_status")):
        from src.evaluate import score, print_report
        metrics = score(records, all_results)
        print_report(metrics)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
