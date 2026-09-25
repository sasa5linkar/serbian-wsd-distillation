from __future__ import annotations

import csv
import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .evaluate import SentenceTransformerRanker, compute_metrics, evaluate_gold_split, write_errors, write_metrics, write_predictions
from .presets import LEXISENSE_PRESETS, LexiSensePreset
from .sense_repo import SenseDefinition
from .train import train_sentence_transformer


@dataclass(frozen=True, slots=True)
class EvalSplit:
    name: str
    out_dir: Path


@dataclass(frozen=True, slots=True)
class ComparisonJob:
    preset: LexiSensePreset
    pairs_path: Path
    gold_dir: Path
    sense_repo: Path
    model_dir: Path
    outputs_dir: Path
    eval_splits: tuple[EvalSplit, ...]
    epochs: int
    batch_size: int
    warmup_steps: int
    text_prefix: str
    normalize_embeddings: bool

    def describe(self) -> str:
        split_parts = " && ".join(f"evaluate {self.preset.name} {split.name}" for split in self.eval_splits)
        return (
            f"train {self.preset.name} {self.preset.model_name} "
            f"--batch-size {self.batch_size} --epochs {self.epochs} && {split_parts}"
        )


SUMMARY_COLUMNS = [
    "preset",
    "model_name",
    "split",
    "coverage",
    "top1_accuracy",
    "top3_accuracy",
    "mrr",
    "total_examples",
    "covered_examples",
    "train_runtime_seconds",
]


def build_comparison_plan(
    *,
    pairs_path: Path | str,
    gold_dir: Path | str,
    sense_repo: Path | str,
    models_dir: Path | str,
    outputs_dir: Path | str,
    epochs: int,
    warmup_steps: int,
    presets: Mapping[str, LexiSensePreset] = LEXISENSE_PRESETS,
) -> list[ComparisonJob]:
    pairs_path = Path(pairs_path)
    gold_dir = Path(gold_dir)
    sense_repo = Path(sense_repo)
    models_dir = Path(models_dir)
    outputs_dir = Path(outputs_dir)
    jobs: list[ComparisonJob] = []
    for preset in presets.values():
        preset_out = outputs_dir / preset.name
        jobs.append(
            ComparisonJob(
                preset=preset,
                pairs_path=pairs_path,
                gold_dir=gold_dir,
                sense_repo=sense_repo,
                model_dir=models_dir / preset.output_name,
                outputs_dir=preset_out,
                eval_splits=(
                    EvalSplit("dev", preset_out / "dev_eval"),
                    EvalSplit("final", preset_out / "final_eval"),
                ),
                epochs=epochs,
                batch_size=preset.batch_size,
                warmup_steps=warmup_steps,
                text_prefix=preset.text_prefix,
                normalize_embeddings=preset.normalize_embeddings,
            )
        )
    return jobs


def run_comparison_job(
    job: ComparisonJob,
    sense_definitions: dict[str, SenseDefinition],
    *,
    retry_mling_oom: bool = True,
) -> dict[str, object]:
    started = time.time()
    batch_size = job.batch_size
    try:
        train_sentence_transformer(
            job.pairs_path,
            job.model_dir,
            model_name=job.preset.model_name,
            epochs=job.epochs,
            batch_size=batch_size,
            warmup_steps=job.warmup_steps,
            text_prefix=job.text_prefix,
        )
    except RuntimeError as exc:
        if not (retry_mling_oom and job.preset.name == "mling" and "out of memory" in str(exc).lower()):
            raise
        job.outputs_dir.mkdir(parents=True, exist_ok=True)
        (job.outputs_dir / "train_failed_batch4.log").write_text(
            "".join(traceback.format_exception(exc)),
            encoding="utf-8",
        )
        batch_size = 2
        train_sentence_transformer(
            job.pairs_path,
            job.model_dir,
            model_name=job.preset.model_name,
            epochs=job.epochs,
            batch_size=batch_size,
            warmup_steps=job.warmup_steps,
            text_prefix=job.text_prefix,
        )

    runtime = round(time.time() - started, 3)
    job.outputs_dir.mkdir(parents=True, exist_ok=True)
    train_summary = {
        "preset": job.preset.name,
        "model_name": job.preset.model_name,
        "model_dir": str(job.model_dir),
        "epochs": job.epochs,
        "batch_size": batch_size,
        "warmup_steps": job.warmup_steps,
        "text_prefix": job.text_prefix,
        "normalize_embeddings": job.normalize_embeddings,
        "train_runtime_seconds": runtime,
    }
    (job.outputs_dir / "train_summary.json").write_text(
        json.dumps(train_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ranker = SentenceTransformerRanker(
        str(job.model_dir),
        text_prefix=job.text_prefix,
        normalize_embeddings=job.normalize_embeddings,
    )
    for split in job.eval_splits:
        predictions = evaluate_gold_split(job.gold_dir, split.name, sense_definitions, ranker)
        metrics = compute_metrics(predictions)
        write_metrics(metrics, split.out_dir / "metrics.json")
        write_predictions(predictions, split.out_dir / "predictions_gold.tsv")
        write_errors(predictions, split.out_dir / "errors_gold.tsv")

    return train_summary


def collect_summary_rows(
    outputs_dir: Path | str,
    presets: Mapping[str, LexiSensePreset] = LEXISENSE_PRESETS,
) -> list[dict[str, object]]:
    outputs_dir = Path(outputs_dir)
    rows: list[dict[str, object]] = []
    for preset in presets.values():
        train_summary_path = outputs_dir / preset.name / "train_summary.json"
        train_runtime = ""
        if train_summary_path.exists():
            train_runtime = json.loads(train_summary_path.read_text(encoding="utf-8")).get("train_runtime_seconds", "")
        for split in ("dev", "final"):
            metrics_path = outputs_dir / preset.name / f"{split}_eval" / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "preset": preset.name,
                    "model_name": preset.model_name,
                    "split": split,
                    "coverage": metrics.get("coverage", 0.0),
                    "top1_accuracy": metrics.get("top1_accuracy", 0.0),
                    "top3_accuracy": metrics.get("top3_accuracy", 0.0),
                    "mrr": metrics.get("mrr", 0.0),
                    "total_examples": metrics.get("total_examples", 0),
                    "covered_examples": metrics.get("covered_examples", 0),
                    "train_runtime_seconds": train_runtime,
                }
            )
    return rows


def write_summary_files(rows: list[dict[str, object]], tsv_path: Path | str, json_path: Path | str) -> None:
    tsv_path = Path(tsv_path)
    json_path = Path(json_path)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def plan_to_jsonable(plan: list[ComparisonJob]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for job in plan:
        payload.append(
            {
                "preset": asdict(job.preset),
                "pairs_path": str(job.pairs_path),
                "gold_dir": str(job.gold_dir),
                "sense_repo": str(job.sense_repo),
                "model_dir": str(job.model_dir),
                "outputs_dir": str(job.outputs_dir),
                "eval_splits": [{"name": split.name, "out_dir": str(split.out_dir)} for split in job.eval_splits],
                "epochs": job.epochs,
                "batch_size": job.batch_size,
                "warmup_steps": job.warmup_steps,
                "text_prefix": job.text_prefix,
                "normalize_embeddings": job.normalize_embeddings,
                "description": job.describe(),
            }
        )
    return payload
