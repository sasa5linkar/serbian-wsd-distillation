from __future__ import annotations

import json
from pathlib import Path

from lexisense_distill.compare import (
    build_comparison_plan,
    collect_summary_rows,
    write_summary_files,
)
from lexisense_distill.presets import LEXISENSE_PRESETS


def test_build_comparison_plan_includes_three_presets_and_splits(tmp_path: Path) -> None:
    plan = build_comparison_plan(
        pairs_path=tmp_path / "pairs.jsonl",
        gold_dir=tmp_path / "gold",
        sense_repo=tmp_path / "repo.xlsx",
        models_dir=tmp_path / "models",
        outputs_dir=tmp_path / "outputs",
        epochs=1,
        warmup_steps=0,
    )

    assert [job.preset.name for job in plan] == ["simple", "tesla", "mling"]
    assert plan[0].model_dir == tmp_path / "models" / "wsd-distilled-simple"
    assert plan[1].batch_size == 16
    assert plan[2].text_prefix == "query: "
    assert plan[2].normalize_embeddings is True
    assert [split.name for split in plan[0].eval_splits] == ["dev", "final"]


def test_dry_run_plan_is_command_like_and_does_not_need_outputs(tmp_path: Path) -> None:
    plan = build_comparison_plan(
        pairs_path=tmp_path / "pairs.jsonl",
        gold_dir=tmp_path / "gold",
        sense_repo=tmp_path / "repo.xlsx",
        models_dir=tmp_path / "models",
        outputs_dir=tmp_path / "outputs",
        epochs=1,
        warmup_steps=0,
    )

    commands = [job.describe() for job in plan]

    assert "train simple all-MiniLM-L6-v2" in commands[0]
    assert "evaluate simple dev" in commands[0]
    assert "train tesla te-sla/TeslaXLM" in commands[1]
    assert "train mling intfloat/multilingual-e5-large" in commands[2]


def test_collect_summary_rows_and_write_summary_files(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    for preset_name, split, top1 in [
        ("simple", "dev", 0.5),
        ("simple", "final", 0.4),
        ("mling", "dev", 0.6),
    ]:
        metrics_dir = outputs_dir / preset_name / f"{split}_eval"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "coverage": 1.0,
                    "top1_accuracy": top1,
                    "top3_accuracy": 0.8,
                    "mrr": 0.7,
                    "total_examples": 10,
                    "covered_examples": 10,
                }
            ),
            encoding="utf-8",
        )
        (metrics_dir.parent / "train_summary.json").write_text(
            json.dumps({"train_runtime_seconds": 12.3}),
            encoding="utf-8",
        )

    rows = collect_summary_rows(outputs_dir, LEXISENSE_PRESETS)

    assert rows[0]["preset"] == "simple"
    assert rows[0]["split"] == "dev"
    assert rows[0]["train_runtime_seconds"] == 12.3
    assert rows[2]["preset"] == "mling"

    write_summary_files(rows, outputs_dir / "summary.tsv", outputs_dir / "summary.json")
    assert "preset\tmodel_name\tsplit" in (outputs_dir / "summary.tsv").read_text(encoding="utf-8")
    assert json.loads((outputs_dir / "summary.json").read_text(encoding="utf-8"))[0]["preset"] == "simple"
