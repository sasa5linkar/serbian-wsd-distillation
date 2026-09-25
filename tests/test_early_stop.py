from __future__ import annotations

from pathlib import Path

from lexisense_distill.early_stop import (
    EarlyStoppingState,
    build_early_stopping_plan,
    epoch_seed,
    run_early_stopping_job,
    should_stop,
    update_early_stopping,
    write_early_stopping_workbooks,
)
from lexisense_distill.presets import LEXISENSE_PRESETS


def test_build_early_stopping_plan_uses_best_model_outputs_and_checkpoint_dirs(tmp_path: Path) -> None:
    plan = build_early_stopping_plan(
        pairs_path=tmp_path / "pairs.jsonl",
        gold_dir=tmp_path / "gold",
        sense_repo=tmp_path / "repo.xlsx",
        models_dir=tmp_path / "models",
        checkpoints_dir=tmp_path / "checkpoints",
        outputs_dir=tmp_path / "outputs",
        max_epochs=10,
        patience=2,
        warmup_steps=0,
        seed=20260515,
        presets={"simple": LEXISENSE_PRESETS["simple"]},
    )

    job = plan[0]

    assert job.best_model_dir == tmp_path / "models" / "wsd-distilled-simple-best"
    assert job.checkpoints_dir == tmp_path / "checkpoints" / "simple"
    assert job.outputs_dir == tmp_path / "outputs" / "simple"
    assert job.max_epochs == 10
    assert job.patience == 2
    assert job.seed == 20260515
    assert "early-stop simple" in job.describe()
    assert "--seed 20260515" in job.describe()


def test_epoch_seed_changes_each_epoch_from_recorded_base_seed() -> None:
    assert epoch_seed(None, 3) is None
    assert epoch_seed(20260515, 1) == 20260515
    assert epoch_seed(20260515, 3) == 20260517


def test_update_early_stopping_tracks_best_dev_accuracy() -> None:
    state = EarlyStoppingState()

    state = update_early_stopping(state, epoch=1, dev_accuracy=0.50)
    state = update_early_stopping(state, epoch=2, dev_accuracy=0.49)
    state = update_early_stopping(state, epoch=3, dev_accuracy=0.55)

    assert state.best_epoch == 3
    assert state.best_accuracy == 0.55
    assert state.epochs_without_improvement == 0


def test_should_stop_after_patience_without_improvement() -> None:
    state = EarlyStoppingState(best_epoch=1, best_accuracy=0.6, epochs_without_improvement=2)

    assert should_stop(state, patience=2) is True
    assert should_stop(state, patience=3) is False


def test_run_early_stopping_selects_on_dev_then_evaluates_final_once(tmp_path: Path) -> None:
    job = build_early_stopping_plan(
        pairs_path=tmp_path / "pairs.jsonl",
        gold_dir=tmp_path / "gold",
        sense_repo=tmp_path / "repo.xlsx",
        models_dir=tmp_path / "models",
        checkpoints_dir=tmp_path / "checkpoints",
        outputs_dir=tmp_path / "outputs",
        max_epochs=10,
        patience=2,
        warmup_steps=0,
        seed=1234,
        presets={"simple": LEXISENSE_PRESETS["simple"]},
    )[0]
    eval_calls: list[tuple[str, str]] = []

    def trainer(fake_job, epoch, checkpoint_dir, source_model):
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "model.safetensors").write_text(f"epoch {epoch}", encoding="utf-8")

    def evaluator(fake_job, model_dir, split, sense_definitions, out_dir):
        eval_calls.append((split, Path(model_dir).name))
        out_dir.mkdir(parents=True)
        accuracies = {"epoch-01": 0.50, "epoch-02": 0.55, "epoch-03": 0.54, "epoch-04": 0.53}
        return {
            "top1_accuracy": accuracies.get(Path(model_dir).name, 0.55),
            "top3_accuracy": 0.75,
            "mrr": 0.6,
            "coverage": 1.0,
        }

    summary = run_early_stopping_job(job, {}, trainer=trainer, evaluator=evaluator)

    assert summary["best_epoch"] == 2
    assert summary["seed"] == 1234
    assert [call[0] for call in eval_calls[:4]] == ["dev", "dev", "dev", "dev"]
    assert eval_calls[4:] == [("final", "wsd-distilled-simple-best"), ("dev", "wsd-distilled-simple-best")]
    assert (job.best_model_dir / "model.safetensors").read_text(encoding="utf-8") == "epoch 2"
    assert (job.outputs_dir / "epoch_metrics.tsv").exists()
    assert "epoch_seed" in (job.outputs_dir / "epoch_metrics.tsv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert (job.outputs_dir / "train_summary.json").exists()
    assert eval_calls[-1][0] == "dev"


def test_write_early_stopping_workbooks_creates_summary_and_curves(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    preset_dir = outputs_dir / "simple"
    preset_dir.mkdir(parents=True)
    (preset_dir / "epoch_metrics.tsv").write_text(
        "preset\tepoch\tdev_top1_accuracy\nsimple\t1\t0.5\n",
        encoding="utf-8-sig",
    )

    write_early_stopping_workbooks(
        outputs_dir,
        [{"preset": "simple", "best_epoch": 1, "final_accuracy": 0.6}],
    )

    assert (outputs_dir / "summary.xlsx").exists()
    assert (outputs_dir / "training_curves.xlsx").exists()
