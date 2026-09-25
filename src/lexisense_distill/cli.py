from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import write_analysis_outputs
from .compare import build_comparison_plan, collect_summary_rows, plan_to_jsonable, run_comparison_job, write_summary_files
from .dataset import build_training_pairs_from_dirs, write_coverage_errors, write_jsonl
from .early_stop import (
    build_early_stopping_plan,
    collect_early_stopping_summary,
    plan_to_jsonable as early_stop_plan_to_jsonable,
    run_early_stopping_job,
    write_early_stopping_summary,
    write_early_stopping_workbooks,
)
from .evaluate import (
    SentenceTransformerRanker,
    compute_metrics,
    evaluate_gold_split,
    evaluate_raw_round2_workbook,
    write_errors,
    write_metrics,
    write_predictions,
)
from .presets import LEXISENSE_PRESETS, preset_names
from .sense_repo import load_sense_definitions
from .train import DEFAULT_MODEL_NAME, train_sentence_transformer


def build_dataset_command(args: argparse.Namespace) -> int:
    sense_definitions = load_sense_definitions(args.sense_repo)
    result = build_training_pairs_from_dirs(args.silver_dir, args.gold_dir, sense_definitions)
    write_jsonl(result.pairs, args.out)
    if args.coverage_out:
        write_coverage_errors(result.coverage_errors, args.coverage_out)
    print(json.dumps({"stats": result.stats, "coverage_errors": len(result.coverage_errors)}, ensure_ascii=False, indent=2))
    return 0


def train_command(args: argparse.Namespace) -> int:
    train_sentence_transformer(
        args.pairs,
        args.out,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        text_prefix=args.text_prefix,
        seed=args.seed,
    )
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    sense_definitions = load_sense_definitions(args.sense_repo)
    ranker = SentenceTransformerRanker(
        args.model,
        text_prefix=args.text_prefix,
        normalize_embeddings=args.normalize_embeddings,
    )
    predictions = evaluate_gold_split(args.gold_dir, args.split, sense_definitions, ranker)
    metrics = compute_metrics(predictions)
    out_dir = Path(args.out)
    write_metrics(metrics, out_dir / "metrics.json")
    write_predictions(predictions, out_dir / "predictions_gold.tsv")
    write_errors(predictions, out_dir / "errors_gold.tsv")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def evaluate_raw_round2_command(args: argparse.Namespace) -> int:
    sense_definitions = load_sense_definitions(args.sense_repo)
    ranker = SentenceTransformerRanker(
        args.model,
        text_prefix=args.text_prefix,
        normalize_embeddings=args.normalize_embeddings,
    )
    predictions = evaluate_raw_round2_workbook(args.raw_round2_xlsx, args.gold_dir, sense_definitions, ranker)
    metrics = compute_metrics(predictions)
    out_dir = Path(args.out)
    write_metrics(metrics, out_dir / "metrics.json")
    write_predictions(predictions, out_dir / "predictions_gold.tsv")
    write_errors(predictions, out_dir / "errors_gold.tsv")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def run_all_command(args: argparse.Namespace) -> int:
    build_out = Path(args.work_dir) / "data" / "processed" / "silver_pairs.jsonl"
    coverage_out = Path(args.work_dir) / "outputs" / "coverage_errors.jsonl"
    model_out = Path(args.work_dir) / "models" / "wsd-distilled"
    dev_out = Path(args.work_dir) / "outputs" / "dev_eval"
    final_out = Path(args.work_dir) / "outputs" / "final_eval"
    run_config = Path(args.work_dir) / "outputs" / "run_config.json"

    build_dataset_command(
        argparse.Namespace(
            silver_dir=args.silver_dir,
            gold_dir=args.gold_dir,
            sense_repo=args.sense_repo,
            out=build_out,
            coverage_out=coverage_out,
        )
    )
    train_sentence_transformer(
        build_out,
        model_out,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        text_prefix=args.text_prefix,
        seed=args.seed,
    )
    for split, out_dir in (("dev", dev_out), ("final", final_out)):
        evaluate_command(
            argparse.Namespace(
                gold_dir=args.gold_dir,
                split=split,
                sense_repo=args.sense_repo,
                model=model_out,
                out=out_dir,
                text_prefix=args.text_prefix,
                normalize_embeddings=args.normalize_embeddings,
            )
        )
    run_config.parent.mkdir(parents=True, exist_ok=True)
    run_config.write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return 0


def compare_presets_command(args: argparse.Namespace) -> int:
    selected_presets = {name: LEXISENSE_PRESETS[name] for name in args.preset}
    plan = build_comparison_plan(
        pairs_path=args.pairs,
        gold_dir=args.gold_dir,
        sense_repo=args.sense_repo,
        models_dir=args.models_dir,
        outputs_dir=args.outputs_dir,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        presets=selected_presets,
    )
    if args.dry_run:
        print(json.dumps(plan_to_jsonable(plan), ensure_ascii=False, indent=2))
        return 0

    sense_definitions = load_sense_definitions(args.sense_repo)
    for job in plan:
        print(json.dumps({"running": job.describe()}, ensure_ascii=False), flush=True)
        run_comparison_job(job, sense_definitions)
    rows = collect_summary_rows(args.outputs_dir, selected_presets)
    write_summary_files(
        rows,
        Path(args.outputs_dir) / "summary.tsv",
        Path(args.outputs_dir) / "summary.json",
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def analyze_comparison_command(args: argparse.Namespace) -> int:
    summary = write_analysis_outputs(args.comparison_dir, args.gold_dir, args.out, raw_round2_xlsx=args.raw_round2_xlsx)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def early_stop_presets_command(args: argparse.Namespace) -> int:
    selected_presets = {name: LEXISENSE_PRESETS[name] for name in args.preset}
    plan = build_early_stopping_plan(
        pairs_path=args.pairs,
        gold_dir=args.gold_dir,
        sense_repo=args.sense_repo,
        models_dir=args.models_dir,
        checkpoints_dir=args.checkpoints_dir,
        outputs_dir=args.outputs_dir,
        max_epochs=args.max_epochs,
        patience=args.patience,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        presets=selected_presets,
    )
    if args.dry_run:
        print(json.dumps(early_stop_plan_to_jsonable(plan), ensure_ascii=False, indent=2))
        return 0

    sense_definitions = load_sense_definitions(args.sense_repo)
    for job in plan:
        print(json.dumps({"running": job.describe()}, ensure_ascii=False), flush=True)
        run_early_stopping_job(job, sense_definitions)
    rows = collect_early_stopping_summary(args.outputs_dir, selected_presets)
    write_early_stopping_summary(
        rows,
        Path(args.outputs_dir) / "summary.tsv",
        Path(args.outputs_dir) / "summary.json",
    )
    write_early_stopping_workbooks(args.outputs_dir, rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lexisense-distill")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-dataset")
    build.add_argument("--silver-dir", required=True)
    build.add_argument("--gold-dir", required=True)
    build.add_argument("--sense-repo", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--coverage-out")
    build.set_defaults(func=build_dataset_command)

    train = subparsers.add_parser("train")
    train.add_argument("--pairs", required=True)
    train.add_argument("--dev-gold")
    train.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    train.add_argument("--out", required=True)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--warmup-steps", type=int, default=100)
    train.add_argument("--text-prefix", default="")
    train.add_argument("--seed", type=int)
    train.set_defaults(func=train_command)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--gold-dir", required=True)
    evaluate.add_argument("--split", choices=["dev", "final"], required=True)
    evaluate.add_argument("--sense-repo", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--text-prefix", default="")
    evaluate.add_argument("--normalize-embeddings", action="store_true")
    evaluate.set_defaults(func=evaluate_command)

    evaluate_raw = subparsers.add_parser("evaluate-raw-round2")
    evaluate_raw.add_argument("--raw-round2-xlsx", required=True)
    evaluate_raw.add_argument("--gold-dir", required=True)
    evaluate_raw.add_argument("--sense-repo", required=True)
    evaluate_raw.add_argument("--model", required=True)
    evaluate_raw.add_argument("--out", required=True)
    evaluate_raw.add_argument("--text-prefix", default="")
    evaluate_raw.add_argument("--normalize-embeddings", action="store_true")
    evaluate_raw.set_defaults(func=evaluate_raw_round2_command)

    run_all = subparsers.add_parser("run-all")
    run_all.add_argument("--silver-dir", required=True)
    run_all.add_argument("--gold-dir", required=True)
    run_all.add_argument("--sense-repo", required=True)
    run_all.add_argument("--work-dir", default=".")
    run_all.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    run_all.add_argument("--epochs", type=int, default=1)
    run_all.add_argument("--batch-size", type=int, default=16)
    run_all.add_argument("--warmup-steps", type=int, default=100)
    run_all.add_argument("--text-prefix", default="")
    run_all.add_argument("--seed", type=int)
    run_all.add_argument("--normalize-embeddings", action="store_true")
    run_all.set_defaults(func=run_all_command)

    compare = subparsers.add_parser("compare-presets")
    compare.add_argument("--pairs", default=str(Path("data") / "processed" / "silver_pairs.jsonl"))
    compare.add_argument("--gold-dir", required=True)
    compare.add_argument("--sense-repo", required=True)
    compare.add_argument("--models-dir", default="models")
    compare.add_argument("--outputs-dir", default=str(Path("outputs") / "comparison"))
    compare.add_argument("--epochs", type=int, default=1)
    compare.add_argument("--warmup-steps", type=int, default=0)
    compare.add_argument("--preset", action="append", choices=preset_names(), default=None)
    compare.add_argument("--dry-run", action="store_true")
    compare.set_defaults(func=compare_presets_command)

    analyze = subparsers.add_parser("analyze-comparison")
    analyze.add_argument("--comparison-dir", default=str(Path("outputs") / "comparison"))
    analyze.add_argument("--gold-dir", required=True)
    analyze.add_argument("--out", default=str(Path("outputs") / "analysis"))
    analyze.add_argument("--raw-round2-xlsx")
    analyze.set_defaults(func=analyze_comparison_command)

    early_stop = subparsers.add_parser("early-stop-presets")
    early_stop.add_argument("--pairs", default=str(Path("data") / "processed" / "silver_pairs.jsonl"))
    early_stop.add_argument("--gold-dir", required=True)
    early_stop.add_argument("--sense-repo", required=True)
    early_stop.add_argument("--models-dir", default="models")
    early_stop.add_argument("--checkpoints-dir", default=str(Path("models") / "checkpoints"))
    early_stop.add_argument("--outputs-dir", default=str(Path("outputs") / "early_stopping"))
    early_stop.add_argument("--max-epochs", type=int, default=10)
    early_stop.add_argument("--patience", type=int, default=2)
    early_stop.add_argument("--warmup-steps", type=int, default=0)
    early_stop.add_argument("--seed", type=int)
    early_stop.add_argument("--preset", action="append", choices=preset_names(), default=None)
    early_stop.add_argument("--dry-run", action="store_true")
    early_stop.set_defaults(func=early_stop_presets_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "preset", None) is None:
        args.preset = preset_names()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
