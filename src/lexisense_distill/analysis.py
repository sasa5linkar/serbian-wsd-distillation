from __future__ import annotations

import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from .dataset import EVALUATION_EXCLUDED_MWE_TYPES, is_evaluation_single_token
from .presets import LEXISENSE_PRESETS
from .webanno import (
    WebAnnoSentence,
    WebAnnoToken,
    gold_files_for_split,
    is_blank,
    normalize_sense_id,
    parse_candidate_ids,
    parse_webanno_tsv,
)

EXCEL_SHEETS = (
    "Summary",
    "Raw Round2 Baselines",
    "Raw Round2 By Sense Count",
    "Raw Round2 By Sense Bucket",
    "Before After SimpleWSD",
    "By Sense Count",
    "By Sense Bucket",
    "By MWE Type",
    "By UPOS",
    "By UPOS And MWE",
    "McNemar",
    "Pairwise Example Outcomes",
    "Errors",
)

MODEL_ORDER = ("mling", "simple", "tesla")
CONTENT_UPOS = {"NOUN", "VERB", "ADJ", "ADV"}
RAW_ROUND2_MODEL_ALIASES = {
    "simple_wsd": "simple",
    "tesla_wsd": "tesla",
}

RAW_ROUND2_BASELINES = [
    {"split": "dev", "target_type": "single_word", "model": "simple", "matched": 313, "mismatched": 385, "unmatched": 0, "accuracy": 0.448},
    {"split": "dev", "target_type": "single_word", "model": "tesla", "matched": 340, "mismatched": 358, "unmatched": 0, "accuracy": 0.487},
    {"split": "dev", "target_type": "single_word", "model": "mling", "matched": 379, "mismatched": 319, "unmatched": 0, "accuracy": 0.543},
    {"split": "dev", "target_type": "mwe", "model": "simple", "matched": 70, "mismatched": 4, "unmatched": 0, "accuracy": 0.946},
    {"split": "dev", "target_type": "mwe", "model": "tesla", "matched": 67, "mismatched": 7, "unmatched": 0, "accuracy": 0.905},
    {"split": "dev", "target_type": "mwe", "model": "mling", "matched": 73, "mismatched": 1, "unmatched": 0, "accuracy": 0.986},
    {"split": "dev", "target_type": "total", "model": "simple", "matched": 383, "mismatched": 389, "unmatched": 0, "accuracy": 383 / 772},
    {"split": "dev", "target_type": "total", "model": "tesla", "matched": 407, "mismatched": 365, "unmatched": 0, "accuracy": 407 / 772},
    {"split": "dev", "target_type": "total", "model": "mling", "matched": 452, "mismatched": 320, "unmatched": 0, "accuracy": 452 / 772},
    {"split": "final", "target_type": "single_word", "model": "simple", "matched": 877, "mismatched": 857, "unmatched": 0, "accuracy": 0.506},
    {"split": "final", "target_type": "single_word", "model": "tesla", "matched": 865, "mismatched": 869, "unmatched": 0, "accuracy": 0.499},
    {"split": "final", "target_type": "single_word", "model": "mling", "matched": 971, "mismatched": 763, "unmatched": 0, "accuracy": 0.560},
    {"split": "final", "target_type": "mwe", "model": "simple", "matched": 167, "mismatched": 20, "unmatched": 0, "accuracy": 0.893},
    {"split": "final", "target_type": "mwe", "model": "tesla", "matched": 168, "mismatched": 19, "unmatched": 0, "accuracy": 0.898},
    {"split": "final", "target_type": "mwe", "model": "mling", "matched": 161, "mismatched": 26, "unmatched": 0, "accuracy": 0.861},
    {"split": "final", "target_type": "total", "model": "simple", "matched": 1045, "mismatched": 876, "unmatched": 0, "accuracy": 1045 / 1921},
    {"split": "final", "target_type": "total", "model": "tesla", "matched": 1026, "mismatched": 895, "unmatched": 0, "accuracy": 1026 / 1921},
    {"split": "final", "target_type": "total", "model": "mling", "matched": 1138, "mismatched": 783, "unmatched": 0, "accuracy": 1138 / 1921},
]


@dataclass(frozen=True, slots=True)
class GoldTargetMetadata:
    sentence: str
    target: str
    gold_sense_id: str
    candidate_ids: tuple[str, ...]
    number_of_senses: int
    target_type: str
    upos: str
    lemma: str


@dataclass(frozen=True, slots=True)
class EnrichedPrediction:
    example_id: str
    split: str
    model: str
    model_name: str
    sentence: str
    target: str
    gold_sense_id: str
    predicted_sense_id: str
    ranked_sense_ids: str
    covered: bool
    top1_correct: bool
    top3_correct: bool
    reciprocal_rank: float
    number_of_senses: int
    sense_bucket: str
    difficulty: str
    target_type: str
    upos: str
    lemma: str


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _ci_label(low: float, high: float) -> str:
    return f"{low * 100:.1f}-{high * 100:.1f}"


def sense_bucket(number_of_senses: int) -> str:
    if number_of_senses <= 1:
        return "1"
    if number_of_senses == 2:
        return "2"
    if number_of_senses == 3:
        return "3"
    if number_of_senses <= 5:
        return "4-5"
    if number_of_senses <= 10:
        return "6-10"
    return "11+"


def difficulty_bucket(number_of_senses: int) -> str:
    if number_of_senses <= 1:
        return "monosemous"
    if number_of_senses <= 3:
        return "low_ambiguity"
    return "high_ambiguity"


def _parse_number_of_senses(value: str, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _join_upos(tokens: Sequence[WebAnnoToken]) -> str:
    values: list[str] = []
    for token in tokens:
        if token.upos and token.upos not in values:
            values.append(token.upos)
    return "+".join(values) if values else "UNKNOWN"


def collect_gold_metadata(sentences: Iterable[WebAnnoSentence]) -> list[GoldTargetMetadata]:
    metadata: list[GoldTargetMetadata] = []
    for sentence in sentences:
        grouped_mwes: dict[str, list[WebAnnoToken]] = {}
        for token in sentence.tokens:
            if not is_blank(token.mwe_id):
                grouped_mwes.setdefault(token.mwe_id, []).append(token)

        mwe_token_ids: set[str] = set()
        for tokens in grouped_mwes.values():
            first_with_sense = next((token for token in tokens if token.sense_id), None)
            if first_with_sense is None:
                mwe_token_ids.update(token.token_id for token in tokens)
                continue
            if first_with_sense.mwe_type in EVALUATION_EXCLUDED_MWE_TYPES:
                mwe_token_ids.update(token.token_id for token in tokens)
                continue
            candidate_ids = tuple(first_with_sense.candidate_ids or [first_with_sense.sense_id])
            target = first_with_sense.mwe_lemma if not is_blank(first_with_sense.mwe_lemma) else " ".join(
                token.text for token in tokens
            )
            metadata.append(
                GoldTargetMetadata(
                    sentence=sentence.text,
                    target=target,
                    gold_sense_id=first_with_sense.sense_id,
                    candidate_ids=candidate_ids,
                    number_of_senses=_parse_number_of_senses(first_with_sense.number_of_senses, len(candidate_ids)),
                    target_type="mwe",
                    upos=_join_upos(tokens),
                    lemma=target,
                )
            )
            mwe_token_ids.update(token.token_id for token in tokens)

        for token in sentence.tokens:
            if token.token_id in mwe_token_ids or not is_evaluation_single_token(token):
                continue
            candidate_ids = tuple(token.candidate_ids or [token.sense_id])
            metadata.append(
                GoldTargetMetadata(
                    sentence=sentence.text,
                    target=token.text,
                    gold_sense_id=token.sense_id,
                    candidate_ids=candidate_ids,
                    number_of_senses=_parse_number_of_senses(token.number_of_senses, len(candidate_ids)),
                    target_type="single_word",
                    upos=token.upos or "UNKNOWN",
                    lemma=token.lemma or token.text,
                )
            )
    return metadata


def read_prediction_rows(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _gold_metadata_for_paths(gold_paths: Iterable[Path]) -> list[GoldTargetMetadata]:
    rows: list[GoldTargetMetadata] = []
    for path in gold_paths:
        rows.extend(collect_gold_metadata(parse_webanno_tsv(path)))
    return rows


def _align_metadata(predictions: list[dict[str, str]], gold_rows: list[GoldTargetMetadata]) -> list[GoldTargetMetadata | None]:
    if len(predictions) == len(gold_rows):
        exact = all(
            pred.get("sentence") == gold.sentence
            and pred.get("target") == gold.target
            and pred.get("gold_sense_id") == gold.gold_sense_id
            for pred, gold in zip(predictions, gold_rows)
        )
        if exact:
            return list(gold_rows)

    by_key: dict[tuple[str, str, str], deque[GoldTargetMetadata]] = defaultdict(deque)
    for gold in gold_rows:
        by_key[(gold.sentence, gold.target, gold.gold_sense_id)].append(gold)
    aligned: list[GoldTargetMetadata | None] = []
    for pred in predictions:
        key = (pred.get("sentence", ""), pred.get("target", ""), pred.get("gold_sense_id", ""))
        aligned.append(by_key[key].popleft() if by_key[key] else None)
    return aligned


def align_predictions_with_gold(
    prediction_path: Path | str,
    gold_paths: Iterable[Path],
    split: str,
    model: str,
    model_name: str,
) -> list[EnrichedPrediction]:
    predictions = read_prediction_rows(prediction_path)
    aligned_gold = _align_metadata(predictions, _gold_metadata_for_paths(gold_paths))
    rows: list[EnrichedPrediction] = []
    for index, (prediction, gold) in enumerate(zip(predictions, aligned_gold), start=1):
        if gold is None:
            continue
        ranked_ids = [item for item in prediction.get("ranked_sense_ids", "").split(";") if item]
        gold_sense_id = prediction.get("gold_sense_id", "")
        covered = prediction.get("covered", "").lower() == "true"
        predicted_sense_id = ranked_ids[0] if ranked_ids else ""
        top1_correct = covered and predicted_sense_id == gold_sense_id
        top3_correct = covered and gold_sense_id in ranked_ids[:3]
        reciprocal_rank = 0.0
        if covered and gold_sense_id in ranked_ids:
            reciprocal_rank = 1.0 / (ranked_ids.index(gold_sense_id) + 1)
        number_of_senses = gold.number_of_senses
        bucket = sense_bucket(number_of_senses)
        rows.append(
            EnrichedPrediction(
                example_id=f"{split}-{index:06d}",
                split=split,
                model=model,
                model_name=model_name,
                sentence=prediction.get("sentence", ""),
                target=prediction.get("target", ""),
                gold_sense_id=gold_sense_id,
                predicted_sense_id=predicted_sense_id,
                ranked_sense_ids=";".join(ranked_ids),
                covered=covered,
                top1_correct=top1_correct,
                top3_correct=top3_correct,
                reciprocal_rank=reciprocal_rank,
                number_of_senses=number_of_senses,
                sense_bucket=bucket,
                difficulty=difficulty_bucket(number_of_senses),
                target_type=gold.target_type,
                upos=gold.upos,
                lemma=gold.lemma,
            )
        )
    return rows


def compute_stratified_metrics(rows: Iterable[EnrichedPrediction], group_fields: Sequence[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[EnrichedPrediction]] = defaultdict(list)
    for row in rows:
        groups[tuple(getattr(row, field) for field in group_fields)].append(row)

    output: list[dict[str, object]] = []
    for key in sorted(groups, key=lambda item: tuple(str(part) for part in item)):
        group_rows = groups[key]
        covered = [row for row in group_rows if row.covered]
        covered_total = len(covered)
        correct = sum(1 for row in covered if row.top1_correct)
        top3 = sum(1 for row in covered if row.top3_correct)
        reciprocal_sum = sum(row.reciprocal_rank for row in covered)
        ci_low, ci_high = wilson_interval(correct, covered_total)
        record = {field: value for field, value in zip(group_fields, key)}
        record.update(
            {
                "total_examples": len(group_rows),
                "covered_examples": covered_total,
                "coverage": covered_total / len(group_rows) if group_rows else 0.0,
                "correct_top1": correct,
                "incorrect_top1": covered_total - correct,
                "accuracy": correct / covered_total if covered_total else 0.0,
                "accuracy_ci_low": ci_low,
                "accuracy_ci_high": ci_high,
                "accuracy_ci_95": _ci_label(ci_low, ci_high),
                "top3_accuracy": top3 / covered_total if covered_total else 0.0,
                "mrr": reciprocal_sum / covered_total if covered_total else 0.0,
            }
        )
        output.append(record)
    return output


def _binomial_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    lower_tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2**n)
    return min(1.0, 2 * lower_tail)


def mcnemar_test(
    a_correct: Sequence[bool],
    b_correct: Sequence[bool],
    *,
    model_a: str,
    model_b: str,
    split: str,
    stratum: str = "overall",
    stratum_value: str = "all",
) -> dict[str, object]:
    paired = list(zip(a_correct, b_correct))
    n01 = sum(1 for a, b in paired if not a and b)
    n10 = sum(1 for a, b in paired if a and not b)
    discordant = n01 + n10
    chi_square = ((abs(n01 - n10) - 1) ** 2 / discordant) if discordant else 0.0
    acc_a = sum(1 for value in a_correct if value) / len(a_correct) if a_correct else 0.0
    acc_b = sum(1 for value in b_correct if value) / len(b_correct) if b_correct else 0.0
    diff = acc_a - acc_b
    return {
        "split": split,
        "stratum": stratum,
        "stratum_value": stratum_value,
        "model_a": model_a,
        "model_b": model_b,
        "paired_examples": len(paired),
        "n01": n01,
        "n10": n10,
        "discordant": discordant,
        "chi_square_cc": chi_square,
        "exact_p_value": _binomial_two_sided(n01, discordant),
        "p_value": _binomial_two_sided(n01, discordant),
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "accuracy_diff": diff,
        "effect_direction": model_a if diff > 0 else model_b if diff < 0 else "tie",
    }


def holm_adjust(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    adjusted_rows: list[dict[str, object]] = []
    grouped: dict[tuple[object, object, object], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("split"), row.get("stratum", "overall"), row.get("stratum_value", "all"))].append(row)

    for _key, group in grouped.items():
        sorted_group = sorted(group, key=lambda item: float(item["p_value"]))
        running_max = 0.0
        m = len(sorted_group)
        for rank, row in enumerate(sorted_group):
            adjusted = min(1.0, (m - rank) * float(row["p_value"]))
            running_max = max(running_max, adjusted)
            copied = dict(row)
            copied["holm_p_value"] = running_max
            copied["significant_0_05"] = running_max < 0.05
            adjusted_rows.append(copied)
    return adjusted_rows


def _model_names(comparison_dir: Path) -> dict[str, str]:
    summary_path = comparison_dir / "summary.json"
    names = {name: preset.model_name for name, preset in LEXISENSE_PRESETS.items()}
    if not summary_path.exists():
        return names
    for row in json.loads(summary_path.read_text(encoding="utf-8")):
        if "preset" in row and "model_name" in row:
            names[row["preset"]] = row["model_name"]
    return names


def _load_all_enriched(comparison_dir: Path, gold_dir: Path) -> list[EnrichedPrediction]:
    rows: list[EnrichedPrediction] = []
    names = _model_names(comparison_dir)
    for model in sorted(names):
        for split in ("dev", "final"):
            prediction_path = comparison_dir / model / f"{split}_eval" / "predictions_gold.tsv"
            if not prediction_path.exists():
                continue
            rows.extend(
                align_predictions_with_gold(
                    prediction_path,
                    gold_files_for_split(gold_dir, split),  # type: ignore[arg-type]
                    split,
                    model,
                    names[model],
                )
            )
    return rows


def _pairwise_outcomes(rows: list[EnrichedPrediction]) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str], dict[str, EnrichedPrediction]] = defaultdict(dict)
    for row in rows:
        by_key[(row.split, row.example_id)][row.model] = row
    output: list[dict[str, object]] = []
    for (split, example_id), model_rows in sorted(by_key.items()):
        if not all(model in model_rows for model in ("simple", "tesla", "mling")):
            continue
        record = {
            "split": split,
            "example_id": example_id,
            "sentence": model_rows["simple"].sentence,
            "target": model_rows["simple"].target,
            "gold_sense_id": model_rows["simple"].gold_sense_id,
            "number_of_senses": model_rows["simple"].number_of_senses,
            "sense_bucket": model_rows["simple"].sense_bucket,
            "target_type": model_rows["simple"].target_type,
            "upos": model_rows["simple"].upos,
        }
        for model in ("simple", "tesla", "mling"):
            record[f"{model}_correct"] = model_rows[model].top1_correct
            record[f"{model}_prediction"] = model_rows[model].predicted_sense_id
        correct_models = [model for model in ("simple", "tesla", "mling") if model_rows[model].top1_correct]
        record["correct_models"] = ";".join(correct_models)
        record["error_overlap"] = "all_wrong" if not correct_models else "all_correct" if len(correct_models) == 3 else "mixed"
        output.append(record)
    return output


def _mcnemar_rows(rows: list[EnrichedPrediction]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    comparisons = (("mling", "simple"), ("mling", "tesla"), ("simple", "tesla"))
    strata = [("overall", None), ("sense_bucket", "sense_bucket"), ("target_type", "target_type"), ("upos", "upos")]
    for split in ("dev", "final"):
        split_rows = [row for row in rows if row.split == split and row.covered]
        by_example: dict[str, dict[str, EnrichedPrediction]] = defaultdict(dict)
        for row in split_rows:
            by_example[row.example_id][row.model] = row
        for model_a, model_b in comparisons:
            for stratum_name, attr in strata:
                values = ["all"] if attr is None else sorted({str(getattr(row, attr)) for row in split_rows})
                for value in values:
                    a_values: list[bool] = []
                    b_values: list[bool] = []
                    for model_rows in by_example.values():
                        if model_a not in model_rows or model_b not in model_rows:
                            continue
                        sample = model_rows[model_a]
                        if attr is not None and str(getattr(sample, attr)) != value:
                            continue
                        a_values.append(model_rows[model_a].top1_correct)
                        b_values.append(model_rows[model_b].top1_correct)
                    if len(a_values) < 20:
                        continue
                    output.append(
                        mcnemar_test(
                            a_values,
                            b_values,
                            model_a=model_a,
                            model_b=model_b,
                            split=split,
                            stratum=stratum_name,
                            stratum_value=value,
                        )
                    )
    return holm_adjust(output)


def _errors(rows: Iterable[EnrichedPrediction]) -> list[dict[str, object]]:
    return [
        asdict(row)
        for row in rows
        if not row.covered or not row.top1_correct
    ]


def _write_tsv(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_excel_workbook(tables: dict[str, list[dict[str, object]]], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    default = workbook.active
    workbook.remove(default)
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    best_fill = PatternFill("solid", fgColor="C6EFCE")
    percent_like = {
        "accuracy",
        "coverage",
        "top3_accuracy",
        "mrr",
        "accuracy_ci_low",
        "accuracy_ci_high",
        "p_value",
        "exact_p_value",
        "holm_p_value",
        "accuracy_a",
        "accuracy_b",
        "accuracy_diff",
        "raw_round2_accuracy",
        "after_training_accuracy",
    }
    for sheet_name in EXCEL_SHEETS:
        worksheet = workbook.create_sheet(sheet_name)
        rows = tables.get(sheet_name, [])
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        if not fieldnames:
            fieldnames = ["note"]
            rows = [{"note": "No rows"}]
        worksheet.append(fieldnames)
        for row in rows:
            worksheet.append([row.get(field, "") for field in fieldnames])
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for column_index, field in enumerate(fieldnames, start=1):
            width = min(60, max(12, len(field) + 2, *(len(str(row.get(field, ""))) + 2 for row in rows[:200])))
            worksheet.column_dimensions[get_column_letter(column_index)].width = width
            if field in percent_like:
                for cell in worksheet.iter_cols(min_col=column_index, max_col=column_index, min_row=2):
                    for item in cell:
                        item.number_format = "0.0%"
        if "accuracy" in fieldnames:
            col = get_column_letter(fieldnames.index("accuracy") + 1)
            worksheet.conditional_formatting.add(
                f"{col}2:{col}{worksheet.max_row}",
                CellIsRule(operator="greaterThan", formula=["0.8"], fill=best_fill),
            )
    workbook.save(path)


def _tables_for_workbook(
    summary: list[dict[str, object]],
    raw_baselines: list[dict[str, object]],
    raw_by_sense_count: list[dict[str, object]],
    raw_by_sense_bucket: list[dict[str, object]],
    before_after: list[dict[str, object]],
    by_sense_count: list[dict[str, object]],
    by_sense_bucket: list[dict[str, object]],
    by_mwe: list[dict[str, object]],
    by_upos: list[dict[str, object]],
    by_upos_mwe: list[dict[str, object]],
    mcnemar: list[dict[str, object]],
    pairwise: list[dict[str, object]],
    errors: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    return {
        "Summary": summary,
        "Raw Round2 Baselines": raw_baselines,
        "Raw Round2 By Sense Count": raw_by_sense_count,
        "Raw Round2 By Sense Bucket": raw_by_sense_bucket,
        "Before After SimpleWSD": before_after,
        "By Sense Count": by_sense_count,
        "By Sense Bucket": by_sense_bucket,
        "By MWE Type": by_mwe,
        "By UPOS": by_upos,
        "By UPOS And MWE": by_upos_mwe,
        "McNemar": mcnemar,
        "Pairwise Example Outcomes": pairwise,
        "Errors": errors,
    }


def _raw_baseline_rows() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in RAW_ROUND2_BASELINES:
        total = row["matched"] + row["mismatched"]
        low, high = wilson_interval(int(row["matched"]), int(total))
        output.append(
            {
                **row,
                "total_matched_mismatched": total,
                "accuracy_ci_low": low,
                "accuracy_ci_high": high,
                "accuracy_ci_95": _ci_label(low, high),
                "source": "LexiSense SimpleWSD raw Round 2 table supplied by user",
            }
        )
    return output


def _raw_round2_nonblank(value: object) -> bool:
    return str(value).strip() not in {"", "_", "nan", "None"}


def _raw_round2_number_of_senses(row: dict[str, object]) -> int:
    value = row.get("NoS_rep2")
    try:
        parsed = int(float(str(value)))
    except (TypeError, ValueError):
        parsed = 0
    if parsed > 0:
        return parsed
    return max(1, len(parse_candidate_ids(row.get("Gold_Possible"))))


def _raw_round2_eval_rows(raw_round2_xlsx: Path | str) -> list[dict[str, object]]:
    import pandas as pd

    data = pd.read_excel(raw_round2_xlsx, sheet_name="All_models_wide")
    model_names = [str(column)[len("Sense_") :] for column in data.columns if str(column).startswith("Sense_")]
    output: list[dict[str, object]] = []
    for index, row in enumerate(data.to_dict("records")):
        if not _raw_round2_nonblank(row.get("Gold_Sense")):
            continue
        if str(row.get("Upos")) not in CONTENT_UPOS:
            continue
        if str(row.get("NER")).strip() not in {"", "_", "nan", "None"}:
            continue

        number_of_senses = _raw_round2_number_of_senses(row)
        gold_sense = normalize_sense_id(row.get("Gold_Sense"))
        target_type = "mwe" if str(row.get("type")) == "mwe" else "single_word"
        for source_model in model_names:
            model = RAW_ROUND2_MODEL_ALIASES.get(source_model, source_model)
            predicted_sense = normalize_sense_id(row.get(f"Sense_{source_model}"))
            covered = bool(predicted_sense)
            output.append(
                {
                    "split": "final",
                    "example_id": f"raw-final-{index:06d}",
                    "model": model,
                    "source_model": source_model,
                    "target_type": target_type,
                    "upos": str(row.get("Upos", "")),
                    "lemma": str(row.get("Lemma", "")),
                    "number_of_senses": number_of_senses,
                    "sense_bucket": sense_bucket(number_of_senses),
                    "difficulty": difficulty_bucket(number_of_senses),
                    "gold_sense_id": gold_sense,
                    "predicted_sense_id": predicted_sense,
                    "covered": covered,
                    "top1_correct": covered and predicted_sense == gold_sense,
                }
            )
    return output


def _raw_round2_metrics(rows: list[dict[str, object]], group_fields: Sequence[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)

    output: list[dict[str, object]] = []
    for key in sorted(groups, key=lambda item: tuple(str(part) for part in item)):
        group_rows = groups[key]
        covered = [row for row in group_rows if row["covered"]]
        total = len(group_rows)
        covered_total = len(covered)
        correct = sum(1 for row in covered if row["top1_correct"])
        ci_low, ci_high = wilson_interval(correct, covered_total)
        record = {field: value for field, value in zip(group_fields, key)}
        record.update(
            {
                "total_examples": total,
                "covered_examples": covered_total,
                "coverage": covered_total / total if total else 0.0,
                "matched": correct,
                "mismatched": covered_total - correct,
                "unmatched": total - covered_total,
                "accuracy": correct / covered_total if covered_total else 0.0,
                "accuracy_ci_low": ci_low,
                "accuracy_ci_high": ci_high,
                "accuracy_ci_95": _ci_label(ci_low, ci_high),
                "source": "LexiSense Round 2 workbook",
            }
        )
        output.append(record)
    return output


def _raw_baseline_rows_from_workbook(
    raw_round2_xlsx: Path | str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    raw_rows = _raw_round2_eval_rows(raw_round2_xlsx)
    by_total = _raw_round2_metrics(raw_rows, ["split", "model"])
    by_type = _raw_round2_metrics(raw_rows, ["split", "model", "target_type"])
    by_sense_count = _raw_round2_metrics(raw_rows, ["split", "model", "number_of_senses"])
    by_sense_bucket = _raw_round2_metrics(raw_rows, ["split", "model", "sense_bucket"])

    baselines: list[dict[str, object]] = []
    for row in by_type:
        baselines.append(
            {
                "split": row["split"],
                "target_type": row["target_type"],
                "model": row["model"],
                "matched": row["matched"],
                "mismatched": row["mismatched"],
                "unmatched": row["unmatched"],
                "total_matched_mismatched": row["covered_examples"],
                "accuracy": row["accuracy"],
                "accuracy_ci_low": row["accuracy_ci_low"],
                "accuracy_ci_high": row["accuracy_ci_high"],
                "accuracy_ci_95": row["accuracy_ci_95"],
                "source": row["source"],
            }
        )
    for row in by_total:
        baselines.append(
            {
                "split": row["split"],
                "target_type": "total",
                "model": row["model"],
                "matched": row["matched"],
                "mismatched": row["mismatched"],
                "unmatched": row["unmatched"],
                "total_matched_mismatched": row["covered_examples"],
                "accuracy": row["accuracy"],
                "accuracy_ci_low": row["accuracy_ci_low"],
                "accuracy_ci_high": row["accuracy_ci_high"],
                "accuracy_ci_95": row["accuracy_ci_95"],
                "source": row["source"],
            }
        )
    return baselines, by_sense_count, by_sense_bucket


def _before_after_rows(
    summary: list[dict[str, object]],
    by_mwe: list[dict[str, object]],
    raw_baselines: list[dict[str, object]],
) -> list[dict[str, object]]:
    after_lookup: dict[tuple[object, object, str], dict[str, object]] = {}
    for row in summary:
        after_lookup[(row.get("split"), row.get("model"), "total")] = row
    for row in by_mwe:
        after_lookup[(row.get("split"), row.get("model"), str(row.get("target_type")))] = row

    output: list[dict[str, object]] = []
    for raw in raw_baselines:
        key = (raw["split"], raw["model"], raw["target_type"])
        after = after_lookup.get(key)
        if not after:
            continue
        raw_accuracy = float(raw["accuracy"])
        after_accuracy = float(after["accuracy"])
        output.append(
            {
                "split": raw["split"],
                "target_type": raw["target_type"],
                "model": raw["model"],
                "raw_round2_correct": raw["matched"],
                "raw_round2_total": raw["total_matched_mismatched"],
                "raw_round2_accuracy": raw_accuracy,
                "raw_round2_ci_95": raw["accuracy_ci_95"],
                "after_training_correct": after["correct_top1"],
                "after_training_covered": after["covered_examples"],
                "after_training_total": after["total_examples"],
                "after_training_accuracy": after_accuracy,
                "after_training_ci_95": after["accuracy_ci_95"],
                "accuracy_delta_points": (after_accuracy - raw_accuracy) * 100,
                "comparison_note": "Aggregate comparison only; raw per-example predictions are unavailable for paired significance testing.",
            }
        )
    return output


def _write_methods_report(summary: list[dict[str, object]], raw_rows: list[dict[str, object]], out_dir: Path) -> None:
    final_summary = [row for row in summary if row.get("split") == "final"]
    lines = [
        "# WSD Distillation Deep Evaluation",
        "",
        "## Experimental Setting",
        "",
        "Three SentenceTransformer-based WSD rankers were trained from the same model families used in LexiSense SimpleWSD: `simple` (`all-MiniLM-L6-v2`), `tesla` (`te-sla/TeslaXLM`) and `mling` (`intfloat/multilingual-e5-large`). Training used silver WebAnno annotations after removing every sentence present in the gold evaluation files. The development split is `sr-elexis-WSD_0001_0120-gold.tsv`; the final split is the concatenation of `0301-0400`, `0401-0500` and `0501-0600` gold files.",
        "",
        "The task is evaluated as lexical-sense selection among candidates for the same lemma/POS. Because operational WSD requires one selected sense, top-1 accuracy is the primary metric. Top-3 accuracy and MRR are retained as ranking diagnostics.",
        "",
        "## Training Configuration",
        "",
        "| preset | base model | epochs | batch size | warmup steps | text prefix | normalized embeddings |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for preset in ("simple", "tesla", "mling"):
        summary_path = out_dir.parent / "comparison" / preset / "train_summary.json"
        if summary_path.exists():
            train = json.loads(summary_path.read_text(encoding="utf-8"))
            lines.append(
                f"| {preset} | `{train['model_name']}` | {train['epochs']} | {train['batch_size']} | {train['warmup_steps']} | `{train['text_prefix']}` | {train['normalize_embeddings']} |"
            )
    lines.extend(
        [
            "",
            "## Final After-Training Results",
            "",
            "| preset | covered / total | accuracy | 95% Wilson CI | top-3 | MRR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in final_summary:
        lines.append(
            f"| {row['model']} | {row['covered_examples']} / {row['total_examples']} | {float(row['accuracy']) * 100:.1f}% | {row['accuracy_ci_95']} | {float(row['top3_accuracy']) * 100:.1f}% | {float(row['mrr']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Raw SimpleWSD Round 2 Reference",
            "",
            "The raw LexiSense table supplied separately is a useful before-training baseline. It should be interpreted as an external aggregate reference, not as input to McNemar tests here, because this repository currently has only aggregate raw counts and not raw per-example predictions.",
            "",
            "| split | type | preset | matched/mismatched/unmatched | accuracy | 95% Wilson CI |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in raw_rows:
        lines.append(
            f"| {row['split']} | {row['target_type']} | {row['model']} | {row['matched']}/{row['mismatched']}/{row['unmatched']} | {float(row['accuracy']) * 100:.1f}% | {row['accuracy_ci_95']} |"
        )
    lines.extend(
        [
            "",
            "## Statistical Testing",
            "",
            "Pairwise model differences are tested with McNemar's test over examples covered by both compared models. The reported `n01` count is the number of examples where model A is wrong and model B is correct; `n10` is the reverse. Holm correction is applied within each split and stratum. The final split is the main evidence for reporting; the development split is exploratory.",
            "",
            "## Stratified Analyses",
            "",
            "The Excel workbook and TSV outputs break results down by exact number of candidate senses, grouped ambiguity bucket, single-word versus multi-word expression, UPOS, and UPOS combined with target type. These tables identify where distillation helps most and whether gains are concentrated in highly ambiguous lexical items or specific word classes.",
            "",
        ]
    )
    (out_dir / "training_methods_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_analysis_outputs(
    comparison_dir: Path | str,
    gold_dir: Path | str,
    out_dir: Path | str,
    raw_round2_xlsx: Path | str | None = None,
) -> dict[str, object]:
    comparison_dir = Path(comparison_dir)
    gold_dir = Path(gold_dir)
    out_dir = Path(out_dir)
    rows = _load_all_enriched(comparison_dir, gold_dir)
    row_dicts = [asdict(row) for row in rows]
    summary = compute_stratified_metrics(rows, ["split", "model"])
    by_sense_count = compute_stratified_metrics(rows, ["split", "model", "number_of_senses"])
    by_sense_bucket = compute_stratified_metrics(rows, ["split", "model", "sense_bucket"])
    by_mwe = compute_stratified_metrics(rows, ["split", "model", "target_type"])
    by_upos = compute_stratified_metrics(rows, ["split", "model", "upos"])
    by_upos_mwe = compute_stratified_metrics(rows, ["split", "model", "upos", "target_type"])
    mcnemar = _mcnemar_rows(rows)
    pairwise = _pairwise_outcomes(rows)
    errors = _errors(rows)
    if raw_round2_xlsx:
        raw_baselines, raw_by_sense_count, raw_by_sense_bucket = _raw_baseline_rows_from_workbook(raw_round2_xlsx)
    else:
        raw_baselines = _raw_baseline_rows()
        raw_by_sense_count = []
        raw_by_sense_bucket = []
    before_after = _before_after_rows(summary, by_mwe, raw_baselines)

    stratified = by_sense_count + by_sense_bucket + by_mwe + by_upos + by_upos_mwe
    _write_tsv(summary, out_dir / "summary_metrics.tsv")
    _write_tsv(stratified, out_dir / "stratified_metrics.tsv")
    _write_tsv(mcnemar, out_dir / "mcnemar_tests.tsv")
    _write_tsv(errors, out_dir / "errors_enriched.tsv")
    _write_tsv(pairwise, out_dir / "pairwise_example_outcomes.tsv")
    _write_tsv(raw_baselines, out_dir / "raw_round2_baselines.tsv")
    _write_tsv(raw_by_sense_count, out_dir / "raw_round2_by_sense_count.tsv")
    _write_tsv(raw_by_sense_bucket, out_dir / "raw_round2_by_sense_bucket.tsv")
    _write_tsv(before_after, out_dir / "before_after_distillation.tsv")
    _write_tsv(row_dicts, out_dir / "enriched_predictions.tsv")

    analysis_summary = {
        "models": sorted({row.model for row in rows}),
        "splits": sorted({row.split for row in rows}),
        "total_enriched_predictions": len(rows),
        "summary_metrics": summary,
        "raw_round2_baselines": raw_baselines,
        "raw_round2_by_sense_count": raw_by_sense_count,
        "raw_round2_by_sense_bucket": raw_by_sense_bucket,
        "before_after_distillation": before_after,
    }
    _write_json(analysis_summary, out_dir / "analysis_summary.json")
    write_excel_workbook(
        _tables_for_workbook(
            summary,
            raw_baselines,
            raw_by_sense_count,
            raw_by_sense_bucket,
            before_after,
            by_sense_count,
            by_sense_bucket,
            by_mwe,
            by_upos,
            by_upos_mwe,
            mcnemar,
            pairwise,
            errors,
        ),
        out_dir / "wsd_deep_eval.xlsx",
    )
    _write_methods_report(summary, raw_baselines, out_dir)
    return analysis_summary
