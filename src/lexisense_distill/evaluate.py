from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from .dataset import collect_evaluation_targets
from .sense_repo import SenseDefinition
from .ssl_utils import prefer_certifi_default_context
from .webanno import gold_files_for_split, mark_span, normalize_sense_id, parse_candidate_ids, parse_webanno_tsv


class Ranker(Protocol):
    def rank(self, text: str, definitions: list[str]) -> list[int]:
        """Return candidate indexes ordered best to worst."""


@dataclass(frozen=True, slots=True)
class RankedPrediction:
    sentence: str
    target: str
    gold_sense_id: str
    ranked_sense_ids: list[str]
    covered: bool


def compute_metrics(predictions: Iterable[RankedPrediction]) -> dict[str, float | int]:
    rows = list(predictions)
    covered = [row for row in rows if row.covered]
    total = len(rows)
    covered_total = len(covered)
    top1 = sum(1 for row in covered if row.ranked_sense_ids[:1] == [row.gold_sense_id])
    top3 = sum(1 for row in covered if row.gold_sense_id in row.ranked_sense_ids[:3])
    reciprocal_sum = 0.0
    for row in covered:
        try:
            reciprocal_sum += 1.0 / (row.ranked_sense_ids.index(row.gold_sense_id) + 1)
        except ValueError:
            reciprocal_sum += 0.0
    return {
        "total_examples": total,
        "covered_examples": covered_total,
        "coverage": covered_total / total if total else 0.0,
        "top1_accuracy": top1 / covered_total if covered_total else 0.0,
        "top3_accuracy": top3 / covered_total if covered_total else 0.0,
        "mrr": reciprocal_sum / covered_total if covered_total else 0.0,
    }


class SentenceTransformerRanker:
    def __init__(self, model_path: str, *, text_prefix: str = "", normalize_embeddings: bool = False) -> None:
        prefer_certifi_default_context()
        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the train extra to use SentenceTransformerRanker") from exc
        self.model = SentenceTransformer(model_path)
        self.util = util
        self.text_prefix = text_prefix
        self.normalize_embeddings = normalize_embeddings

    def _prefix(self, text: str) -> str:
        return text if not self.text_prefix or text.startswith(self.text_prefix) else f"{self.text_prefix}{text}"

    def rank(self, text: str, definitions: list[str]) -> list[int]:
        embeddings = self.model.encode(
            [self._prefix(text), *[self._prefix(definition) for definition in definitions]],
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        scores = self.util.cos_sim(embeddings[0], embeddings[1:]).squeeze(0)
        if hasattr(scores, "detach"):
            scores = scores.detach()
        if hasattr(scores, "cpu"):
            scores = scores.cpu()
        return [index for index, _score in sorted(enumerate(scores.tolist()), key=lambda item: item[1], reverse=True)]


def evaluate_gold_split(
    gold_dir: Path | str,
    split: str,
    sense_definitions: dict[str, SenseDefinition],
    ranker: Ranker,
) -> list[RankedPrediction]:
    predictions: list[RankedPrediction] = []
    for path in gold_files_for_split(gold_dir, split):  # type: ignore[arg-type]
        for target in collect_evaluation_targets(parse_webanno_tsv(path)):
            candidate_ids = list(target.candidate_ids)
            if len(candidate_ids) == 1:
                predictions.append(
                    RankedPrediction(
                        sentence=target.sentence.text,
                        target=target.target,
                        gold_sense_id=target.sense_id,
                        ranked_sense_ids=candidate_ids,
                        covered=True,
                    )
                )
                continue
            senses = [sense_definitions.get(candidate_id) for candidate_id in candidate_ids]
            if not candidate_ids or any(sense is None for sense in senses):
                predictions.append(
                    RankedPrediction(
                        sentence=target.sentence.text,
                        target=target.target,
                        gold_sense_id=target.sense_id,
                        ranked_sense_ids=[],
                        covered=False,
                    )
                )
                continue
            definitions = [sense.definition for sense in senses if sense is not None]
            order = ranker.rank(target.marked_text, definitions)
            predictions.append(
                RankedPrediction(
                    sentence=target.sentence.text,
                    target=target.target,
                    gold_sense_id=target.sense_id,
                    ranked_sense_ids=[candidate_ids[index] for index in order],
                    covered=True,
                )
            )
    return predictions


def _workbook_row_universe(raw_round2_xlsx: Path | str):
    import pandas as pd

    data = pd.read_excel(raw_round2_xlsx, sheet_name="All_models_wide")
    for index, row in enumerate(data.to_dict("records")):
        if str(row.get("Gold_Sense")).strip() in {"", "_", "nan", "None"}:
            continue
        gold_sense = normalize_sense_id(row.get("Gold_Sense"))
        if str(row.get("Upos")) not in {"NOUN", "VERB", "ADJ", "ADV"}:
            continue
        if str(row.get("NER")).strip() not in {"", "_", "nan", "None"}:
            continue
        yield index, row, gold_sense


def _sentences_by_workbook_id(gold_dir: Path | str) -> dict[int, str]:
    sentences: dict[int, str] = {}
    for path in gold_files_for_split(gold_dir, "final"):
        name = path.name
        if "0301-0400" in name:
            first_id = 301
        elif "0401-0500" in name:
            first_id = 401
        elif "0501-0600" in name:
            first_id = 501
        else:
            continue
        for sentence in parse_webanno_tsv(path):
            sentences[first_id + sentence.sentence_index - 1] = sentence.text
    return sentences


def _parse_norm_span(value: object) -> tuple[int, int] | None:
    text = str(value).strip()
    if "-" not in text:
        return None
    start_text, end_text = text.split("-", 1)
    try:
        return int(start_text), int(end_text)
    except ValueError:
        return None


def evaluate_raw_round2_workbook(
    raw_round2_xlsx: Path | str,
    gold_dir: Path | str,
    sense_definitions: dict[str, SenseDefinition],
    ranker: Ranker,
) -> list[RankedPrediction]:
    sentence_lookup = _sentences_by_workbook_id(gold_dir)
    predictions: list[RankedPrediction] = []
    for _index, row, gold_sense in _workbook_row_universe(raw_round2_xlsx):
        sent_id = int(float(str(row.get("sent_id"))))
        sentence = sentence_lookup.get(sent_id, "")
        target = str(row.get("form") or row.get("Lemma") or "")
        span = _parse_norm_span(row.get("norm_span"))
        marked = mark_span(sentence, [span]) if sentence and span else sentence
        candidate_ids = parse_candidate_ids(row.get("Gold_Possible"))
        if gold_sense and gold_sense not in candidate_ids:
            candidate_ids.append(gold_sense)
        if len(candidate_ids) == 1:
            predictions.append(
                RankedPrediction(
                    sentence=sentence,
                    target=target,
                    gold_sense_id=gold_sense,
                    ranked_sense_ids=candidate_ids,
                    covered=True,
                )
            )
            continue
        senses = [sense_definitions.get(candidate_id) for candidate_id in candidate_ids]
        if not sentence or not candidate_ids or any(sense is None for sense in senses):
            predictions.append(
                RankedPrediction(
                    sentence=sentence,
                    target=target,
                    gold_sense_id=gold_sense,
                    ranked_sense_ids=[],
                    covered=False,
                )
            )
            continue
        definitions = [sense.definition for sense in senses if sense is not None]
        order = ranker.rank(marked, definitions)
        predictions.append(
            RankedPrediction(
                sentence=sentence,
                target=target,
                gold_sense_id=gold_sense,
                ranked_sense_ids=[candidate_ids[index] for index in order],
                covered=True,
            )
        )
    return predictions


def write_metrics(metrics: dict[str, float | int], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def write_predictions(predictions: Iterable[RankedPrediction], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["sentence", "target", "gold_sense_id", "ranked_sense_ids", "covered"],
        )
        writer.writeheader()
        for row in predictions:
            writer.writerow(
                {
                    "sentence": row.sentence,
                    "target": row.target,
                    "gold_sense_id": row.gold_sense_id,
                    "ranked_sense_ids": ";".join(row.ranked_sense_ids),
                    "covered": str(row.covered),
                }
            )


def write_errors(predictions: Iterable[RankedPrediction], path: Path | str) -> None:
    errors = [
        row
        for row in predictions
        if not row.covered or row.ranked_sense_ids[:1] != [row.gold_sense_id]
    ]
    write_predictions(errors, path)
