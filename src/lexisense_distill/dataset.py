from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .sense_repo import SenseDefinition
from .webanno import WebAnnoSentence, WebAnnoToken, gold_files_for_split, is_blank, iter_tsv_files, mark_span, parse_webanno_tsv

CONTENT_UPOS = {"NOUN", "VERB", "ADJ", "ADV"}
EVENT_NE_TYPES = {"ROLE", "EVENT", "DEMO", "PRODUCT", "WORK"}
EVALUATION_EXCLUDED_MWE_TYPES = {"AdvID", "ConjID", "PronID"}


@dataclass(frozen=True, slots=True)
class TrainingPair:
    text: str
    definition: str
    label: float
    sentence: str
    target: str
    sense_id: str
    source_file: str


@dataclass(slots=True)
class DatasetBuildResult:
    pairs: list[TrainingPair] = field(default_factory=list)
    coverage_errors: list[dict[str, str]] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WsdTarget:
    sentence: WebAnnoSentence
    target: str
    spans: tuple[tuple[int, int], ...]
    sense_id: str
    candidate_ids: tuple[str, ...]

    @property
    def marked_text(self) -> str:
        return mark_span(self.sentence.text, self.spans)


def collect_gold_sentence_texts(gold_dir: Path | str) -> set[str]:
    sentence_texts: set[str] = set()
    for path in gold_files_for_split(gold_dir, "all"):
        for sentence in parse_webanno_tsv(path):
            sentence_texts.add(sentence.text)
    return sentence_texts


def collect_targets(sentences: Iterable[WebAnnoSentence]) -> list[WsdTarget]:
    targets: list[WsdTarget] = []
    for sentence in sentences:
        grouped_mwes: dict[str, list[WebAnnoToken]] = {}
        for token in sentence.tokens:
            if not is_blank(token.mwe_id):
                grouped_mwes.setdefault(token.mwe_id, []).append(token)

        mwe_token_ids: set[str] = set()
        for tokens in grouped_mwes.values():
            first_with_sense = next((token for token in tokens if token.sense_id), None)
            if first_with_sense is None:
                continue
            candidate_ids = first_with_sense.candidate_ids or [first_with_sense.sense_id]
            spans = tuple((token.start, token.end) for token in tokens)
            target_text = first_with_sense.mwe_lemma if not is_blank(first_with_sense.mwe_lemma) else " ".join(
                token.text for token in tokens
            )
            targets.append(
                WsdTarget(
                    sentence=sentence,
                    target=target_text,
                    spans=spans,
                    sense_id=first_with_sense.sense_id,
                    candidate_ids=tuple(candidate_ids),
                )
            )
            mwe_token_ids.update(token.token_id for token in tokens)

        for token in sentence.tokens:
            if token.token_id in mwe_token_ids or not token.sense_id:
                continue
            candidate_ids = token.candidate_ids or [token.sense_id]
            targets.append(
                WsdTarget(
                    sentence=sentence,
                    target=token.text,
                    spans=((token.start, token.end),),
                    sense_id=token.sense_id,
                    candidate_ids=tuple(candidate_ids),
                )
            )
    return targets


def gold_in_candidates(token: WebAnnoToken) -> bool:
    return bool(token.sense_id and token.candidate_ids and token.sense_id in token.candidate_ids)


def token_allowed_by_ne_filter(token: WebAnnoToken) -> bool:
    return is_blank(token.ne_type) or token.ne_type in EVENT_NE_TYPES


def token_has_valid_lexisense_lemma(token: WebAnnoToken) -> bool:
    return bool(token.lemma and not is_blank(token.lemma) and not token.lemma[0].isdigit())


def is_evaluation_single_token(token: WebAnnoToken) -> bool:
    return (
        bool(token.sense_id)
        and token.upos in CONTENT_UPOS
        and token_allowed_by_ne_filter(token)
        and token_has_valid_lexisense_lemma(token)
    )


def is_evaluation_mwe(tokens: list[WebAnnoToken]) -> bool:
    first_with_sense = next((token for token in tokens if token.sense_id), None)
    return bool(
        first_with_sense
        and first_with_sense.mwe_type not in EVALUATION_EXCLUDED_MWE_TYPES
    )


def collect_evaluation_targets(sentences: Iterable[WebAnnoSentence]) -> list[WsdTarget]:
    targets: list[WsdTarget] = []
    for sentence in sentences:
        grouped_mwes: dict[str, list[WebAnnoToken]] = {}
        for token in sentence.tokens:
            if not is_blank(token.mwe_id):
                grouped_mwes.setdefault(token.mwe_id, []).append(token)

        mwe_token_ids: set[str] = set()
        for tokens in grouped_mwes.values():
            first_with_sense = next((token for token in tokens if token.sense_id), None)
            if first_with_sense is not None and is_evaluation_mwe(tokens):
                spans = tuple((token.start, token.end) for token in tokens)
                target_text = first_with_sense.mwe_lemma if not is_blank(first_with_sense.mwe_lemma) else " ".join(
                    token.text for token in tokens
                )
                targets.append(
                    WsdTarget(
                        sentence=sentence,
                        target=target_text,
                        spans=spans,
                        sense_id=first_with_sense.sense_id,
                        candidate_ids=tuple(first_with_sense.candidate_ids or [first_with_sense.sense_id]),
                    )
                )
            mwe_token_ids.update(token.token_id for token in tokens)

        for token in sentence.tokens:
            if token.token_id in mwe_token_ids or not is_evaluation_single_token(token):
                continue
            targets.append(
                WsdTarget(
                    sentence=sentence,
                    target=token.text,
                    spans=((token.start, token.end),),
                    sense_id=token.sense_id,
                    candidate_ids=tuple(token.candidate_ids or [token.sense_id]),
                )
            )
    return targets


def build_training_pairs(
    silver_files: Iterable[Path | str],
    sense_definitions: dict[str, SenseDefinition],
    *,
    gold_sentence_texts: set[str],
) -> DatasetBuildResult:
    stats = {
        "silver_files": 0,
        "sentences": 0,
        "skipped_gold_sentences": 0,
        "targets": 0,
        "positive_pairs": 0,
        "negative_pairs": 0,
    }
    pairs: list[TrainingPair] = []
    coverage_errors: list[dict[str, str]] = []

    for file_path in silver_files:
        stats["silver_files"] += 1
        parsed = parse_webanno_tsv(file_path)
        usable_sentences: list[WebAnnoSentence] = []
        for sentence in parsed:
            stats["sentences"] += 1
            if sentence.text in gold_sentence_texts:
                stats["skipped_gold_sentences"] += 1
                continue
            usable_sentences.append(sentence)

        for target in collect_targets(usable_sentences):
            stats["targets"] += 1
            for candidate_id in target.candidate_ids:
                sense = sense_definitions.get(candidate_id)
                if sense is None:
                    coverage_errors.append(
                        {
                            "sentence": target.sentence.text,
                            "target": target.target,
                            "sense_id": candidate_id,
                            "reason": "missing_definition",
                        }
                    )
                    continue
                label = 1.0 if candidate_id == target.sense_id else 0.0
                pairs.append(
                    TrainingPair(
                        text=target.marked_text,
                        definition=sense.definition,
                        label=label,
                        sentence=target.sentence.text,
                        target=target.target,
                        sense_id=candidate_id,
                        source_file=target.sentence.source_file,
                    )
                )
                if label == 1.0:
                    stats["positive_pairs"] += 1
                else:
                    stats["negative_pairs"] += 1

    return DatasetBuildResult(pairs=pairs, coverage_errors=coverage_errors, stats=stats)


def build_training_pairs_from_dirs(
    silver_dir: Path | str,
    gold_dir: Path | str,
    sense_definitions: dict[str, SenseDefinition],
) -> DatasetBuildResult:
    return build_training_pairs(
        iter_tsv_files(silver_dir),
        sense_definitions,
        gold_sentence_texts=collect_gold_sentence_texts(gold_dir),
    )


def write_jsonl(pairs: Iterable[TrainingPair], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            handle.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")


def read_jsonl(path: Path | str) -> list[TrainingPair]:
    pairs: list[TrainingPair] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                pairs.append(TrainingPair(**json.loads(line)))
    return pairs


def write_coverage_errors(errors: Iterable[dict[str, str]], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
