from __future__ import annotations

from lexisense_distill.dataset import TrainingPair
from lexisense_distill.train import build_input_examples


def test_build_input_examples_applies_text_prefix_to_both_sides() -> None:
    pair = TrainingPair(
        text="Program je **održan**.",
        definition="organizovati događaj",
        label=1.0,
        sentence="Program je održan.",
        target="održan",
        sense_id="ENG30-01733477-v",
        source_file="fixture.tsv",
    )

    examples = build_input_examples([pair], text_prefix="query: ")

    assert len(examples) == 1
    assert examples[0].texts == ["query: Program je **održan**.", "query: organizovati događaj"]
    assert examples[0].label == 1.0


def test_build_input_examples_does_not_double_prefix() -> None:
    pair = TrainingPair(
        text="query: Program je **održan**.",
        definition="query: organizovati događaj",
        label=0.0,
        sentence="Program je održan.",
        target="održan",
        sense_id="ENG30-01733477-v",
        source_file="fixture.tsv",
    )

    examples = build_input_examples([pair], text_prefix="query: ")

    assert examples[0].texts == ["query: Program je **održan**.", "query: organizovati događaj"]
