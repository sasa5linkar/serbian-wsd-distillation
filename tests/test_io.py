from __future__ import annotations

import json
from pathlib import Path

from lexisense_distill.dataset import TrainingPair, write_jsonl


def test_write_jsonl_preserves_serbian_utf8(tmp_path: Path) -> None:
    out = tmp_path / "pairs.jsonl"
    write_jsonl(
        [
            TrainingPair(
                text="Program je **održan**.",
                definition="organizovati događaj",
                label=1.0,
                sentence="Program je održan.",
                target="održan",
                sense_id="ENG30-01733477-v",
                source_file="fixture.tsv",
            )
        ],
        out,
    )

    raw = out.read_text(encoding="utf-8")
    assert "održan" in raw
    payload = json.loads(raw)
    assert payload["definition"] == "organizovati događaj"
