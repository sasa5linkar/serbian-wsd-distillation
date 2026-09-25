from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .webanno import normalize_sense_id


@dataclass(frozen=True, slots=True)
class SenseDefinition:
    sense_id: str
    lemma: str
    upos: str
    definition: str


def load_sense_definitions(path: Path | str) -> dict[str, SenseDefinition]:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return _load_excel_sense_definitions(path)
    return _load_delimited_sense_definitions(path)


def _load_excel_sense_definitions(path: Path) -> dict[str, SenseDefinition]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas and openpyxl are required to read Excel sense repositories") from exc

    frame = pd.read_excel(path, engine="openpyxl")
    return _records_to_definitions(frame.to_dict(orient="records"))


def _load_delimited_sense_definitions(path: Path) -> dict[str, SenseDefinition]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return _records_to_definitions(csv.DictReader(handle, delimiter=delimiter))


def _records_to_definitions(records) -> dict[str, SenseDefinition]:
    definitions: dict[str, SenseDefinition] = {}
    for record in records:
        raw_id = record.get("senseID") or record.get("sense_id") or record.get("id")
        sense_id = normalize_sense_id(raw_id)
        definition = str(record.get("definition") or record.get("gloss") or "").strip()
        if not sense_id or not definition:
            continue
        definitions[sense_id] = SenseDefinition(
            sense_id=sense_id,
            lemma=str(record.get("lemma") or "").strip(),
            upos=str(record.get("upos") or record.get("pos") or "").strip(),
            definition=definition,
        )
    return definitions
