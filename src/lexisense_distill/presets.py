from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LexiSensePreset:
    name: str
    model_name: str
    output_name: str
    text_prefix: str
    normalize_embeddings: bool
    batch_size: int


LEXISENSE_PRESETS: dict[str, LexiSensePreset] = {
    "simple": LexiSensePreset(
        name="simple",
        model_name="all-MiniLM-L6-v2",
        output_name="wsd-distilled-simple",
        text_prefix="",
        normalize_embeddings=False,
        batch_size=32,
    ),
    "tesla": LexiSensePreset(
        name="tesla",
        model_name="te-sla/TeslaXLM",
        output_name="wsd-distilled-tesla",
        text_prefix="",
        normalize_embeddings=False,
        batch_size=16,
    ),
    "mling": LexiSensePreset(
        name="mling",
        model_name="intfloat/multilingual-e5-large",
        output_name="wsd-distilled-mling",
        text_prefix="query: ",
        normalize_embeddings=True,
        batch_size=4,
    ),
}


def preset_names() -> list[str]:
    return list(LEXISENSE_PRESETS)
