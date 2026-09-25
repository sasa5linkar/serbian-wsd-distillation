from __future__ import annotations

from lexisense_distill.presets import LEXISENSE_PRESETS, preset_names


def test_presets_match_lexisense_simple_wsd_config() -> None:
    assert preset_names() == ["simple", "tesla", "mling"]
    assert LEXISENSE_PRESETS["simple"].model_name == "all-MiniLM-L6-v2"
    assert LEXISENSE_PRESETS["simple"].output_name == "wsd-distilled-simple"
    assert LEXISENSE_PRESETS["simple"].text_prefix == ""
    assert LEXISENSE_PRESETS["simple"].normalize_embeddings is False
    assert LEXISENSE_PRESETS["simple"].batch_size == 32

    assert LEXISENSE_PRESETS["tesla"].model_name == "te-sla/TeslaXLM"
    assert LEXISENSE_PRESETS["tesla"].output_name == "wsd-distilled-tesla"
    assert LEXISENSE_PRESETS["tesla"].text_prefix == ""
    assert LEXISENSE_PRESETS["tesla"].normalize_embeddings is False
    assert LEXISENSE_PRESETS["tesla"].batch_size == 16

    assert LEXISENSE_PRESETS["mling"].model_name == "intfloat/multilingual-e5-large"
    assert LEXISENSE_PRESETS["mling"].output_name == "wsd-distilled-mling"
    assert LEXISENSE_PRESETS["mling"].text_prefix == "query: "
    assert LEXISENSE_PRESETS["mling"].normalize_embeddings is True
    assert LEXISENSE_PRESETS["mling"].batch_size == 4
