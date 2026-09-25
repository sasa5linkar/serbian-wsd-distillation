from __future__ import annotations

from pathlib import Path

from lexisense_distill.webanno import (
    gold_files_for_split,
    mark_span,
    normalize_sense_id,
    parse_candidate_ids,
    parse_webanno_tsv,
    strip_webanno_suffix,
)


def write_fixture(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_normalizes_webanno_suffix_urls_and_candidate_lists() -> None:
    assert strip_webanno_suffix("http://llod.jerteh.rs/WSD/seENG3000551215n[42]") == "http://llod.jerteh.rs/WSD/seENG3000551215n"
    assert normalize_sense_id("http://llod.jerteh.rs/WSD/seENG3000551215n[42]") == "ENG30-00551215-n"
    assert normalize_sense_id("http://llod.jerteh.rs/WSD/seLLM0288[450]") == "LLM-0288"
    assert normalize_sense_id("http://llod.jerteh.rs/WSD/seCVMWE0399[607]") == "CVMWE-0399"
    assert normalize_sense_id("http://llod.jerteh.rs/WSD/seCV004") == "CV004"
    assert normalize_sense_id("seDRJ0193498_0") == "DRJ0-193498/0"
    assert normalize_sense_id("DRJ0-193498/0") == "DRJ0-193498/0"
    assert normalize_sense_id("seRSSJ49196_0") == "RSSJ-49196/0"
    assert normalize_sense_id("seCVDf176_0") == "CVDf-176/0"
    assert normalize_sense_id("seDRS0457358_00") == "DRS0-457358/0.0"
    assert parse_candidate_ids("ENG30-00118523-v\\;ENG30-02202928-v[450]|CV004") == [
        "ENG30-00118523-v",
        "ENG30-02202928-v",
        "CV004",
    ]
    assert parse_candidate_ids("DRJ0-193498/0\\;RSSJ-49196/0") == ["DRJ0-193498/0", "RSSJ-49196/0"]


def test_parses_webanno_sentences_and_preserves_serbian_text(tmp_path: Path) -> None:
    fixture = tmp_path / "sample.tsv"
    write_fixture(
        fixture,
        "\n".join(
            [
                "#FORMAT=WebAnno TSV 3.3",
                "#T_SP=webanno.custom.WSD|Comment|Explanation|KBid|NumberOfSenses|Origine|Possible",
                "",
                "#Text=Program je održan u skladu sa novim konceptom.",
                "1-1\t0-7\tProgram\tN\tNOUN\t_\t_\tprogram\t_\t_\t_\t*\tObjašnjenje\t"
                "http://llod.jerteh.rs/WSD/seENG3000551215n\t2\tgpt-4.1\tENG30-00551215-n;ENG30-06676416-n",
                "1-2\t8-10\tje\tV\tAUX\t_\t_\tjesam\t_\t_\t_\t_\t_\t_\t_\t_\t_",
                "1-3\t11-17\todržan\tV\tVERB\t_\t_\todržati\t_\t_\t_\t*\tObjašnjenje\t"
                "http://llod.jerteh.rs/WSD/seENG3001733477v\t2\tgpt-4.1\tENG30-01733477-v;ENG30-02280132-v",
                "",
            ]
        ),
    )

    sentences = parse_webanno_tsv(fixture)

    assert len(sentences) == 1
    assert sentences[0].text == "Program je održan u skladu sa novim konceptom."
    assert sentences[0].tokens[0].lemma == "program"
    assert sentences[0].tokens[0].sense_id == "ENG30-00551215-n"
    assert sentences[0].tokens[0].candidate_ids == ["ENG30-00551215-n", "ENG30-06676416-n"]


def test_mark_span_matches_lexisense_marker_style() -> None:
    assert mark_span("Program je održan.", [(11, 17)]) == "Program je **održan**."
    assert mark_span("Program je u skladu sa planom.", [(11, 12), (13, 19), (20, 22)]) == (
        "Program je **u** **skladu** **sa** planom."
    )


def test_gold_split_files_are_fixed_by_filename(tmp_path: Path) -> None:
    for name in [
        "sr-elexis-WSD_0001_0120-gold.tsv",
        "sr-elexis-WSD_0301-0400-gold.tsv",
        "sr-elexis-WSD_0401-0500-gold.tsv",
        "sr-elexis-WSD_0501-0600-gold.tsv",
    ]:
        (tmp_path / name).write_text("", encoding="utf-8")

    assert [p.name for p in gold_files_for_split(tmp_path, "dev")] == ["sr-elexis-WSD_0001_0120-gold.tsv"]
    assert [p.name for p in gold_files_for_split(tmp_path, "final")] == [
        "sr-elexis-WSD_0301-0400-gold.tsv",
        "sr-elexis-WSD_0401-0500-gold.tsv",
        "sr-elexis-WSD_0501-0600-gold.tsv",
    ]
