from __future__ import annotations

from pathlib import Path

from lexisense_distill.dataset import DatasetBuildResult, build_training_pairs, collect_evaluation_targets, collect_gold_sentence_texts
from lexisense_distill.sense_repo import SenseDefinition
from lexisense_distill.webanno import parse_webanno_tsv


def write_fixture(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def webanno_doc(sentence: str, token_line: str) -> str:
    return "\n".join(
        [
            "#FORMAT=WebAnno TSV 3.3",
            "#T_SP=webanno.custom.WSD|Comment|Explanation|KBid|NumberOfSenses|Origine|Possible",
            "",
            f"#Text={sentence}",
            token_line,
            "",
        ]
    )


def test_collects_gold_sentence_texts_from_all_gold_files(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    write_fixture(gold_dir / "sr-elexis-WSD_0001_0120-gold.tsv", webanno_doc("Ovo je dev rečenica.", "1-1\t0-3\tOvo\tPRO\tPRON\t_\t_\tovo\t_\t_\t_\t_\t_\t_\t_\t_\t_"))
    write_fixture(gold_dir / "sr-elexis-WSD_0301-0400-gold.tsv", webanno_doc("Ovo je finalna rečenica.", "1-1\t0-3\tOvo\tPRO\tPRON\t_\t_\tovo\t_\t_\t_\t_\t_\t_\t_\t_\t_"))

    assert collect_gold_sentence_texts(gold_dir) == {"Ovo je dev rečenica.", "Ovo je finalna rečenica."}


def test_build_training_pairs_excludes_gold_sentences_and_reports_missing_definitions(tmp_path: Path) -> None:
    silver = tmp_path / "silver.tsv"
    write_fixture(
        silver,
        "\n".join(
            [
                "#FORMAT=WebAnno TSV 3.3",
                "#T_SP=webanno.custom.WSD|Comment|Explanation|KBid|NumberOfSenses|Origine|Possible",
                "",
                "#Text=Program je održan.",
                "1-1\t0-7\tProgram\tN\tNOUN\t_\t_\tprogram\t_\t_\t_\t*\tObjašnjenje\t"
                "http://llod.jerteh.rs/WSD/seENG3000551215n\t3\tgpt-4.1\tENG30-00551215-n;ENG30-06676416-n;MISSING-1",
                "",
                "#Text=Gold rečenica ne sme u trening.",
                "2-1\t0-4\tGold\tN\tNOUN\t_\t_\tgold\t_\t_\t_\t*\tObjašnjenje\t"
                "http://llod.jerteh.rs/WSD/seENG3000551215n\t2\tgpt-4.1\tENG30-00551215-n;ENG30-06676416-n",
                "",
            ]
        ),
    )
    sense_map = {
        "ENG30-00551215-n": SenseDefinition("ENG30-00551215-n", "program", "NOUN", "javna priredba"),
        "ENG30-06676416-n": SenseDefinition("ENG30-06676416-n", "program", "NOUN", "televizijski sadržaj"),
    }

    result = build_training_pairs([silver], sense_map, gold_sentence_texts={"Gold rečenica ne sme u trening."})

    assert isinstance(result, DatasetBuildResult)
    assert result.stats["skipped_gold_sentences"] == 1
    assert result.stats["positive_pairs"] == 1
    assert result.stats["negative_pairs"] == 1
    assert result.coverage_errors == [
        {
            "sentence": "Program je održan.",
            "target": "Program",
            "sense_id": "MISSING-1",
            "reason": "missing_definition",
        }
    ]
    assert [pair.label for pair in result.pairs] == [1.0, 0.0]
    assert result.pairs[0].text == "**Program** je održan."
    assert result.pairs[0].definition == "javna priredba"


def test_mwe_training_pair_marks_each_target_token(tmp_path: Path) -> None:
    silver = tmp_path / "mwe.tsv"
    write_fixture(
        silver,
        "\n".join(
            [
                "#FORMAT=WebAnno TSV 3.3",
                "#T_SP=webanno.custom.WSD|Comment|Explanation|KBid|NumberOfSenses|Origine|Possible",
                "",
                "#Text=Program je u skladu sa planom.",
                "1-1\t0-7\tProgram\tN\tNOUN\t_\t_\tprogram\t_\t_\t_\t_\t_\t_\t_\t_\t_",
                "1-2\t8-10\tje\tV\tAUX\t_\t_\tjesam\t_\t_\t_\t_\t_\t_\t_\t_\t_",
                "1-3\t11-12\tu\tPREP\tADP\t_\t_\tu\t1[187]\tu skladu sa[187]\tAdpID[187]\t*[450]\tObjašnjenje[450]\t"
                "http://llod.jerteh.rs/WSD/seLLM0288[450]\t1[450]\tgpt-4.1\tLLM-0288[450]",
                "1-4\t13-19\tskladu\tN\tNOUN\t_\t_\tsklad\t1[187]\tu skladu sa[187]\tAdpID[187]\t*[450]\tObjašnjenje[450]\t"
                "http://llod.jerteh.rs/WSD/seLLM0288[450]\t1[450]\tgpt-4.1\tLLM-0288[450]",
                "1-5\t20-22\tsa\tPREP\tADP\t_\t_\tsa\t1[187]\tu skladu sa[187]\tAdpID[187]\t*[450]\tObjašnjenje[450]\t"
                "http://llod.jerteh.rs/WSD/seLLM0288[450]\t1[450]\tgpt-4.1\tLLM-0288[450]",
                "",
            ]
        ),
    )
    sense_map = {"LLM-0288": SenseDefinition("LLM-0288", "u skladu sa", "ADP", "prema nečemu")}

    result = build_training_pairs([silver], sense_map, gold_sentence_texts=set())

    assert len(result.pairs) == 1
    assert result.pairs[0].text == "Program je **u** **skladu** **sa** planom."
    assert result.pairs[0].target == "u skladu sa"


def test_evaluation_targets_apply_lexisense_filters(tmp_path: Path) -> None:
    gold = tmp_path / "gold.tsv"
    write_fixture(
        gold,
        "\n".join(
            [
                "#FORMAT=WebAnno TSV 3.3",
                "#T_SP=webanno.custom.WSD|Comment|Explanation|KBid|NumberOfSenses|Origine|Possible",
                "",
                "#Text=Program je Evropski parlament u toku rada.",
                "1-1\t0-7\tProgram\tN\tNOUN\t_\t_\tprogram\t_\t_\t_\t_\t_\tENG30-00000001-n\t1\t_\tENG30-00000001-n",
                "1-2\t8-10\tje\tV\tAUX\t_\t_\tjesam\t_\t_\t_\t_\t_\tENG30-00000002-v\t1\t_\tENG30-00000002-v",
                "1-3\t11-19\tEvropski\tA\tADJ\tQ1\tORG\tevropski\t_\t_\t_\t_\t_\tENG30-00000003-a\t1\t_\tENG30-00000003-a",
                "1-4\t20-29\tparlament\tN\tNOUN\tQ1\tORG\tparlament\t_\t_\t_\t_\t_\tENG30-00000004-n\t1\t_\tENG30-00000004-n",
                "1-5\t30-31\tu\tPREP\tADP\t_\t_\tu\t1\tu toku\tAdvID\t_\t_\tLLM-0001\t1\t_\tLLM-0001",
                "1-6\t32-36\ttoku\tN\tNOUN\t_\t_\ttok\t1\tu toku\tAdvID\t_\t_\tLLM-0001\t1\t_\tLLM-0001",
                "1-7\t37-41\trada\tN\tNOUN\t_\t_\trad\t_\t_\t_\t_\t_\tENG30-00000005-n\t1\t_\tENG30-99999999-n",
                "",
                "#Text=Javna uprava radi.",
                "2-1\t0-5\tJavna\tA\tADJ\t_\t_\tjavan\t1\tjavna uprava\tNID\t_\t_\tCVMWE-0001\t1\t_\tCVMWE-0001",
                "2-2\t6-12\tuprava\tN\tNOUN\t_\t_\tuprava\t1\tjavna uprava\tNID\t_\t_\tCVMWE-0001\t1\t_\tCVMWE-0001",
                "2-3\t13-17\tradi\tV\tVERB\t_\t_\traditi\t_\t_\t_\t_\t_\tENG30-00000006-v\t1\t_\tENG30-00000006-v",
                "",
            ]
        ),
    )

    targets = collect_evaluation_targets(parse_webanno_tsv(gold))

    assert [(target.target, target.sense_id) for target in targets] == [
        ("Program", "ENG30-00000001-n"),
        ("rada", "ENG30-00000005-n"),
        ("javna uprava", "CVMWE-0001"),
        ("radi", "ENG30-00000006-v"),
    ]
