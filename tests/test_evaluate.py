from __future__ import annotations

from lexisense_distill.evaluate import RankedPrediction, compute_metrics


def test_compute_ranking_metrics_topk_mrr_and_coverage() -> None:
    predictions = [
        RankedPrediction("s1", "reč", "GOLD-A", ["GOLD-A", "B", "C"], True),
        RankedPrediction("s2", "reč", "GOLD-B", ["A", "GOLD-B", "C"], True),
        RankedPrediction("s3", "reč", "GOLD-C", ["A", "B", "C"], True),
        RankedPrediction("s4", "reč", "GOLD-D", [], False),
    ]

    metrics = compute_metrics(predictions)

    assert metrics["total_examples"] == 4
    assert metrics["covered_examples"] == 3
    assert metrics["coverage"] == 0.75
    assert metrics["top1_accuracy"] == 1 / 3
    assert metrics["top3_accuracy"] == 2 / 3
    assert metrics["mrr"] == (1 + 0.5 + 0) / 3
