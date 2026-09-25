# Evaluation protocol

The reported final universe contains **1,921 targets: 1,734 single-word targets and 187 multiword expressions**, from the 300 final sentences (0301–0600). Development uses 120 separate sentences (0001–0120).

The workbook evaluator selects rows in All_models_wide with a Gold_Sense, content-word Upos (NOUN/VERB/ADJ/ADV) and empty or underscore NER. It does not require the gold ID to be in Possible. The generic WebAnno evaluator uses its own target filters; its unit count should not be substituted for the workbook universe.

A single candidate is selected directly. Ambiguous targets require the candidate definitions for ranking. The trained models cover 1,894 of 1,921 reported targets; the 27 uncovered targets count as wrong in strict accuracy.

- **Strict top-1:** correct top-1 predictions / all targets.
- **Covered top-1:** correct top-1 predictions / covered targets; this is the existing compute_metrics top1_accuracy field.
- **Coverage:** covered targets / all targets.
- **Top-3 and MRR:** computed on covered targets.

[reported_summary.csv](../results/reported_summary.csv) preserves correct and denominator counts; [reported_by_target.csv](../results/reported_by_target.csv) separates single words from multiword expressions. The paired raw-vs-trained tests in [reported_paired_tests.csv](../results/reported_paired_tests.csv) use the same 1,921 units. n01 counts examples fixed by training and n10 counts examples worsened by training; p-values have Holm correction across the three comparisons.

These files export the experiment's recorded aggregate results. They are not recalculations from the synthetic example. This software release contains summary tables; full per-target research predictions and trained weights are separate resources.
