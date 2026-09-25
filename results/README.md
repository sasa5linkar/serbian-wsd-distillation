# Reported experiment summaries

These CSV files transcribe the recorded distillation results accompanying dissertation section 5.8. They are published aggregate tables, not output from the offline demonstration and not a newly rerun experiment.

- reported_summary.csv: raw/trained counts, full denominator, coverage and accuracies.
- reported_by_target.csv: single-word and multiword counts and error transitions.
- reported_paired_tests.csv: paired raw-to-trained changes and Holm-adjusted McNemar p-values.

The fractions in reported_summary.csv are calculated from the recorded integer counts (eight decimal places). Research per-target predictions and model weights are not included in this release. See ../docs/evaluation.md for evaluation definitions and ../configs/reported_experiment.json for the reported settings.
