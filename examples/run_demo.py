"""Create synthetic training pairs and score saved toy predictions offline."""
import argparse
import json
from pathlib import Path
from lexisense_distill.dataset import build_training_pairs_from_dirs, write_jsonl
from lexisense_distill.sense_repo import load_sense_definitions
from lexisense_distill.evaluate import RankedPrediction, compute_metrics, write_metrics, write_predictions

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
args = parser.parse_args()
source = Path(__file__).resolve().parent
inventory = load_sense_definitions(source / "sense_inventory.tsv")
result = build_training_pairs_from_dirs(source / "silver", source / "gold", inventory)
write_jsonl(result.pairs, args.out / "pairs.jsonl")
rows = [RankedPrediction(**json.loads(line)) for line in (source / "saved_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
metrics = compute_metrics(rows)
correct = sum(row.covered and row.ranked_sense_ids[:1] == [row.gold_sense_id] for row in rows)
metrics["strict_top1_accuracy"] = correct / len(rows) if rows else 0.0
write_metrics(metrics, args.out / "metrics.json")
write_predictions(rows, args.out / "predictions.tsv")
print(json.dumps({"demo_only": True, "pair_counts": result.stats, "metrics": metrics}, ensure_ascii=False, indent=2))
