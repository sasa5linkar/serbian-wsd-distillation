# Serbian WSD Distillation

Prepare context–definition pairs from silver WebAnno annotations, train SentenceTransformer WSD rankers, and evaluate selected senses against gold annotations. This is the public software companion to the distillation experiments in sections 3.4 and 5.8 of Saša Petalinkar's doctoral dissertation.

**Version 0.1.0** includes the implementation, tests, reported experiment settings and summary results, and a complete offline example. The Python distribution remains `lexisense-wsd-distillation` and the command remains `lexisense-distill`.

## Install and try the offline example

Python 3.10+ is required; tested on Python 3.12. In an activated virtual environment:

```bash
git clone --branch v0.1.0 https://github.com/sasa5linkar/serbian-wsd-distillation.git
cd serbian-wsd-distillation
python -m pip install .
python examples/run_demo.py
```

The example uses a synthetic sense inventory and annotations in [examples](examples). It removes both gold sentences from four silver sentences and produces **four pairs** (two positive and two negative). Four saved toy predictions give **3/4 coverage**, **2/3 covered top-1 accuracy**, and **2/4 strict accuracy**. Outputs go to `outputs/demo`. These are format demonstrations, not research results. No training, model download, GPU or API call is involved.

## Research inputs and split

- **Inventory:** CSV/TSV with `sense_id, lemma, upos, definition` (aliases `senseID` and `gloss` are supported), or an XLSX first sheet with those columns. Keep the inventory compatible with the annotation sense IDs. The [second-round public IDA inventory](https://github.com/te-sla/A-Semi-Automated-LLM-Based-Framework-for-Word-Sense-Disambiguation-in-Serbian/blob/main/Data/Elexis-WSD-Repo-sr-v2.xlsx) provides the related research resource; the first-round inventory is a different version.
- **Silver:** WebAnno TSV 3.3 exports with token offsets, lemma/UPOS, optional MWE fields, the selected KBid and candidate list. The fixed column layout is described in [data formats](docs/data_formats.md).
- **Gold:** download the [public IDA gold_eval directory](https://github.com/te-sla/A-Semi-Automated-LLM-Based-Framework-for-Word-Sense-Disambiguation-in-Serbian/tree/main/Data/gold_eval). Development uses sentences **0001–0120**. Final evaluation uses **0301–0600**, in three 100-sentence files. Keep the original filenames. All gold sentence texts, including final-set texts, are excluded from silver training.

## Prepare, train and evaluate

Commands use paths relative to your working directory; replace them with your own inputs.

```bash
lexisense-distill build-dataset --silver-dir data/silver --gold-dir data/gold --sense-repo data/senses.tsv --out data/processed/silver_pairs.jsonl --coverage-out outputs/coverage_errors.jsonl
```

The training extra is only needed when running models:

```bash
python -m pip install ".[train]"
lexisense-distill early-stop-presets --pairs data/processed/silver_pairs.jsonl --gold-dir data/gold --sense-repo data/senses.tsv --preset mling --models-dir models --outputs-dir outputs/early_stopping --max-epochs 10 --patience 2 --dry-run
```

Inspect the plan, then omit `--dry-run` to train. Development accuracy selects the checkpoint; the final split is for reporting. `train` is a separate basic single-model command; its default e5-base checkpoint is **not** the mling experiment preset. Use `early-stop-presets` to select the documented preset explicitly.

```bash
lexisense-distill evaluate --gold-dir data/gold --split final --sense-repo data/senses.tsv --model models/wsd-distilled-mling-best --text-prefix "query: " --normalize-embeddings --out outputs/final_eval
```

The generic gold evaluator selects units from WebAnno annotations. To evaluate the **1,921-unit second-round workbook universe**, use the dedicated command with an `All_models_wide` workbook:

```bash
lexisense-distill evaluate-raw-round2 --raw-round2-xlsx data/Phase4_evaluacija_round2_300.xlsx --gold-dir data/gold --sense-repo data/senses.tsv --model models/wsd-distilled-mling-best --text-prefix "query: " --normalize-embeddings --out outputs/round2_eval
```

Both commands write metrics, predictions and errors. The existing `top1_accuracy` field is **covered-only accuracy**. Strict accuracy counts uncovered units as wrong: correct top-1 predictions divided by all evaluation units. Report coverage alongside both measures. See [evaluation and results](docs/evaluation.md).

## Reported experiment

| Preset | Base model | Prefix | Normalize | Batch | Selected epoch |
|---|---|---|---|---:|---:|
| simple | all-MiniLM-L6-v2 | none | no | 32 | 5 |
| tesla | te-sla/TeslaXLM | none | no | 16 | 4 |
| mling | intfloat/multilingual-e5-large | query: (with trailing space) | yes | 4 | 7 |

| Model | Raw correct / 1,921 | Trained correct / 1,921 | Trained strict accuracy |
|---|---:|---:|---:|
| simple | 1,045 | 1,315 | 68.5% |
| tesla | 1,026 | 986 | 51.3% |
| mling | 1,138 | 1,394 | 72.6% |

[configs/reported_experiment.json](configs/reported_experiment.json) stores the reported settings. [results](results/README.md) contains exported summary tables; these are separate from the toy example. Trained checkpoints are not bundled in this release. Training and inference accept local model directories. The released weights are public as [E5 Large / mling](https://huggingface.co/Tanor/serbian-wsd-distilled-e5-large), [MiniLM / simple](https://huggingface.co/Tanor/serbian-wsd-distilled-minilm), and [TeslaXLM / tesla](https://huggingface.co/Tanor/serbian-wsd-distilled-teslaxlm). The [Hugging Face guide](docs/huggingface_models.md) provides pinned downloads, model cards, sizes, and licenses. The 24 sentiment classifiers linked below perform polarity classification and are not these WSD rankers.

## Citation and license

Cite the software using [CITATION.cff](CITATION.cff). The related IDA framework is described in the [published article](https://doi.org/10.1177/1088467X261469292); the distillation summaries here accompany the dissertation and should not be attributed to that article as new published results.

Code, documentation and synthetic examples are distributed under [Apache-2.0](LICENSE). External lexical data, annotations and pretrained model weights retain their own terms. This release does not relicense or redistribute model weights or the full lexical inventory.

## Related resources

- [Sentiment lexicons S1–S7 and 24 Hugging Face classifiers](https://github.com/sasa5linkar/Serbian-WordNet-Sentiment-Lexicon-Analysis)
- [Synthetic sentiment evaluation set](https://github.com/sasa5linkar/SWN-synth-eval-set)
- [IDA WSD data and research code](https://github.com/te-sla/A-Semi-Automated-LLM-Based-Framework-for-Word-Sense-Disambiguation-in-Serbian)
- [WordNet expansion experiments](https://github.com/sasa5linkar/wordnet_autotranslate-)

- [srpskiwn Python wrapper](https://github.com/sasa5linkar/srpskiwn)

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Core tests and the offline example do not require the training extra.
