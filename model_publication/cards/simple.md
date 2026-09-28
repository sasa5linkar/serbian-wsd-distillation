---
language:
- sr
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: sentence-transformers/all-MiniLM-L6-v2
base_model_relation: finetune
license: apache-2.0
tags:
- wsd
- word-sense-disambiguation
- serbian
- knowledge-distillation
- sentence-transformers
- sentence-similarity
---

# Serbian WSD Distilled MiniLM

This is a Sentence Transformers bi-encoder adapted by **[Saša Petalinkar](https://orcid.org/0009-0007-9664-3594)** from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for **Serbian word-sense candidate ranking**. Given a sentence context and externally supplied candidate definitions, it ranks the candidates by cosine similarity. Its experiment preset is `simple`.

This model accompanies Saša Z. Petalinkar’s unpublished doctoral dissertation manuscript, submitted for evaluation at the University of Belgrade in 2026. The methodology and reported results are described in Sections 3.4 and 5.8. See [Thesis and citation](#thesis-and-citation) for the full reference.

## Model and intended use

| Property | Value |
|---|---|
| Architecture | Transformer encoder with mean pooling |
| Output dimensions | 384 |
| Saved maximum sequence length | 256 tokens |
| Training objective | CosineSimilarityLoss with mean squared error |
| Selected epoch | 5 |
| Stored weight dtype | float32 |
| Application prefix | None |

Includes a final Normalize module; the evaluation encoding flag is `normalize_embeddings=False`.

The intended use is research on Serbian candidate-sense ranking. Target detection, candidate generation, definitions, and sense-ID normalization are external to these weights. The model does not generate definitions or explanations. Cosine scores are ranking scores, not calibrated probabilities. General retrieval, other languages, other Serbian domains, and script-specific performance have not been established by the retained evaluation.

## Related models and software

All three distilled WSD rankers are public on the Tanor account. The experiment preset names remain unchanged in the companion code. Sizes below refer to `model.safetensors` only (decimal MB).

| Experiment preset | Public model | Weight file | License |
|---|---|---:|---|
| `mling` | [Tanor/serbian-wsd-distilled-e5-large](https://huggingface.co/Tanor/serbian-wsd-distilled-e5-large) | 2239.61 MB | MIT |
| `simple` | [Tanor/serbian-wsd-distilled-minilm](https://huggingface.co/Tanor/serbian-wsd-distilled-minilm) | 90.86 MB | Apache-2.0 |
| `tesla` | [Tanor/serbian-wsd-distilled-teslaxlm](https://huggingface.co/Tanor/serbian-wsd-distilled-teslaxlm) | 2239.61 MB | CC BY-SA 4.0 |

- [WSD distillation software and pinned downloads](https://github.com/sasa5linkar/serbian-wsd-distillation/blob/main/docs/huggingface_models.md): training and evaluation workflow, reported settings, and result summaries.
- [Serbian WordNet Sentiment Toolkit](https://github.com/sasa5linkar/serbian-wordnet-sentiment-toolkit/blob/main/docs/model-resources.md): optional local WSD in a reusable sentiment pipeline, plus a lightweight example without model downloads.
- [Serbian Sentiment WSD Evaluation](https://github.com/sasa5linkar/serbian-sentiment-wsd-evaluation#wsd-checkpoint-publication): research evaluation with WSD-selected senses and sentiment lexicons.

The WSD weights rank candidate definitions; the [24 sentiment classifiers](https://github.com/sasa5linkar/Serbian-WordNet-Sentiment-Lexicon-Analysis) are separate resources. Obtain the sense inventory and sentiment lexicons separately under their own terms.

## Usage

The example pins a revision containing the released weights and runtime configuration.

Install the recorded Sentence Transformers and Transformers versions:

```bash
pip install sentence-transformers==5.5.0 transformers==5.8.1
```

This example uses two of the six records for the Serbian lemma `list` in the retained LexiSense-SR inventory. `ENG30-06255777-n` and `ENG30-13152742-n` are the real sense IDs for the paper and plant senses, respectively. The context is synthetic and the short Serbian definitions below are illustrative paraphrases, not verbatim inventory entries. Only the context and definitions are encoded by the model; the selected sense ID is returned, as in the original implementation. The `ENG30` prefix is part of the inventory identifier, not the language of the definition. This shortened example does not perform candidate generation.

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer(
    "Tanor/serbian-wsd-distilled-minilm",
    revision="25d55084559c1d4d87ffea16699ff23367ef282b",
)
context = "Na stolu je ležao **list** papira."
candidates = [
    ("ENG30-06255777-n", "tanak komad papira"),
    ("ENG30-13152742-n", "deo biljke koji raste na stablu ili grani"),
]
prefix = ""
texts = [prefix + context] + [prefix + definition for _, definition in candidates]
embeddings = model.encode(
    texts,
    convert_to_tensor=True,
    normalize_embeddings=False,
)
scores = util.cos_sim(embeddings[0], embeddings[1:])[0]
best = int(scores.argmax().item())
print(candidates[best][0])
```

The experiment adds no query or passage prefix to either input. For normal inference, use sentence-relative character offsets to surround the intended target with `**...**`. Return the only candidate directly for a one-candidate item; for ambiguous items, all candidate definitions are required by the recorded evaluation protocol.

## Training and checkpoint selection

Training uses hard silver sense assignments from the LexiSense annotation pipeline. A context-definition pair has label 1 for the assigned sense and label 0 for other listed candidates. Candidates lacking definitions are skipped. Despite the experiment's “distillation” name, the implemented objective is hard-label cosine regression, without teacher logits, soft-label matching, temperature scaling, or KL divergence.

The retained training-pair file contains **38,829 pairs**: 10,385 positive and 28,444 negative, covering 1,604 distinct sentence strings and five source-file values. All designated gold sentence texts were excluded by exact matching. A publication audit found zero exact overlaps with the 420 distinct sentence strings in the four retained gold files. This does not establish absence of near-duplicates or other leakage. The inspected training-file hash is recorded in the packaged provenance; see the archive inspection note below.

| Recorded setting | Value |
|---|---|
| Batch size | 32 |
| Selected epoch / epochs run | 5 / 7 |
| Maximum epochs / patience | 10 / 2 |
| Warmup steps | 0 |
| Run seed / selected epoch seed | 20260515 / 20260519 |
| Selected development top-1 | 491/918 (53.49%), covered-only |
| Development universe | 918 covered / 922 total |

Selection maximizes development covered-only top-1 accuracy; final evaluation follows selection. Consecutive epochs are separate one-epoch training calls that load the preceding model weights and recreate the optimizer and scheduler. A saved `epochs: 1` field describes the last call, not the entire training history.

Generated trainer metadata records learning rate `5e-5`, linear scheduling, `adamw_torch_fused`, weight decay 0, and no fp16/bf16. It also records trainer seed 42, separately from the orchestration seeds above; full deterministic reproducibility is not claimed. Development target counts differ across the three released presets, so their saved development scores are not a controlled comparison on identical examples.

Recorded software: Python 3.11.9, Sentence Transformers 5.5.0, Transformers 5.8.1, PyTorch 2.11.0+cu128, Accelerate 1.13.0, Datasets 4.8.5, and Tokenizers 0.22.2.

## Results reported in the dissertation

The dissertation (Section 5.8, Table 5-16) reports **1,315/1,921 correct (68.5% strict top-1)** for the `simple` trained experiment. The retained experiment summary records 1,894 covered targets and 69.43% covered-only accuracy. The final universe comprises 1,734 single-word targets and 187 multiword expressions. Strict accuracy counts the 27 uncovered targets as wrong; the retained report attributes them to missing candidate definitions.

These figures are the results reported in the dissertation and agree with the retained experiment summary. The selected epochs and development scores in the saved training records agree with the experiment handoff. The full evaluation was not rerun for this Hub release; the release check below verifies model loading and a small inference example.

The workbook evaluator adds the gold sense when it is absent from the supplied candidate set. Accordingly, these results measure ranking with a **gold-augmented candidate inventory**, not end-to-end candidate recall. One-candidate cases are answered deterministically without the encoder. These protocol choices matter when comparing other systems.

## Publication runtime check

A local CPU smoke test on 2026-09-27 loaded these exact weights and executed the Python example above. It produced finite embeddings of the expected dimensions. The example selected the paper sense. See [runtime_verification.json](https://huggingface.co/Tanor/serbian-wsd-distilled-minilm/blob/4a08bf97fa51d769ab2cbfa64dc522e005b12337/runtime_verification.json) for the observed scores. This is a three-text technical check, not an accuracy evaluation or a replication of the historical results.

## Archive inspection note

In the retained `silver_pairs.jsonl`, 38,727 of 38,829 pairs place the target markers at the end of the sentence rather than around the target span; 102 pairs contain a nonempty `**...**` span. The original checkpoint cards show the same format in their training examples, and their training configurations refer to this pair file. The exact count describes the retained file inspected during release preparation. The effect of this marking format on model performance was not separately measured.

The retained Round 2 evaluation summary does not record weight-file checksums. The checksums in this release identify the uploaded files; the research results above are attributed to the dissertation and its experiment records.

## Provenance, attribution, and license

The weights were adapted by **[Saša Petalinkar](https://orcid.org/0009-0007-9664-3594)**. Credit for the base model and its original authors belongs to the contributors identified in the [upstream model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). This adaptation is distributed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0); see the included license files and preserve upstream attribution.

[Companion implementation and reported settings, v0.1.0](https://github.com/sasa5linkar/serbian-wsd-distillation/tree/v0.1.0) document the workflow. That repository is the public software companion to the dissertation experiments.

See [provenance.json](https://huggingface.co/Tanor/serbian-wsd-distilled-minilm/blob/4a08bf97fa51d769ab2cbfa64dc522e005b12337/provenance.json) and [recorded_training_summary.json](https://huggingface.co/Tanor/serbian-wsd-distilled-minilm/blob/4a08bf97fa51d769ab2cbfa64dc522e005b12337/recorded_training_summary.json) for packaged metadata. The released `model.safetensors` SHA-256 is:

```text
d6c1624774330b2ef392d90c9b0dbf9b9ccaf5a3a60cf645fa9385ca7f3c475d
```

The checksum identifies the released weight file. Training corpora and external definition inventories are not included in this model repository.

## Thesis and citation

[Saša Z. Petalinkar](https://orcid.org/0009-0007-9664-3594) (2026). *Machine Learning and Large Language Models in the Development of Semantic Networks and Their Application in Automatic Text Understanding*. Unpublished doctoral dissertation manuscript, submitted for evaluation at the University of Belgrade. Written in Serbian.

Original title: *Машинско учење и велики језички модели у развоју семантичких мрежа и њиховој примени на аутоматско разумевање текста*.

- **Section 3.4, „Разрешење значења речи – дестилација“**: distillation method and experimental setup.
- **Section 5.8, „Резултати дестилације модела за разрешење значења речи“, Table 5-16**: reported results for the `simple`, `tesla`, and `multilingual-e5-large` rankers. The last is called `mling` in the experiment code. Table 5-17 reports the paired significance tests.
- **Section 5.6, Table 5-10**: mapping of the dissertation's model names to the pretrained checkpoints.

The manuscript does not currently have a public URL or DOI. The [public software companion](https://github.com/sasa5linkar/serbian-wsd-distillation/tree/v0.1.0) contains the implementation, reported settings, and result summaries.

```bibtex
@unpublished{petalinkar2026semanticnetworks,
  author = {Petalinkar, Saša Z.},
  title = {Машинско учење и велики језички модели у развоју семантичких мрежа и њиховој примени на аутоматско разумевање текста},
  year = {2026},
  note = {Unpublished doctoral dissertation manuscript, submitted for evaluation at the University of Belgrade. Sections 3.4 and 5.8; Table 5-16},
  language = {Serbian}
}
```
