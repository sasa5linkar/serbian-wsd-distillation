---
language: sr
license: cc-by-sa-4.0
base_model: te-sla/TeslaXLM
base_model_relation: finetune
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- word-sense-disambiguation
- serbian
- distillation
---

# Serbian WSD distilled tesla

A context–definition ranker fine-tuned by Saša Z. Petalinkar for Serbian word-sense disambiguation. It ranks definitions of candidate senses with cosine similarity. It does not produce positive/negative sentiment labels.

## Training and evaluation

The reported experiment used [te-sla/TeslaXLM](https://huggingface.co/te-sla/TeslaXLM) as its base. Silver WebAnno annotations supply positive context–definition pairs and competing negative definitions. Training uses SentenceTransformers CosineSimilarityLoss. All gold sentence texts are excluded from silver training. Development uses 120 sentences (0001–0120); the final split uses 300 sentences (0301–0600). Development accuracy selects the checkpoint. Reported batch size: 16; selected epoch: 4.

The second-round final universe has 1,921 units: 1,734 single words and 187 multiword expressions. The trained configuration covers 1,894 units; 27 uncovered units count as wrong under strict accuracy.

| Reported measure | Value |
|---|---:|
| Raw base correct / all units | 1026 / 1,921 |
| Trained correct / all units | 986 / 1,921 |
| Trained strict accuracy | 51.3% |
| Trained covered-only accuracy | 52.06% |

These are the dissertation's reported results, not a fresh benchmark from the short upload verification. The software's `top1_accuracy` is covered-only. The trained Tesla configuration underperformed its raw base in this evaluation; fine-tuning did not improve that result.

## Local use

Download a pinned release revision using the [publication and download guide](https://github.com/sasa5linkar/serbian-wsd-distillation/blob/main/docs/huggingface_models.md). Then load the local folder:

~~~python
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

path = Path("models/wsd-distilled-tesla")
config = json.loads((path / "training_config.json").read_text(encoding="utf-8"))
prefix = config["text_prefix"]
texts = ["Он је љубазан.", "Онај који се предусретљиво односи према другима.", "Онај који се односи на временске прилике."]
prepared = [text if text.startswith(prefix) else prefix + text for text in texts]
model = SentenceTransformer(str(path), local_files_only=True)
vectors = model.encode(prepared, convert_to_tensor=True, normalize_embeddings=False)
scores = util.cos_sim(vectors[0], vectors[1:])[0]
print(scores.argsort(descending=True).tolist())
~~~

The example demonstrates the interface and makes no assertion about the winning sense. Apply the saved prefix to both context and definitions; for mling it is `query: ` with a trailing space. The saved SentenceTransformer configuration defines pooling and maximum sequence length. Ranking uses cosine similarity even when normalization at encoding is disabled.

## Limitations and resources

Candidate coverage and inventory version affect results. Silver annotations can contain errors. Whole-text ranking in an application is not the same as the target-aware research evaluation; use the [evaluation code](https://github.com/sasa5linkar/serbian-wsd-distillation) to reproduce its inputs and scoring. The checkpoint does not bundle the Serbian WordNet inventory or annotated corpora. No general-domain Serbian semantic or sentiment accuracy is claimed.

## Attribution, license and citation

Weights are released under cc-by-sa-4.0, with upstream attribution in `BASE_MODEL.md` and the license in `LICENSE`. The underlying [base model](https://huggingface.co/te-sla/TeslaXLM) retains its authorship and terms. The fine-tuning modifications are by Saša Z. Petalinkar. External corpora and lexical data retain their separate terms. Cite this model's repository and exact revision together with the [software citation](https://github.com/sasa5linkar/serbian-wsd-distillation/blob/main/CITATION.cff). The distillation results accompany the submitted doctoral dissertation; no thesis DOI is asserted.
