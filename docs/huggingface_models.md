# Published WSD models and downloads

All three distilled WSD checkpoints are public on Hugging Face. The names below are the actual model repositories; `mling`, `simple`, and `tesla` remain the experiment preset names. These rankers are separate from the 24 sentiment polarity classifiers.

| Experiment preset | Public model | Weight file | License |
|---|---|---:|---|
| `mling` | [Tanor/serbian-wsd-distilled-e5-large](https://huggingface.co/Tanor/serbian-wsd-distilled-e5-large) | 2239.61 MB | MIT |
| `simple` | [Tanor/serbian-wsd-distilled-minilm](https://huggingface.co/Tanor/serbian-wsd-distilled-minilm) | 90.86 MB | Apache-2.0 |
| `tesla` | [Tanor/serbian-wsd-distilled-teslaxlm](https://huggingface.co/Tanor/serbian-wsd-distilled-teslaxlm) | 2239.61 MB | CC BY-SA 4.0 |

Sizes are for the weight file only, in decimal MB; tokenizer and configuration files add to each download. Model repositories include their own licenses and upstream notices. Apache-2.0 for this software does not replace the model licenses.

## Download a released revision

The revisions below were checked on 2026-09-28 and include the model cards updated that day.

| Preset | Pinned revision |
|---|---|
| `mling` | [`749f694999b256039011acfb8f0a4b1b4f388c8c`](https://huggingface.co/Tanor/serbian-wsd-distilled-e5-large/tree/749f694999b256039011acfb8f0a4b1b4f388c8c) |
| `simple` | [`4a08bf97fa51d769ab2cbfa64dc522e005b12337`](https://huggingface.co/Tanor/serbian-wsd-distilled-minilm/tree/4a08bf97fa51d769ab2cbfa64dc522e005b12337) |
| `tesla` | [`c907e07a488205ad6c5779b5d7211cddec08d8f6`](https://huggingface.co/Tanor/serbian-wsd-distilled-teslaxlm/tree/c907e07a488205ad6c5779b5d7211cddec08d8f6) |

With the Hugging Face CLI installed, download the model you need:

~~~bash
hf download Tanor/serbian-wsd-distilled-e5-large --revision 749f694999b256039011acfb8f0a4b1b4f388c8c --local-dir models/wsd-distilled-mling
hf download Tanor/serbian-wsd-distilled-minilm --revision 4a08bf97fa51d769ab2cbfa64dc522e005b12337 --local-dir models/wsd-distilled-simple
hf download Tanor/serbian-wsd-distilled-teslaxlm --revision c907e07a488205ad6c5779b5d7211cddec08d8f6 --local-dir models/wsd-distilled-tesla
~~~

Each command downloads a complete local checkpoint. The local directory names retain the experiment presets even though the Hub names describe the base models. Model-specific Python examples, training settings, reported results, and citations are on the linked model cards. For `mling`, preserve the exact `query: ` prefix on both contexts and definitions; loading the SentenceTransformer alone does not add it.

## Optional runnable example

[examples/load_hf_wsd.py](../examples/load_hf_wsd.py) selects any of the three pinned releases, reads its saved prefix, and ranks two illustrative definitions. See the [example commands and model choices](../README.md#optional-hugging-face-example). It works with cached/local files by default; `--download` explicitly enables fetching a model. `--download-only` prepares a checkpoint for the existing application without running inference.

## Companion applications

- The [sentiment toolkit](https://github.com/sasa5linkar/serbian-wordnet-sentiment-toolkit/blob/main/docs/model-resources.md) documents an explicit E5 download into its default local resource directory and reads the saved prefix. Its lightweight example works without downloading weights.
- The [sentiment research evaluator](https://github.com/sasa5linkar/serbian-sentiment-wsd-evaluation#wsd-checkpoint-publication) accepts a complete local checkpoint directory and an external sense inventory.

The cards retain the dissertation results separately from the small CPU loading checks recorded during model publication on 2026-09-27. Those checks do not rerun the full historical evaluation. The TeslaXLM card retains the reported lack of improvement after distillation.

## Release records and maintenance

[model_publication/models.json](../model_publication/models.json) maps presets to public IDs, pinned revisions, weight sizes, and hashes. [model_publication/cards](../model_publication/cards) mirrors the current model-card text with absolute links to the packaged metadata. The base-card revisions in the manifest record a license check, not historical training revisions.

The existing [packaging helper](../scripts/publish_wsd_model.py) is retained for preparing separate checkpoint packages. Its upload/publish steps require a private staging repository and refuse to overwrite these already-public releases. No repackaging or training is needed to use the published models. Cards describe these specific checkpoints and must be reviewed before reuse for any different weights.
