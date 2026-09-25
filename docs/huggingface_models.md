# WSD model publication and download

The public software and reported results are available. The distilled checkpoint weights are **not yet released**. This page provides the procedure to publish existing final checkpoints; no model is retrained. The 24 polarity classifiers linked from the sentiment lexicon project are different models.

| Preset | Planned Hub repository | Base | Weight license |
|---|---|---|---|
| mling | `Tanor/serbian-wsd-distilled-mling` | [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | MIT |
| simple | `Tanor/serbian-wsd-distilled-simple` | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache-2.0 |
| tesla | `Tanor/serbian-wsd-distilled-tesla` | [TeslaXLM](https://huggingface.co/te-sla/TeslaXLM) | CC BY-SA 4.0 |

These names are publication targets, not working download links. [model_publication](../model_publication) holds card templates, license texts and reported settings. The base-card revisions in the manifest record a license check, not historical training revisions.

## On the computer holding the trained checkpoints

Use selected final directories (usually `models/wsd-distilled-<preset>-best`), with their saved tokenizer, SentenceTransformer modules and `training_config.json`. Do not substitute an untrained base-model cache. Create a local environment with the training dependencies and HF CLI:

~~~bash
uv sync --extra train --extra dev
uv run --extra train --with huggingface_hub hf auth login
uv run --extra train --with huggingface_hub python scripts/publish_wsd_model.py prepare --preset mling --source models/wsd-distilled-mling-best --destination model_packages/mling
uv run --extra train --with huggingface_hub python scripts/publish_wsd_model.py verify --source models/wsd-distilled-mling-best --package model_packages/mling
uv run --extra train --with huggingface_hub python scripts/publish_wsd_model.py upload --package model_packages/mling
uv run --extra train --with huggingface_hub python scripts/publish_wsd_model.py publish --package model_packages/mling
~~~

Repeat with `simple` and `tesla` when available. Authentication must identify Tanor and allow repository writes. Keep credentials out of files and command arguments.

`prepare` copies runtime files into a new directory, excludes training corpora/optimizer state/other epochs, records sizes and SHA-256 hashes, preserves upstream notices and removes local training paths from exported metadata. Source files are never modified. Inspect the selected checkpoint and card before upload.

`verify` compares source and packaged embeddings and candidate ordering on fixed Serbian inputs, including a multiword expression and a long input. It is a packaging check, not a new benchmark.

`upload` uses `hf upload-large-folder` in a private repository. Re-run the same upload after an interruption. `publish` downloads the exact uploaded revision, compares files and embeddings, creates `v1.0` and then makes the repository public. It writes a publication receipt beside the package. The procedure creates no paid endpoint or training job. See the [official upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload).

## Download a published revision

After publication, use the exact `repo_id` and `revision` from the receipt:

~~~bash
hf download REPO_ID --revision COMMIT_SHA --local-dir models/wsd-distilled-mling
~~~

Replace both placeholders with receipt values. Local inference consumes that directory. The [sentiment toolkit](https://github.com/sasa5linkar/serbian-wordnet-sentiment-toolkit) reads the saved mling `query: ` prefix and sequence length. Its lightweight example works before checkpoint publication.

## After publication

Add each receipt to `model_publication/releases/`, record its ID and revision in `model_publication/models.json` under `released_models`, and replace the planned entry above with its public link. Add the same pinned download to the sentiment evaluation and toolkit documentation. Do not mark a model released before its upload verification succeeds.
