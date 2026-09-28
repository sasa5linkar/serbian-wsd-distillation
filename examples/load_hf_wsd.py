"""Optional local WSD example. Network access requires --download."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Same pinned releases as serbian-wsd-distillation/model_publication/models.json.
MODELS = {
    "mling": {
        "repo_id": "Tanor/serbian-wsd-distilled-e5-large",
        "revision": "749f694999b256039011acfb8f0a4b1b4f388c8c",
    },
    "simple": {
        "repo_id": "Tanor/serbian-wsd-distilled-minilm",
        "revision": "4a08bf97fa51d769ab2cbfa64dc522e005b12337",
    },
    "tesla": {
        "repo_id": "Tanor/serbian-wsd-distilled-teslaxlm",
        "revision": "c907e07a488205ad6c5779b5d7211cddec08d8f6",
    },
}


def resolve_checkpoint(
    preset: str, *, download: bool = False, local_dir: Path | None = None
) -> Path:
    """Use local files by default; fetch a pinned release only on request."""
    if local_dir is not None and not download:
        if not local_dir.is_dir():
            raise FileNotFoundError(f"No checkpoint at {local_dir}; add --download to fetch it.")
        return local_dir
    from huggingface_hub import snapshot_download

    spec = MODELS[preset]
    return Path(
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            local_dir=local_dir,
            local_files_only=not download,
        )
    )


def rank_example(checkpoint: Path) -> list[dict[str, str | float]]:
    """Rank illustrative definitions using the saved prefix and cosine similarity."""
    from sentence_transformers import SentenceTransformer, util

    config = json.loads((checkpoint / "training_config.json").read_text(encoding="utf-8"))
    prefix = config["text_prefix"]
    if not isinstance(prefix, str):
        raise ValueError("training_config.json text_prefix must be a string")
    model = SentenceTransformer(str(checkpoint), device="cpu", local_files_only=True)
    context = "Na stolu je ležao **list** papira."
    candidates = [
        ("ENG30-06255777-n", "tanak komad papira"),
        ("ENG30-13152742-n", "deo biljke koji raste na stablu ili grani"),
    ]
    texts = [context, *(definition for _, definition in candidates)]
    texts = [text if text.startswith(prefix) else prefix + text for text in texts]
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
    scores = util.cos_sim(embeddings[0], embeddings[1:])[0].tolist()
    ranked = [
        {"sense_id": sense_id, "definition": definition, "cosine": float(score)}
        for (sense_id, definition), score in zip(candidates, scores, strict=True)
    ]
    return sorted(ranked, key=lambda item: float(item["cosine"]), reverse=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="mling")
    parser.add_argument(
        "--list", action="store_true", help="List releases; no extra packages needed"
    )
    parser.add_argument("--download", action="store_true", help="Allow an explicit Hub download")
    parser.add_argument("--local-dir", type=Path, help="Read/download a local checkpoint directory")
    parser.add_argument(
        "--download-only", action="store_true", help="Download without running inference"
    )
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(MODELS, ensure_ascii=False, indent=2))
        return
    if args.download_only and not args.download:
        parser.error("--download-only requires --download")
    try:
        checkpoint = resolve_checkpoint(
            args.model, download=args.download, local_dir=args.local_dir
        )
        print(f"Local checkpoint: {checkpoint.resolve()}")
        if not args.download_only:
            print(json.dumps(rank_example(checkpoint), ensure_ascii=False, indent=2))
    except ImportError as exc:
        parser.exit(1, f"Missing optional dependency: {exc}. See the README example commands.\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Cannot load the checkpoint: {exc}\n")


if __name__ == "__main__":
    main()
