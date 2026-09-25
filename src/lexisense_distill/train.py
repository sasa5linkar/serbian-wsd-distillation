from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from .dataset import TrainingPair, read_jsonl
from .ssl_utils import prefer_certifi_default_context

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"
SMOKE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class TrainingExample:
    def __init__(self, texts: list[str], label: float) -> None:
        self.texts = texts
        self.label = label


def with_text_prefix(text: str, text_prefix: str) -> str:
    if not text_prefix:
        return str(text)
    text = str(text)
    return text if text.startswith(text_prefix) else f"{text_prefix}{text}"


def build_input_examples(pairs: Iterable[TrainingPair], *, text_prefix: str = "") -> list[TrainingExample]:
    return [
        TrainingExample(
            texts=[
                with_text_prefix(pair.text, text_prefix),
                with_text_prefix(pair.definition, text_prefix),
            ],
            label=float(pair.label),
        )
        for pair in pairs
    ]


def seed_training(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass


def train_sentence_transformer(
    pairs_path: Path | str,
    output_dir: Path | str,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 1,
    batch_size: int = 16,
    warmup_steps: int = 100,
    text_prefix: str = "",
    seed: int | None = None,
) -> None:
    prefer_certifi_default_context()
    seed_training(seed)
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install with `pip install -e .[train]` to train the model") from exc

    pairs = read_jsonl(pairs_path)
    examples = [
        InputExample(texts=example.texts, label=example.label)
        for example in build_input_examples(pairs, text_prefix=text_prefix)
    ]
    if not examples:
        raise ValueError("No training pairs found.")

    model = SentenceTransformer(model_name)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=batch_size,
        generator=generator if seed is not None else None,
    )
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    (output / "training_config.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "warmup_steps": warmup_steps,
                "text_prefix": text_prefix,
                "seed": seed,
                "pairs_path": str(Path(pairs_path)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
