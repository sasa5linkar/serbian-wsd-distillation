"""Network and input-format checks for the optional Hugging Face example."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "load_hf_wsd.py"
spec = importlib.util.spec_from_file_location("load_hf_wsd_example", SCRIPT)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_listing_needs_no_model_libraries(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    example.main(["--list"])
    assert set(json.loads(capsys.readouterr().out)) == {"mling", "simple", "tesla"}


@pytest.mark.parametrize("download", [False, True])
@pytest.mark.parametrize("preset", ["mling", "simple", "tesla"])
def test_download_is_explicit_and_revision_is_pinned(monkeypatch, tmp_path, preset, download):
    snapshot = Mock(return_value=str(tmp_path))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    assert example.resolve_checkpoint(preset, download=download) == tmp_path
    kwargs = snapshot.call_args.kwargs
    assert kwargs["local_files_only"] is (not download)
    assert kwargs["repo_id"] == example.MODELS[preset]["repo_id"]
    assert kwargs["revision"] == example.MODELS[preset]["revision"]
    assert len(kwargs["revision"]) == 40


def test_local_directory_needs_no_hub_and_missing_directory_does_not_download(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert example.resolve_checkpoint("mling", local_dir=tmp_path) == tmp_path
    with pytest.raises(FileNotFoundError, match="--download"):
        example.resolve_checkpoint("mling", local_dir=tmp_path / "missing")


@pytest.mark.parametrize("prefix", ["query: ", ""])
def test_saved_prefix_applies_to_context_and_definitions(monkeypatch, tmp_path, prefix):
    (tmp_path / "training_config.json").write_text(json.dumps({"text_prefix": prefix}))
    encoder = Mock()
    encoder.encode.return_value = [[1], [2], [3]]
    constructor = Mock(return_value=encoder)
    scores = SimpleNamespace(tolist=lambda: [0.1, 0.9])
    cosine = Mock(return_value=[scores])
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=constructor, util=SimpleNamespace(cos_sim=cosine)),
    )
    ranked = example.rank_example(tmp_path)
    constructor.assert_called_once_with(str(tmp_path), device="cpu", local_files_only=True)
    inputs = encoder.encode.call_args.args[0]
    assert inputs == [
        prefix + text
        for text in [
            "Na stolu je ležao **list** papira.",
            "tanak komad papira",
            "deo biljke koji raste na stablu ili grani",
        ]
    ]
    assert ranked[0]["sense_id"] == "ENG30-13152742-n"
    assert ranked[0]["cosine"] == 0.9


def test_download_only_requires_explicit_download():
    with pytest.raises(SystemExit) as exc:
        example.main(["--download-only"])
    assert exc.value.code == 2
