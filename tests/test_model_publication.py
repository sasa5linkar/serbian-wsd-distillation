import importlib.util
import json
from pathlib import Path
import pytest

SCRIPT=Path(__file__).parents[1]/"scripts/publish_wsd_model.py"
spec=importlib.util.spec_from_file_location("publish_wsd_model",SCRIPT)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def checkpoint(tmp_path):
    source=tmp_path/"source"
    source.mkdir()
    for name,data in {"config.json":{"_name_or_path":"private/local/path"},"modules.json":[{"type":"sentence_transformers.models.Transformer","path":""}],"training_config.json":{"text_prefix":"query: ","pairs_path":"private/corpus.jsonl"}}.items():
        (source/name).write_text(json.dumps(data))
    (source/"tokenizer.json").write_text("{}")
    (source/"model.safetensors").write_bytes(b"synthetic packaging fixture, not a loadable model")
    (source/"optimizer.pt").write_bytes(b"excluded")
    (source/"corpus.jsonl").write_text("excluded")
    return source

def test_prepare_preserves_weights_and_excludes_training_files(tmp_path):
    source=checkpoint(tmp_path)
    package=tmp_path/"package"
    before=module.digest(source/"model.safetensors")
    module.prepare("mling",source,package)
    assert module.digest(package/"model.safetensors")==before
    assert not (package/"optimizer.pt").exists()
    assert not (package/"corpus.jsonl").exists()
    assert "pairs_path" not in json.loads((package/"training_config.json").read_text())
    assert "pairs_path" in json.loads((source/"training_config.json").read_text())
    assert "_name_or_path" not in json.loads((package/"config.json").read_text())
    module.check_manifest(package)
    with pytest.raises(ValueError,match="never overwritten"):
        module.prepare("mling",source,package)
    (package/"tokenizer.json").write_text("changed")
    with pytest.raises(ValueError,match="differs"):
        module.check_manifest(package)

def test_prepare_rejects_wrong_preset_and_missing_weights(tmp_path):
    source=checkpoint(tmp_path)
    with pytest.raises(ValueError,match="text_prefix"):
        module.prepare("simple",source,tmp_path/"wrong")
    (source/"model.safetensors").unlink()
    with pytest.raises(ValueError,match="weights"):
        module.prepare("mling",source,tmp_path/"empty")
