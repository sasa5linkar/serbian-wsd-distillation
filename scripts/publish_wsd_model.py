"""Prepare and publish existing final WSD checkpoints; never train a model."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "model_publication"
MODELS = json.loads((ASSETS / "models.json").read_text(encoding="utf-8"))["models"]
RUNTIME_FILES = {
    "config.json", "modules.json", "config_sentence_transformers.json",
    "sentence_bert_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.txt", "vocab.json",
    "merges.txt", "sentencepiece.bpe.model", "tokenizer.model", "spiece.model",
    "model.safetensors.index.json", "pytorch_model.bin.index.json",
}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def weight_file(name):
    return name == "model.safetensors" or name == "pytorch_model.bin" or (
        name.startswith("model-") and name.endswith(".safetensors")
    ) or (name.startswith("pytorch_model-") and name.endswith(".bin"))

def inventory(folder):
    return {p.relative_to(folder).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(folder.rglob("*")) if p.is_file()
            and p.name not in {"release_manifest.json", ".gitattributes"} and ".cache" not in p.relative_to(folder).parts}

def check_manifest(folder):
    manifest = json.loads((folder / "release_manifest.json").read_text(encoding="utf-8"))
    if inventory(folder) != manifest["files"]:
        raise ValueError("Package differs from release_manifest.json; prepare a fresh package")
    return manifest

def prepare(preset, source, destination):
    source, destination = source.resolve(), destination.resolve()
    if not source.is_dir():
        raise ValueError("Source checkpoint directory does not exist")
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Destination must be outside the source checkpoint")
    if destination.exists():
        raise ValueError("Use a new destination; existing files are never overwritten")
    spec = MODELS[preset]
    training = json.loads((source / "training_config.json").read_text(encoding="utf-8"))
    if training.get("text_prefix") != spec["text_prefix"]:
        raise ValueError("Saved text_prefix does not match the selected experiment preset")
    if not (source / "config.json").is_file() or not (source / "modules.json").is_file():
        raise ValueError("Expected a complete saved SentenceTransformer checkpoint")
    files = [p for p in source.rglob("*") if p.is_file()
             and not any(part.startswith(".") for part in p.relative_to(source).parts)
             and (p.name in RUNTIME_FILES or weight_file(p.name))]
    weights = [p for p in files if weight_file(p.name)]
    if not weights or any(p.stat().st_size == 0 for p in weights):
        raise ValueError("No non-empty trained model weights found")
    if not any(p.name in {"tokenizer.json", "vocab.txt", "sentencepiece.bpe.model", "spiece.model", "tokenizer.model"} for p in files):
        raise ValueError("Tokenizer files are missing")
    # Only include the module directories declared by SentenceTransformers.
    modules = json.loads((source / "modules.json").read_text(encoding="utf-8"))
    module_paths = {x.get("path", "") for x in modules}
    allowed = []
    for p in files:
        rel = p.relative_to(source)
        parent = rel.parent.as_posix()
        if parent == "." or parent in module_paths:
            allowed.append(p)
    destination.mkdir(parents=True)
    for p in allowed:
        target = destination / p.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        if p.suffix == ".json" and p.name == "config.json":
            value = json.loads(target.read_text(encoding="utf-8"))
            value.pop("_name_or_path", None)  # not needed for local loading
            target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    safe_training = {k: training[k] for k in ("epochs", "batch_size", "warmup_steps", "text_prefix", "seed") if k in training}
    # This added field records the publication preset, not an inferred training path.
    safe_training["publication_preset"] = preset
    safe_training["normalize_embeddings"] = spec["normalize_embeddings"]
    (destination / "training_config.json").write_text(json.dumps(safe_training, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(ASSETS / "cards" / f"{preset}.md", destination / "README.md")
    shutil.copy2(ASSETS / "licenses" / spec["license_file"], destination / "LICENSE")
    # Preserve any additional upstream notices present in the checkpoint.
    for p in source.iterdir():
        if p.is_file() and p.name.upper().startswith(("LICENSE", "NOTICE")):
            shutil.copy2(p, destination / ("UPSTREAM_" + p.name))
    (destination / "BASE_MODEL.md").write_text(
        f"# Base model attribution\n\nBase: [{spec['base_model']}](https://huggingface.co/{spec['base_model']}).\n"
        f"Base authorship belongs to its named contributors. License: {spec['license']}.\n"
        "Fine-tuning modifications: Saša Z. Petalinkar, 2026.\n"
        f"The base model card was checked at revision {spec['base_card_revision_checked']}; this is not a claim about the historical training revision.\n",
        encoding="utf-8")
    listing = inventory(destination)
    manifest = {"preset": preset, "repo_id": spec["repo_id"], "files": listing,
                "total_bytes": sum(x["bytes"] for x in listing.values())}
    (destination / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"package": str(destination), "repo_id": spec["repo_id"], "bytes": manifest["total_bytes"]}, indent=2))

def embeddings(path, spec):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(path), device="cpu", local_files_only=True)
    texts = ["Он је љубазан.", "Онај који се предусретљиво односи према другима.",
             "Онај који се односи на временске прилике.", "Дошао је у последњи час.",
             "query: Већ додат префикс.", "Реченица са више речи. " * 200]
    prefix = spec["text_prefix"]
    texts = [s if s.startswith(prefix) else prefix + s for s in texts]
    values = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return values

def verify(source, package):
    import numpy as np
    manifest = check_manifest(package)
    spec = MODELS[manifest["preset"]]
    original, packed = embeddings(source, spec), embeddings(package, spec)
    np.testing.assert_allclose(original, packed, rtol=1e-5, atol=1e-6)
    for values in (original, packed):
        assert np.isfinite(values).all()
    assert np.array_equal(np.argsort(-(original[0] @ original[1:3].T)), np.argsort(-(packed[0] @ packed[1:3].T)))
    print("Local checkpoint and packaged embeddings/ranking agree on fixed inputs.")

def upload(package):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    manifest = check_manifest(package)
    api = HfApi()
    if api.whoami()["name"] != "Tanor":
        raise ValueError("Log in to the Tanor account before uploading")
    try:
        info = api.model_info(manifest["repo_id"])
    except RepositoryNotFoundError:
        api.create_repo(manifest["repo_id"], private=True, repo_type="model")
    else:
        if not info.private:
            raise ValueError("This helper only uploads to a private staging repository")
    hf = shutil.which("hf")
    if not hf:
        raise RuntimeError("Activate the local environment containing the hf CLI")
    subprocess.run([hf, "upload-large-folder", manifest["repo_id"], str(package), "--private", "--no-bars"], check=True)
    print("Upload complete. Run publish to verify the pinned download before making it public.")

def publish(package):
    from huggingface_hub import HfApi, snapshot_download
    manifest = check_manifest(package)
    api = HfApi()
    if api.whoami()["name"] != "Tanor":
        raise ValueError("Log in to the Tanor account before publishing")
    repo_id = manifest["repo_id"]
    info = api.model_info(repo_id)
    if not info.private:
        raise ValueError("Repository is already public; inspect its release instead of republishing")
    revision = info.sha
    downloaded = Path(snapshot_download(repo_id, revision=revision))
    if check_manifest(downloaded)["files"] != manifest["files"]:
        raise ValueError("Hub files differ from the prepared package")
    verify(package, downloaded)
    api.create_tag(repo_id, tag="v1.0", revision=revision, exist_ok=True)
    api.update_repo_settings(repo_id, private=False)
    receipt = {"repo_id": repo_id, "revision": revision, "tag": "v1.0", "total_bytes": manifest["total_bytes"]}
    (package.parent / f"{manifest['preset']}-published.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--preset", choices=MODELS, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--package", type=Path, required=True)
    for name in ("upload", "publish"):
        p = sub.add_parser(name)
        p.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.preset, args.source, args.destination)
    elif args.command == "verify": verify(args.source, args.package)
    elif args.command == "upload": upload(args.package)
    else: publish(args.package)

if __name__ == "__main__":
    main()
