from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from ml.topics import (
    TopicTrainingConfig,
    TopicTrainingError,
    _dataset_revision,
    inspect_topic_artifact,
    train_topic_model,
)


def test_topic_training_config_has_multilingual_default_and_smoke_flag() -> None:
    config = TopicTrainingConfig(smoke_test=True)
    assert "multilingual-MiniLM" in config.embedding_model
    assert config.smoke_test is True
    assert config.random_seed == 42


def test_topic_artifact_inspection_reads_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "topic-v1"
    artifact.mkdir()
    (artifact / "topic_manifest.json").write_text(
        json.dumps({"project": "MahaPulse", "smoke_test": True}), encoding="utf-8"
    )
    assert inspect_topic_artifact(artifact)["smoke_test"] is True


def test_topic_artifact_inspection_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(TopicTrainingError, match="manifest is missing"):
        inspect_topic_artifact(tmp_path / "missing")


def test_topic_provenance_reads_actual_prepared_dataset_metadata(tmp_path: Path) -> None:
    revision = "8ee29fa1329d6a841030eb46659d3c10614b5e59"
    (tmp_path / "dataset_report.json").write_text(
        json.dumps({"source_metadata": {"source_revision": revision}}), encoding="utf-8"
    )
    assert _dataset_revision(tmp_path) == revision


def test_topic_training_seeds_umap_and_never_reads_held_out_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the offline pipeline without importing models or downloading data."""
    observed: dict[str, object] = {}
    documents = [SimpleNamespace(model_text="आज दुकान सकाळी उघडले.")] * 9

    def load_records(_processed_dir: Path, split: str) -> list[SimpleNamespace]:
        assert split == "train", "topic fitting must not consume held-out splits"
        observed["split"] = split
        return documents

    def fake_umap(**kwargs: object) -> object:
        observed["umap"] = kwargs
        return SimpleNamespace(**kwargs)

    def fake_embeddings(model_name: str, **kwargs: object) -> object:
        observed["embeddings"] = {"model_name": model_name, **kwargs}
        return object()

    class FakeTopicModel:
        def __init__(self, **kwargs: object) -> None:
            observed["topic_config"] = kwargs

        def fit_transform(self, texts: list[str]) -> tuple[list[int], None]:
            observed["document_count"] = len(texts)
            return [0] * len(texts), None

        def save(self, path: str, **kwargs: object) -> None:
            observed["save"] = kwargs
            Path(path).mkdir(parents=True)

        def get_topics(self) -> dict[int, list[tuple[str, float]]]:
            return {0: [("दुकान", 1.0)]}

        def get_topic_info(self) -> object:
            return SimpleNamespace(to_dict=lambda **kwargs: [{"Topic": 0, "Count": 9}])

    modules: dict[str, dict[str, object]] = {
        "torch": {
            "manual_seed": lambda seed: observed.update(torch_seed=seed),
            "cuda": SimpleNamespace(is_available=lambda: False),
        },
        "numpy": {"random": SimpleNamespace(seed=lambda seed: None)},
        "bertopic": {"BERTopic": FakeTopicModel},
        "sentence_transformers": {"SentenceTransformer": fake_embeddings},
        "sklearn.feature_extraction.text": {
            "CountVectorizer": lambda **kwargs: SimpleNamespace(**kwargs)
        },
        "umap": {"UMAP": fake_umap},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("ml.topics.load_split_records", load_records)

    config = TopicTrainingConfig(random_seed=7)
    artifact = train_topic_model(tmp_path, tmp_path / "artifacts", config, "topics-full")
    manifest = inspect_topic_artifact(artifact)

    assert observed["split"] == "train"
    assert observed["document_count"] == 9
    assert observed["torch_seed"] == 7
    assert observed["umap"]["random_state"] == 7
    assert observed["umap"]["transform_seed"] == 7
    assert observed["umap"]["n_jobs"] == 1
    assert observed["embeddings"]["local_files_only"] is True
    assert observed["save"]["save_embedding_model"] is False
    assert manifest["document_count"] == 9
    assert manifest["training_config"]["random_seed"] == 7
    assert manifest["smoke_test"] is False
