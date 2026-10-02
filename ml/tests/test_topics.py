from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.topics import TopicTrainingConfig, TopicTrainingError, inspect_topic_artifact


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
