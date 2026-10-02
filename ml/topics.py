"""Offline BERTopic training and artifact inspection.

BERTopic is deliberately kept out of the request path.  This module consumes
the prepared MahaSent-MD split manifests, trains once on a corpus, and writes
an artifact that the FastAPI TopicService can load later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

from .dataset import load_split_records
from .preprocessing import PREPROCESSING_VERSION, build_analysis_tokens, preprocess_text


class TopicTrainingError(RuntimeError):
    """Raised when topic training or artifact creation cannot proceed."""


@dataclass(frozen=True)
class TopicTrainingConfig:
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    min_topic_size: int = 10
    nr_topics: int | None = None
    random_seed: int = 42
    smoke_test: bool = False


def train_topic_model(
    processed_dir: Path,
    artifact_root: Path,
    config: TopicTrainingConfig,
    model_version: str,
) -> Path:
    records = load_split_records(processed_dir, "train")
    if config.smoke_test:
        records = records[: min(len(records), 12)]
    documents = [
        preprocess_text(record.model_text).analysis_text
        for record in records
        if record.model_text.strip()
    ]
    if len(documents) < 2:
        raise TopicTrainingError("at least two non-empty training documents are required")
    try:
        import torch
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer

        random.seed(config.random_seed)
        try:
            import numpy as np

            np.random.seed(config.random_seed)
        except ImportError:
            pass
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = SentenceTransformer(
            config.embedding_model,
            device=device,
            local_files_only=True,
        )
        topic_model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=config.min_topic_size,
            nr_topics=config.nr_topics,
            calculate_probabilities=True,
            low_memory=True,
            verbose=False,
            vectorizer_model=CountVectorizer(
                tokenizer=build_analysis_tokens, token_pattern=None, lowercase=False
            ),
        )
        topics, _ = topic_model.fit_transform(documents)
    except TopicTrainingError:
        raise
    except Exception as exc:  # noqa: BLE001 - CLI converts to a useful error
        raise TopicTrainingError(
            "BERTopic training dependencies or local embedding model are unavailable"
        ) from exc

    artifact_dir = artifact_root / "topics" / model_version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir = artifact_dir / "model"
    try:
        try:
            topic_model.save(
                str(model_dir),
                serialization="safetensors",
                save_embedding_model=False,
            )
        except (TypeError, ValueError):
            topic_model.save(
                str(model_dir), serialization="pickle", save_embedding_model=False
            )
    except Exception as exc:  # noqa: BLE001 - CLI converts to a useful error
        raise TopicTrainingError("BERTopic artifact could not be saved") from exc

    topic_labels = _topic_labels(topic_model)
    metadata = _topic_metadata(topic_model)
    (artifact_dir / "topic_metadata.json").write_text(
        json.dumps({"topics": metadata}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (artifact_dir / "training_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "project": "MahaPulse",
        "task": "topic inference",
        "model_version": model_version,
        "embedding_model": config.embedding_model,
        "dataset_name": "L3Cube MahaSent-MD",
        "dataset_source": "prepared processed split manifests",
        "dataset_revision": _dataset_revision(processed_dir),
        "document_count": len(documents),
        "topic_ids": sorted({int(topic) for topic in topics if int(topic) != -1}),
        "topic_labels": {str(key): value for key, value in topic_labels.items()},
        "preprocessing_version": PREPROCESSING_VERSION,
        "training_config": asdict(config),
        "smoke_test": config.smoke_test,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (artifact_dir / "topic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return artifact_dir


def inspect_topic_artifact(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "topic_manifest.json"
    if not path.is_file():
        raise TopicTrainingError("topic artifact manifest is missing")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TopicTrainingError("topic artifact manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise TopicTrainingError("topic artifact manifest must be a JSON object")
    return manifest


def _topic_labels(topic_model: Any) -> dict[int, str]:
    labels: dict[int, str] = {}
    for topic_id, words in topic_model.get_topics().items():
        if int(topic_id) == -1:
            continue
        terms = [str(item[0]) for item in (words or [])[:5] if item]
        if terms:
            labels[int(topic_id)] = ", ".join(terms)
    return labels


def _topic_metadata(topic_model: Any) -> list[dict[str, Any]]:
    info = topic_model.get_topic_info()
    if hasattr(info, "to_dict"):
        return info.to_dict(orient="records")
    return []


def _dataset_revision(processed_dir: Path) -> str | None:
    report_path = processed_dir / "dataset_report.json"
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return report.get("source_metadata", {}).get("source_revision")
