"""Online BERTopic artifact inference with local-only model loading."""

from __future__ import annotations

import json
import math
from pathlib import Path
from threading import RLock
from typing import Any

from ...config import Settings
from ...exceptions import ServiceFailure
from ..interfaces import ServiceMetadata, TopicResult
from ml.preprocessing import PREPROCESSING_VERSION


class BertopicServiceError(ServiceFailure):
    """Raised when a saved BERTopic artifact cannot be loaded or queried."""


class BertopicTopicService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = RLock()
        self._model: Any | None = None
        self._topic_labels: dict[int, str] = {}
        self._version = "unavailable"
        self._device: str | None = None
        self._error: str | None = None
        self._embedding_model = settings.topic_embedding_model

    def metadata(self) -> ServiceMetadata:
        try:
            self._ensure_loaded()
        except BertopicServiceError:
            pass
        state = "ready" if self._model is not None else "unavailable"
        return ServiceMetadata(
            name="BERTopic",
            version=self._version,
            device=self._device or "not-loaded",
            state=state,
            production_ready=state == "ready",
            backend="bertopic",
            provider="BERTopic",
            embedding_model=self._embedding_model,
        )

    def classify(self, analysis_text: str) -> TopicResult:
        if not analysis_text.strip():
            return TopicResult(id=None, label=None, probability=None)
        model = self._ensure_loaded()
        try:
            topics, probabilities = model.transform([analysis_text])
            topic_id = int(topics[0])
            if topic_id == -1:
                return TopicResult(id=None, label=None, probability=None)
            # Compact safetensors artifacts use cosine similarity, with an
            # initial outlier column when -1 exists. Full HDBSCAN membership
            # matrices exclude that column. Match BERTopic's actual lite
            # transform mode rather than offsetting every probability matrix.
            cluster_type = type(getattr(model, "hdbscan_model", None))
            cosine_mode = (
                cluster_type.__module__ == "bertopic.cluster._base"
                and cluster_type.__name__ == "BaseCluster"
            )
            outlier_column = int(getattr(model, "_outliers", 0)) if cosine_mode else 0
            probability = self._probability_for(
                probabilities, topic_id, outlier_column=outlier_column
            )
            return TopicResult(
                id=topic_id,
                label=self._topic_labels.get(topic_id),
                probability=probability,
            )
        except Exception as exc:  # noqa: BLE001 - converted at service boundary
            raise BertopicServiceError("BERTopic inference failed") from exc

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        if self._error is not None:
            raise BertopicServiceError("BERTopic artifact or embedding model is unavailable locally")
        with self._lock:
            if self._model is not None:
                return self._model
            if self._error is not None:
                raise BertopicServiceError("BERTopic artifact or embedding model is unavailable locally")
            artifact = self._artifact_path()
            try:
                manifest = self._read_json(artifact / "topic_manifest.json")
                metadata = self._read_json(artifact / "topic_metadata.json")
                model_path = artifact / "model"
                if not model_path.exists():
                    raise BertopicServiceError("BERTopic model directory is missing")
                if manifest.get("project") != "MahaPulse":
                    raise BertopicServiceError("BERTopic artifact project is incompatible")
                if manifest.get("task") != "topic inference":
                    raise BertopicServiceError("BERTopic artifact task is incompatible")
                if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
                    raise BertopicServiceError("BERTopic preprocessing version is incompatible")
                model_version = manifest.get("model_version")
                if not isinstance(model_version, str) or not model_version:
                    raise BertopicServiceError("BERTopic artifact model version is missing")
                embedding_name = manifest.get("embedding_model")
                if not isinstance(embedding_name, str) or not embedding_name:
                    raise BertopicServiceError("BERTopic embedding model is missing")
                self._version = model_version
                self._embedding_model = embedding_name
                self._topic_labels = self._labels_from_metadata(manifest, metadata)
                import torch
                from bertopic import BERTopic
                from sentence_transformers import SentenceTransformer

                self._device = self._select_device(torch, self.settings.model_device)
                embedding_model = SentenceTransformer(
                    self._embedding_model,
                    device=self._device,
                    local_files_only=True,
                )
                try:
                    self._model = BERTopic.load(
                        str(model_path), embedding_model=embedding_model
                    )
                except TypeError:
                    self._model = BERTopic.load(str(model_path))
                return self._model
            except BertopicServiceError:
                raise
            except Exception as exc:  # noqa: BLE001 - safe metadata and API boundary
                self._error = str(exc)
                raise BertopicServiceError(
                    "BERTopic artifact or embedding model is unavailable locally"
                ) from exc

    def _artifact_path(self) -> Path:
        if not self.settings.topic_model_path:
            raise BertopicServiceError("BERTopic artifact path is not configured")
        artifact = Path(self.settings.topic_model_path).expanduser()
        if not artifact.is_dir():
            raise BertopicServiceError("BERTopic artifact directory is unavailable")
        return artifact

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise BertopicServiceError("BERTopic artifact metadata is missing")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BertopicServiceError("BERTopic artifact metadata is invalid") from exc
        if not isinstance(value, dict):
            raise BertopicServiceError("BERTopic artifact metadata is invalid")
        return value

    @staticmethod
    def _labels_from_metadata(
        manifest: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[int, str]:
        labels = manifest.get("topic_labels", {})
        if isinstance(labels, dict):
            return {int(key): str(value) for key, value in labels.items()}
        records = metadata.get("topics", [])
        result: dict[int, str] = {}
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict) and "Topic" in record:
                    words = record.get("Representation") or record.get("Name")
                    if isinstance(words, list):
                        result[int(record["Topic"])] = ", ".join(map(str, words[:5]))
                    elif words:
                        result[int(record["Topic"])] = str(words)
        return result

    @staticmethod
    def _probability_for(
        probabilities: Any, topic_id: int, *, outlier_column: int = 0
    ) -> float | None:
        """Map the assigned topic's score, not a calibrated probability.

        BERTopic's compact reload returns cosine similarities, which need not
        sum to one. Keep the existing API's bounded score contract.
        """
        if probabilities is None:
            return None
        values = probabilities.tolist() if hasattr(probabilities, "tolist") else probabilities
        is_matrix = isinstance(values, (list, tuple)) and bool(values) and isinstance(
            values[0], (list, tuple)
        )
        if is_matrix:
            values = values[0]
        if isinstance(values, (float, int)):
            score = float(values)
        elif not values:
            return None
        else:
            index = topic_id + outlier_column if is_matrix else topic_id
            score = float(values[index]) if 0 <= index < len(values) else float(max(values))
        return max(0.0, min(1.0, score)) if math.isfinite(score) else None

    @staticmethod
    def _select_device(torch_module: Any, configured: str) -> str:
        if configured == "cuda":
            if not torch_module.cuda.is_available():
                raise BertopicServiceError("CUDA was explicitly requested but is unavailable")
            return "cuda"
        if configured == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        return "cpu"
