"""Local MuRIL inference service for a validated MahaPulse artifact."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import re
import threading
from typing import Any

from ml.preprocessing import PREPROCESSING_VERSION
from ml.artifact_validation import (
    ArtifactEvidenceError, PRODUCTION_EVIDENCE_FILES, PRODUCTION_LIFECYCLE_GATES,
    validate_production_evidence, verify_integrity, weight_files,
)

from ...config import Settings
from ..interfaces import SentimentResult, ServiceMetadata


LOGGER = logging.getLogger("mahapulse")
CANONICAL_LABEL_MAPPING = {"negative": 0, "neutral": 1, "positive": 2}
EXPECTED_BASE_MODEL = "google/muril-base-cased"
EXPECTED_TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json")
EXPECTED_ARTIFACT_FILES = ("model_manifest.json", "label_mapping.json", "config.json")


class MurilArtifactError(RuntimeError):
    """Raised when a local artifact is unsafe or incompatible."""

    metadata: Any | None = None


class MurilServiceError(RuntimeError):
    """Raised when the real sentiment service cannot serve a prediction."""


@dataclass(frozen=True)
class ArtifactMetadata:
    model_version: str
    base_model: str
    preprocessing_version: str
    max_length: int
    smoke_test: bool
    production_ready: bool
    label_mapping: dict[str, int]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MurilArtifactError("sentiment artifact metadata is unreadable") from exc
    if not isinstance(value, dict):
        raise MurilArtifactError("sentiment artifact metadata must be a JSON object")
    return value


def _validate_artifact(path: Path, allow_smoke_model: bool) -> ArtifactMetadata:
    if not path.is_dir():
        raise MurilArtifactError("sentiment artifact directory is unavailable")
    for filename in EXPECTED_ARTIFACT_FILES:
        if not (path / filename).is_file():
            raise MurilArtifactError("sentiment artifact is missing required metadata")
    try:
        weight_files(path)
    except ArtifactEvidenceError as exc:
        raise MurilArtifactError(str(exc)) from exc
    if any(not (path / filename).is_file() for filename in EXPECTED_TOKENIZER_FILES):
        raise MurilArtifactError("sentiment artifact is missing tokenizer files")

    manifest = _read_json(path / "model_manifest.json")
    mapping_file = _read_json(path / "label_mapping.json")
    model_config = _read_json(path / "config.json")
    mapping = mapping_file.get("canonical_to_id")
    if manifest.get("project") != "MahaPulse":
        raise MurilArtifactError("sentiment artifact project is incompatible")
    if manifest.get("task") != "sentiment classification":
        raise MurilArtifactError("sentiment artifact task is incompatible")
    if manifest.get("architecture") != "AutoModelForSequenceClassification":
        raise MurilArtifactError("sentiment artifact architecture is incompatible")
    if manifest.get("base_model", manifest.get("model_name")) != EXPECTED_BASE_MODEL:
        raise MurilArtifactError("sentiment artifact base model is incompatible")
    if mapping != CANONICAL_LABEL_MAPPING or manifest.get("label_mapping") != CANONICAL_LABEL_MAPPING:
        raise MurilArtifactError("sentiment artifact label mapping is incompatible")
    expected_ids = {str(identifier): label for label, identifier in CANONICAL_LABEL_MAPPING.items()}
    if model_config.get("label2id") != CANONICAL_LABEL_MAPPING or model_config.get("id2label") != expected_ids:
        raise MurilArtifactError("sentiment model config label mapping is incompatible")
    if not isinstance(manifest.get("smoke_test"), bool):
        raise MurilArtifactError("sentiment artifact smoke_test flag is invalid")
    production_ready = manifest.get("production_ready", False)
    if not isinstance(production_ready, bool):
        raise MurilArtifactError("sentiment artifact production_ready flag is invalid")
    if production_ready and manifest["smoke_test"]:
        raise MurilArtifactError("smoke sentiment artifacts cannot be production-ready")
    if production_ready:
        lifecycle = manifest.get("lifecycle_validation")
        if not isinstance(lifecycle, dict) or any(
            lifecycle.get(gate) is not True for gate in PRODUCTION_LIFECYCLE_GATES
        ):
            raise MurilArtifactError("sentiment artifact production lifecycle is incomplete")
        if any(not (path / filename).is_file() for filename in PRODUCTION_EVIDENCE_FILES):
            raise MurilArtifactError("sentiment artifact production evidence is incomplete")
        if not isinstance(manifest.get("evaluation_summary"), dict):
            raise MurilArtifactError("sentiment artifact held-out evaluation is missing")
        if not isinstance(manifest.get("dataset_revision"), str) or not manifest["dataset_revision"]:
            raise MurilArtifactError("sentiment artifact dataset revision is missing")
        api_evidence = manifest.get("api_validation_evidence")
        if (
            not isinstance(api_evidence, dict) or api_evidence.get("schema_version") != 1
            or api_evidence.get("model_version") != manifest.get("model_version")
            or api_evidence.get("artifact_integrity") != manifest.get("artifact_integrity")
            or api_evidence.get("independent_local_reload_passed") is not True
            or api_evidence.get("application_passed") is not True
            or api_evidence.get("batch_successes") != 4 or api_evidence.get("batch_expected_failures") != 1
            or api_evidence.get("csv_rows") != 5 or api_evidence.get("json_documents") != 5
            or any(
                not isinstance(api_evidence.get(key), str)
                or re.fullmatch(r"[0-9a-f]{64}", api_evidence[key]) is None
                for key in ("report_sha256", "tested_manifest_sha256")
            )
        ):
            raise MurilArtifactError("sentiment artifact real API validation evidence is incomplete")
    model_version = manifest.get("model_version")
    preprocessing_version = manifest.get("preprocessing_version")
    max_length = manifest.get("max_length")
    if not isinstance(model_version, str) or not model_version:
        raise MurilArtifactError("sentiment artifact model version is missing")
    if preprocessing_version != PREPROCESSING_VERSION:
        raise MurilArtifactError("sentiment artifact preprocessing version is incompatible")
    if type(max_length) is not int or max_length <= 0:
        raise MurilArtifactError("sentiment artifact max length is invalid")

    try:
        verify_integrity(path, manifest, production=production_ready)
        if production_ready:
            validate_production_evidence(path, manifest)
    except ArtifactEvidenceError as exc:
        raise MurilArtifactError(str(exc)) from exc

    artifact_metadata = ArtifactMetadata(
        model_version=model_version,
        base_model=EXPECTED_BASE_MODEL,
        preprocessing_version=preprocessing_version,
        max_length=max_length,
        smoke_test=manifest["smoke_test"],
        production_ready=production_ready,
        label_mapping=dict(mapping),
    )
    if artifact_metadata.smoke_test and not allow_smoke_model:
        error = MurilArtifactError("smoke sentiment artifacts are disabled by configuration")
        error.metadata = artifact_metadata
        raise error
    return artifact_metadata


class MurilSentimentService:
    """Singleton-like, lazy-loading MuRIL service owned by one app process."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._artifact_path = Path(settings.sentiment_model_path or "")
        self._lock = threading.RLock()
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None
        self._metadata: ArtifactMetadata | None = None
        self._load_error: Exception | None = None
        try:
            self._metadata = _validate_artifact(self._artifact_path, settings.allow_smoke_model)
        except Exception as exc:
            self._load_error = exc
            self._metadata = getattr(exc, "metadata", None)
            LOGGER.error("MuRIL artifact validation failed", extra={"fields": {"error_type": type(exc).__name__}})

    @staticmethod
    def _select_device(torch_module: Any, requested: str) -> str:
        cuda_available = bool(torch_module.cuda.is_available())
        if requested == "cuda" and not cuda_available:
            raise MurilServiceError("CUDA was explicitly requested but is unavailable")
        if requested == "auto":
            return "cuda" if cuda_available else "cpu"
        return requested

    def _load_components(self) -> tuple[Any, Any, Any, str]:
        if self._metadata is None:
            raise MurilServiceError("MuRIL artifact is not ready")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise MurilServiceError("MuRIL runtime dependencies are unavailable") from exc
        device = self._select_device(torch, self._settings.model_device)
        tokenizer = AutoTokenizer.from_pretrained(self._artifact_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            self._artifact_path, local_files_only=True
        )
        model.to(device)
        model.eval()
        return tokenizer, model, torch, device

    def _ensure_loaded(self, *, raise_error: bool) -> bool:
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return True
            if self._load_error is not None:
                if raise_error:
                    raise MurilServiceError("MuRIL sentiment service is not ready") from self._load_error
                return False
            try:
                self._tokenizer, self._model, self._torch, self._device = self._load_components()
            except Exception as exc:
                self._load_error = exc
                LOGGER.error("MuRIL model load failed", extra={"fields": {"error_type": type(exc).__name__}})
                if raise_error:
                    raise MurilServiceError("MuRIL sentiment service is not ready") from exc
                return False
            return True

    def metadata(self) -> ServiceMetadata:
        loaded = self._ensure_loaded(raise_error=False)
        metadata = self._metadata
        if metadata is None:
            return ServiceMetadata(
                name="MuRIL",
                version="unavailable",
                device=self._device,
                state="not_ready",
                smoke_test=None,
                production_ready=False,
                base_model=EXPECTED_BASE_MODEL,
                preprocessing_version=PREPROCESSING_VERSION,
                backend="muril",
                provider="MuRIL",
            )
        return ServiceMetadata(
            name="MuRIL",
            version=metadata.model_version,
            device=self._device if loaded else "not-loaded",
            state="ready" if loaded else "not_ready",
            smoke_test=metadata.smoke_test,
            production_ready=loaded and metadata.production_ready,
            base_model=metadata.base_model,
            preprocessing_version=metadata.preprocessing_version,
            backend="muril",
            provider="MuRIL",
        )

    def predict(self, model_text: str) -> SentimentResult:
        if not isinstance(model_text, str):
            raise MurilServiceError("MuRIL input must be text")
        from ml.preprocessing import normalize_model_text

        model_text = normalize_model_text(model_text)
        self._ensure_loaded(raise_error=True)
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._metadata is not None
        try:
            encoded = self._tokenizer(
                model_text,
                truncation=True,
                max_length=self._metadata.max_length,
                padding=False,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits
            raw_logits = logits.detach().cpu().tolist()[0]
            id_to_label = {
                value: key for key, value in self._metadata.label_mapping.items()
            }
            if len(raw_logits) != len(id_to_label):
                raise MurilServiceError("MuRIL returned an incompatible number of labels")
            if any(not math.isfinite(float(value)) for value in raw_logits):
                raise MurilServiceError("MuRIL returned non-finite classification scores")
            maximum = max(float(value) for value in raw_logits)
            exponentials = [math.exp(float(value) - maximum) for value in raw_logits]
            total = sum(exponentials)
            probabilities_by_id = [value / total for value in exponentials]
            probabilities = {
                label: probabilities_by_id[index] for index, label in id_to_label.items()
            }
            label = id_to_label[
                max(range(len(probabilities_by_id)), key=probabilities_by_id.__getitem__)
            ]
            return SentimentResult(
                label=label,
                confidence=probabilities[label],
                probabilities=probabilities,
            )
        except MurilServiceError:
            raise
        except Exception as exc:
            LOGGER.exception("MuRIL inference failed", extra={"fields": {"error_type": type(exc).__name__}})
            raise MurilServiceError("MuRIL inference failed") from exc
