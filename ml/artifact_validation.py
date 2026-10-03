"""Lightweight checks for production evidence; no ML runtime is imported."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from .dataset import LABEL_MAPPING
from .preprocessing import PREPROCESSING_VERSION


class ArtifactEvidenceError(ValueError):
    """A saved artifact does not contain consistent production evidence."""


PRODUCTION_LIFECYCLE_GATES = (
    "training_completed", "held_out_test_evaluated", "artifact_reload_validated",
    "integrity_verified", "api_integration_validated",
)
PRODUCTION_EVIDENCE_FILES = (
    "training_config.json", "dataset_report.json", "metrics.json", "predictions.jsonl",
)
CORE_INTEGRITY_FILES = (
    "config.json", "tokenizer_config.json", "tokenizer.json", "label_mapping.json",
)
METRIC_NAMES = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")


def read_evidence_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactEvidenceError(f"artifact metadata is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise ArtifactEvidenceError(f"artifact metadata must be an object: {path.name}")
    return value


def weight_files(path: Path) -> set[str]:
    """Resolve real files, including every shard referenced by a weights index."""
    result = {
        name for name in ("model.safetensors", "pytorch_model.bin")
        if (path / name).is_file()
    }
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if not (path / index_name).is_file():
            continue
        index = read_evidence_json(path / index_name)
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ArtifactEvidenceError("artifact weights index is invalid")
        result.add(index_name)
        for filename in mapping.values():
            if not isinstance(filename, str) or (path / filename).resolve().parent != path.resolve():
                raise ArtifactEvidenceError("artifact weights filename is invalid")
            if not (path / filename).is_file():
                raise ArtifactEvidenceError("artifact is missing a model weights shard")
            result.add(filename)
    if not result:
        raise ArtifactEvidenceError("artifact is missing model weights")
    if any((path / name).resolve().parent != path.resolve() for name in result):
        raise ArtifactEvidenceError("artifact weights filename is invalid")
    if any((path / name).stat().st_size <= 0 for name in result):
        raise ArtifactEvidenceError("artifact model weights are empty")
    return result


def verify_integrity(path: Path, manifest: dict[str, Any], *, production: bool) -> None:
    integrity = manifest.get("artifact_integrity", {})
    if not isinstance(integrity, dict):
        raise ArtifactEvidenceError("artifact integrity metadata is invalid")
    if production:
        required = set(CORE_INTEGRITY_FILES + PRODUCTION_EVIDENCE_FILES) | weight_files(path)
        actual_files = {
            item.name for item in path.iterdir()
            if item.is_file() and item.name != "model_manifest.json"
        }
        if not required.issubset(integrity) or set(integrity) != actual_files:
            raise ArtifactEvidenceError("artifact integrity coverage is incomplete")
    for filename, metadata in integrity.items():
        file_path = (path / filename).resolve()
        if file_path.parent != path.resolve() or filename == "model_manifest.json":
            raise ArtifactEvidenceError("artifact integrity filename is invalid")
        if not file_path.is_file() or not isinstance(metadata, dict):
            raise ArtifactEvidenceError("artifact integrity files are invalid")
        expected_size, expected_hash = metadata.get("bytes"), metadata.get("sha256")
        if (
            type(expected_size) is not int or expected_size < 0
            or not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        ):
            raise ArtifactEvidenceError("artifact integrity digest metadata is invalid")
        if file_path.stat().st_size != expected_size:
            raise ArtifactEvidenceError("artifact file size does not match manifest")
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ArtifactEvidenceError("artifact integrity check failed")


def _number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ArtifactEvidenceError(f"{description} must be finite numeric evidence")
    return float(value)


def _matching_metric(actual: Any, expected: float, description: str) -> None:
    if not math.isclose(_number(actual, description), expected, rel_tol=1e-7, abs_tol=1e-9):
        raise ArtifactEvidenceError(f"{description} does not match prediction evidence")


def _validate_test_predictions(path: Path, count: int, metrics: dict[str, Any]) -> None:
    confusion = [[0] * 3 for _ in range(3)]
    identifiers: set[str] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ArtifactEvidenceError("prediction evidence must contain objects")
                identifier = row.get("id")
                if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                    raise ArtifactEvidenceError("prediction evidence identifiers are missing or duplicated")
                identifiers.add(identifier)
                probabilities = row.get("class_probabilities")
                if not isinstance(probabilities, dict) or set(probabilities) != set(LABEL_MAPPING):
                    raise ArtifactEvidenceError("prediction probabilities use incompatible labels")
                values = [_number(probabilities[label], "prediction probability") for label in LABEL_MAPPING]
                if any(value < 0 or value > 1 for value in values) or not math.isclose(sum(values), 1, abs_tol=1e-6):
                    raise ArtifactEvidenceError("prediction probabilities are invalid")
                gold, predicted = row.get("gold_label"), row.get("predicted_label")
                if not isinstance(gold, str) or not isinstance(predicted, str) or gold not in LABEL_MAPPING or predicted not in LABEL_MAPPING:
                    raise ArtifactEvidenceError("prediction labels are incompatible")
                if values[LABEL_MAPPING[predicted]] != max(values):
                    raise ArtifactEvidenceError("prediction label disagrees with probabilities")
                _matching_metric(row.get("confidence"), max(values), "prediction confidence")
                if not isinstance(row.get("text"), str) or not row["text"].strip():
                    raise ArtifactEvidenceError("prediction text is missing")
                confusion[LABEL_MAPPING[gold]][LABEL_MAPPING[predicted]] += 1
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactEvidenceError("prediction evidence is unreadable") from exc
    if len(identifiers) != count:
        raise ArtifactEvidenceError("prediction count does not match held-out test count")
    if metrics.get("confusion_matrix") != confusion:
        raise ArtifactEvidenceError("test confusion matrix does not match prediction evidence")
    per_class = metrics.get("per_class")
    if not isinstance(per_class, dict) or set(per_class) != set(LABEL_MAPPING):
        raise ArtifactEvidenceError("test per-class metrics are incomplete")
    precisions, recalls, f1s, supports = [], [], [], []
    for label, index in LABEL_MAPPING.items():
        support = sum(confusion[index])
        predicted_count = sum(row[index] for row in confusion)
        tp = confusion[index][index]
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        entry = per_class[label]
        if not isinstance(entry, dict) or type(entry.get("support")) is not int or entry["support"] != support:
            raise ArtifactEvidenceError("test class support does not match prediction evidence")
        for name, value in (("precision", precision), ("recall", recall), ("f1", f1)):
            _matching_metric(entry.get(name), value, f"test {label} {name}")
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
    expected = {
        "accuracy": sum(confusion[index][index] for index in range(3)) / count,
        "macro_precision": sum(precisions) / 3, "macro_recall": sum(recalls) / 3,
        "macro_f1": sum(f1s) / 3,
        "weighted_f1": sum(f1 * support for f1, support in zip(f1s, supports)) / count,
    }
    for name, value in expected.items():
        _matching_metric(metrics.get(name), value, f"test {name}")


def validate_production_evidence(path: Path, manifest: dict[str, Any]) -> int:
    """Check real training provenance and recompute test metrics from JSONL."""
    config = read_evidence_json(path / "training_config.json")
    report = read_evidence_json(path / "dataset_report.json")
    metrics = read_evidence_json(path / "metrics.json")
    lifecycle = manifest.get("lifecycle_validation")
    if not isinstance(lifecycle, dict) or lifecycle.get("training_completed") is not True or lifecycle.get("held_out_test_evaluated") is not True:
        raise ArtifactEvidenceError("artifact training or held-out evaluation is incomplete")
    if config.get("smoke_test") is not False or manifest.get("smoke_test") is not False or metrics.get("smoke_test") is not False:
        raise ArtifactEvidenceError("smoke artifacts cannot be promoted")
    if manifest.get("training_config") != config or config.get("model_name") != "google/muril-base-cased":
        raise ArtifactEvidenceError("artifact real training configuration is inconsistent")
    if config.get("max_length") != manifest.get("max_length"):
        raise ArtifactEvidenceError("artifact tokenizer training configuration is inconsistent")
    training = manifest.get("training_summary")
    if (
        not isinstance(training, dict) or training.get("completed") is not True
        or training.get("selection_split") != "validation"
        or training.get("selection_metric") != "macro_f1"
        or not isinstance(training.get("selected_checkpoint"), str)
        or not training["selected_checkpoint"]
    ):
        raise ArtifactEvidenceError("artifact validation checkpoint selection evidence is incomplete")
    if _number(training.get("training_duration_seconds"), "training duration") <= 0:
        raise ArtifactEvidenceError("artifact training duration is invalid")
    if type(training.get("global_step")) is not int or training["global_step"] <= 0 or _number(training.get("epoch"), "training epoch") <= 0:
        raise ArtifactEvidenceError("artifact completed training steps are invalid")
    if report.get("label_mapping") != LABEL_MAPPING or manifest.get("label_mapping") != LABEL_MAPPING:
        raise ArtifactEvidenceError("artifact dataset labels are incompatible")
    if report.get("preprocessing_version") != PREPROCESSING_VERSION or manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ArtifactEvidenceError("artifact preprocessing provenance is incompatible")
    source = report.get("source_metadata")
    revision = source.get("source_revision") if isinstance(source, dict) else None
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None or manifest.get("dataset_revision") != revision:
        raise ArtifactEvidenceError("artifact dataset revision is missing or inconsistent")
    splits = report.get("split_distribution")
    classes = report.get("class_distribution")
    if not isinstance(splits, dict) or any(type(splits.get(name)) is not int or splits[name] <= 0 for name in ("train", "validation", "test")):
        raise ArtifactEvidenceError("artifact dataset split counts are invalid")
    if not isinstance(classes, dict) or set(classes) != set(LABEL_MAPPING) or any(type(value) is not int or value <= 0 for value in classes.values()):
        raise ArtifactEvidenceError("artifact dataset class counts are invalid")
    if set(splits) != {"train", "validation", "test"} or sum(splits.values()) != report.get("valid_rows") or sum(classes.values()) != report.get("valid_rows"):
        raise ArtifactEvidenceError("artifact dataset counts are inconsistent")
    evaluation = manifest.get("final_test_evaluation")
    if (
        not isinstance(evaluation, dict) or evaluation.get("completed") is not True
        or evaluation.get("split") != "test" or evaluation.get("count") != splits["test"]
        or evaluation.get("checkpoint") != training["selected_checkpoint"]
        or evaluation.get("dataset_revision") != revision
        or evaluation.get("preprocessing_version") != PREPROCESSING_VERSION
    ):
        raise ArtifactEvidenceError("artifact final held-out evaluation provenance is incomplete")
    validation, test = metrics.get("validation"), metrics.get("test")
    if not isinstance(validation, dict) or not isinstance(test, dict):
        raise ArtifactEvidenceError("validation or held-out test metrics are missing")
    for name in METRIC_NAMES:
        value = _number(validation.get(name), f"validation {name}")
        if not 0 <= value <= 1:
            raise ArtifactEvidenceError("validation metrics are outside their valid range")
    if manifest.get("evaluation_summary") != test:
        raise ArtifactEvidenceError("held-out evaluation summary disagrees with test metrics")
    _validate_test_predictions(path / "predictions.jsonl", splits["test"], test)
    return splits["test"]
