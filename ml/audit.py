"""Dependency-free integrity audit of prepared data before full training."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .dataset import DatasetPreparationError, LABEL_MAPPING, load_prepared_records
from .preprocessing import PREPROCESSING_VERSION, normalize_model_text


def prepared_data_integrity(processed_dir: Path) -> dict[str, str]:
    result = {}
    for filename in (
        "records.jsonl", "dataset_report.json", "label_mapping.json",
        "split_manifests/train.jsonl", "split_manifests/validation.jsonl",
        "split_manifests/test.jsonl",
    ):
        digest = hashlib.sha256()
        with (processed_dir / filename).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[filename] = digest.hexdigest()
    return result


def audit_prepared_dataset(processed_dir: Path) -> dict[str, object]:
    """Verify labels/splits/counts/normalization without any model evaluation."""
    records = load_prepared_records(processed_dir)
    report = json.loads((processed_dir / "dataset_report.json").read_text(encoding="utf-8"))
    mapping = json.loads((processed_dir / "label_mapping.json").read_text(encoding="utf-8"))
    if (
        report.get("label_mapping") != LABEL_MAPPING
        or mapping.get("canonical_to_id") != LABEL_MAPPING
        or report.get("preprocessing_version") != PREPROCESSING_VERSION
    ):
        raise DatasetPreparationError("Prepared dataset labels or preprocessing are incompatible")
    if report.get("valid_rows") != len(records):
        raise DatasetPreparationError("Prepared dataset record count does not match metadata")
    split_ids: dict[str, set[str]] = {}
    texts: dict[str, set[str]] = {}
    split_classes = {}
    for split in ("train", "validation", "test"):
        rows = [
            json.loads(line)
            for line in (processed_dir / "split_manifests" / f"{split}.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        ids = [row["record_id"] for row in rows]
        if len(set(ids)) != len(ids) or len(ids) != report["split_distribution"].get(split):
            raise DatasetPreparationError("Split count or record uniqueness is invalid")
        classes: Counter[str] = Counter()
        normalized_texts = set()
        for row in rows:
            record = records.get(row["record_id"])
            if record is None or record.split != split:
                raise DatasetPreparationError("Split references a missing or incorrectly assigned record")
            if (
                LABEL_MAPPING.get(record.label_name) != record.label
                or row.get("label") != record.label
                or row.get("label_name") != record.label_name
            ):
                raise DatasetPreparationError("Split contains an incompatible sentiment label")
            normalized = normalize_model_text(record.text)
            if normalized != record.model_text or not normalized.strip():
                raise DatasetPreparationError("Prepared model_text does not match training preprocessing")
            normalized_texts.add(normalized)
            classes[record.label_name] += 1
        if set(classes) != set(LABEL_MAPPING):
            raise DatasetPreparationError("Every split must contain all three classes")
        split_ids[split] = set(ids)
        texts[split] = normalized_texts
        split_classes[split] = dict(classes)
    if set().union(*split_ids.values()) != set(records):
        raise DatasetPreparationError("Split manifests do not cover exactly the prepared corpus")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_ids[left] & split_ids[right] or texts[left] & texts[right]:
            raise DatasetPreparationError("Normalized-text or record leakage exists across splits")
    classes = Counter(record.label_name for record in records.values())
    if dict(classes) != report.get("class_distribution"):
        raise DatasetPreparationError("Class counts do not match generated metadata")
    return {
        "verified": True,
        "valid_rows": len(records),
        "split_distribution": report["split_distribution"],
        "split_class_distribution": split_classes,
        "label_mapping": LABEL_MAPPING,
        "normalized_text_cross_split_overlap": 0,
        "dataset_revision": report["source_metadata"].get("source_revision"),
        "preprocessing_version": PREPROCESSING_VERSION,
        "sha256": prepared_data_integrity(processed_dir),
    }
