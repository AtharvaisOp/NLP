import json
from pathlib import Path

import pytest

from ml.audit import audit_prepared_dataset
from ml.dataset import DatasetPreparationError, prepare_dataset

FIXTURES = Path(__file__).parent / "fixtures" / "mahasent_md"


def test_audit_checks_prepared_split_counts_and_hashes(tmp_path: Path) -> None:
    prepare_dataset(FIXTURES, tmp_path)
    result = audit_prepared_dataset(tmp_path)
    assert result["verified"] is True
    assert result["normalized_text_cross_split_overlap"] == 0
    assert len(result["sha256"]) == 6


def test_audit_rejects_inconsistent_manifest_labels(tmp_path: Path) -> None:
    prepare_dataset(FIXTURES, tmp_path)
    path = tmp_path / "split_manifests" / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["label"] = (rows[0]["label"] + 1) % 3
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(DatasetPreparationError, match="incompatible sentiment label"):
        audit_prepared_dataset(tmp_path)
