import json
from pathlib import Path

import pytest

from ml.config import TrainingConfig, smoke_config
from ml.dataset import (
    DatasetPreparationError,
    DatasetRecord,
    LABEL_MAPPING,
    _assign_splits,
    normalize_label,
    prepare_dataset,
    resolve_schema,
)
from ml.metrics import classification_metrics, prediction_records
from ml.preprocessing import PREPROCESSING_VERSION, normalize_model_text


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mahasent_md"


def test_label_normalization_is_canonical() -> None:
    assert normalize_label(-1) == (0, "negative")
    assert normalize_label("0") == (1, "neutral")
    assert normalize_label("positive") == (2, "positive")
    assert LABEL_MAPPING == {"negative": 0, "neutral": 1, "positive": 2}


def test_invalid_label_is_rejected() -> None:
    with pytest.raises(DatasetPreparationError, match="unknown sentiment label"):
        normalize_label("mixed")


def test_schema_resolution_uses_actual_upstream_variants() -> None:
    assert resolve_schema(["tweet", "label", "political"], Path("GT.csv")).text_column == "tweet"
    mapping = resolve_schema(["", "marathi_sentence", "label"], Path("MR.csv"))
    assert mapping.text_column == "marathi_sentence"
    assert mapping.label_column == "label"
    assert resolve_schema(["text", "label", "split"], Path("records.csv")).split_column == "split"


def _synthetic_records(count_per_class: int = 10) -> list[DatasetRecord]:
    records = []
    names = {0: "negative", 1: "neutral", 2: "positive"}
    for label in range(3):
        for index in range(count_per_class):
            records.append(
                DatasetRecord(
                    record_id=f"{label}-{index}", text=f"वाक्य {label} {index}",
                    model_text=f"वाक्य {label} {index}", label=label,
                    label_name=names[label], domain=None, source_file="fixture.csv",
                    source_row=index + 2, split="",
                )
            )
    return records


def test_split_is_deterministic_stratified_and_leak_free() -> None:
    first, strategy = _assign_splits(_synthetic_records(), 17)
    second, _ = _assign_splits(_synthetic_records(), 17)
    assert strategy == "stratified_80_10_10_seed_17"
    assert [(row.record_id, row.split) for row in first] == [(row.record_id, row.split) for row in second]
    for label in range(3):
        rows = [row for row in first if row.label == label]
        assert {row.split for row in rows} == {"train", "validation", "test"}
    split_texts = [{row.model_text for row in first if row.split == split} for split in ("train", "validation", "test")]
    assert not (split_texts[0] & split_texts[1] or split_texts[0] & split_texts[2] or split_texts[1] & split_texts[2])


def test_preparation_writes_report_and_reuses_model_preprocessing(tmp_path: Path) -> None:
    report = prepare_dataset(FIXTURE_DIR, tmp_path / "processed", seed=11)
    assert report["split_strategy"] == "official_upstream"
    assert report["source_metadata"]["source_filenames"] == [
        "MahaSent_All_Test.csv", "MahaSent_All_Train.csv", "MahaSent_All_Val.csv"
    ]
    assert report["preprocessing_version"] == PREPROCESSING_VERSION
    rows = [json.loads(line) for line in (tmp_path / "processed" / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows
    assert all(row["model_text"] == normalize_model_text(row["text"]) for row in rows)
    assert (tmp_path / "processed" / "dataset_report.json").is_file()
    assert (tmp_path / "processed" / "split_manifests" / "test.jsonl").is_file()


def test_metrics_and_prediction_probability_schema() -> None:
    probabilities = [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]]
    metrics = classification_metrics([0, 1, 2], [0, 1, 2], probabilities)
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    rows = prediction_records(_synthetic_records(1), [0, 1, 2], probabilities)
    assert all(abs(sum(row["class_probabilities"].values()) - 1.0) < 1e-6 for row in rows)


def test_smoke_configuration_is_explicit() -> None:
    config = smoke_config(TrainingConfig())
    assert config.smoke_test is True
    assert config.num_epochs == 1.0
    assert config.smoke_max_steps > 0
