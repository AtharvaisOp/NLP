import json
from dataclasses import replace
from pathlib import Path

import pytest

from ml.checkpoints import CheckpointError, validate_resume_checkpoint
from ml.config import TrainingConfig
from ml.dataset import LABEL_MAPPING
from ml.train import _write_json


def checkpoint_fixture(tmp_path):
    artifact = tmp_path / "artifact"
    checkpoint = artifact / "_trainer" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    config = TrainingConfig(train_batch_size=4, eval_batch_size=8)
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
        (checkpoint / name).write_bytes(b"fixture, never deserialized")
    _write_json(checkpoint / "config.json", {
        "label2id": LABEL_MAPPING,
        "id2label": {str(value): key for key, value in LABEL_MAPPING.items()},
        "model_type": "bert",
    })
    _write_json(checkpoint / "trainer_state.json", {
        "global_step": 10, "epoch": 1.0, "num_train_epochs": 3.0, "train_batch_size": 4,
    })
    _write_json(checkpoint.parent / "run-fixture.json", {"training_config": config.as_dict()})
    _write_json(checkpoint.parent / "effective_training_arguments.json", {
        "per_device_train_batch_size": 4, "per_device_eval_batch_size": 8,
        "learning_rate": config.learning_rate, "num_train_epochs": 3.0,
        "weight_decay": config.weight_decay, "seed": 42, "data_seed": 42,
        "metric_for_best_model": "eval_macro_f1", "greater_is_better": True,
        "load_best_model_at_end": True,
    })
    return artifact, checkpoint, config


def test_valid_checkpoint_can_resume_without_loading_pickle(tmp_path):
    artifact, checkpoint, config = checkpoint_fixture(tmp_path)
    assert validate_resume_checkpoint(artifact, checkpoint, config) == checkpoint.resolve()


@pytest.mark.parametrize("name", ["optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors", "config.json", "trainer_state.json"])
def test_checkpoint_requires_complete_resume_state(tmp_path, name):
    artifact, checkpoint, config = checkpoint_fixture(tmp_path)
    (checkpoint / name).unlink()
    with pytest.raises(CheckpointError):
        validate_resume_checkpoint(artifact, checkpoint, config)


@pytest.mark.parametrize("name", ["metrics.json", "predictions.jsonl", "model_manifest.json"])
def test_finalized_artifact_cannot_resume(tmp_path, name):
    artifact, checkpoint, config = checkpoint_fixture(tmp_path)
    (artifact / name).write_text("fixture", encoding="utf-8")
    with pytest.raises(CheckpointError, match="finalized"):
        validate_resume_checkpoint(artifact, checkpoint, config)


@pytest.mark.parametrize("directory", ["_trainer", "_evaluation"])
def test_claimed_final_test_requires_interruption_review(tmp_path, directory):
    artifact, checkpoint, config = checkpoint_fixture(tmp_path)
    (artifact / directory).mkdir(exist_ok=True)
    (artifact / directory / "final-test.lock").write_text("claimed", encoding="utf-8")
    with pytest.raises(CheckpointError, match="interruption review"):
        validate_resume_checkpoint(artifact, checkpoint, config)


def test_checkpoint_must_belong_to_this_artifact(tmp_path):
    artifact, checkpoint, config = checkpoint_fixture(tmp_path)
    with pytest.raises(CheckpointError, match="outside"):
        validate_resume_checkpoint(tmp_path / "other-artifact", checkpoint, config)


@pytest.mark.parametrize("change", ["max_length", "learning_rate", "random_seed", "train_batch_size", "model_name"])
def test_resume_refuses_silent_experiment_changes(tmp_path, change):
    artifact, checkpoint, config = checkpoint_fixture(tmp_path)
    different = replace(config, **{change: "different/model" if change == "model_name" else getattr(config, change) * 2})
    with pytest.raises(CheckpointError, match="configuration"):
        validate_resume_checkpoint(artifact, checkpoint, different)


@pytest.mark.parametrize("change", ["swapped_labels", "different_step", "test_selection", "missing_run"])
def test_resume_metadata_must_be_consistent(tmp_path, change):
    artifact, checkpoint, config = checkpoint_fixture(tmp_path)
    if change == "swapped_labels":
        path = checkpoint / "config.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["id2label"]["0"] = "positive"
    elif change == "different_step":
        path = checkpoint / "trainer_state.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["global_step"] = 11
    elif change == "test_selection":
        path = checkpoint.parent / "effective_training_arguments.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["metric_for_best_model"] = "test_macro_f1"
    else:
        (checkpoint.parent / "run-fixture.json").unlink()
        with pytest.raises(CheckpointError, match="recorded immutable"):
            validate_resume_checkpoint(artifact, checkpoint, config)
        return
    _write_json(path, value)
    with pytest.raises(CheckpointError):
        validate_resume_checkpoint(artifact, checkpoint, config)
