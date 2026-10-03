"""Read-only, lightweight safety checks for resumable MuRIL checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config import TrainingConfig
from .dataset import LABEL_MAPPING


class CheckpointError(RuntimeError):
    """A checkpoint cannot safely resume the recorded training experiment."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"Resume evidence is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"Resume evidence must be an object: {path.name}")
    return value


def _nonempty_local_file(path: Path, checkpoint: Path) -> bool:
    return path.is_file() and path.resolve().parent == checkpoint and path.stat().st_size > 0


def validate_resume_checkpoint(
    artifact_dir: Path, checkpoint_dir: Path, config: TrainingConfig
) -> Path:
    """Validate optimizer/RNG continuity and immutable experiment configuration.

    This function only reads small JSON metadata and filesystem properties. It
    never imports Torch, deserializes pickle checkpoints, or loads test records.
    """
    artifact_dir = artifact_dir.resolve()
    training_root = (artifact_dir / "_trainer").resolve()
    checkpoint = checkpoint_dir.resolve()
    if not checkpoint.is_dir() or checkpoint.parent != training_root:
        raise CheckpointError("Resume checkpoint is outside this artifact training directory")
    if not checkpoint.name.startswith("checkpoint-") or not checkpoint.name.removeprefix("checkpoint-").isdigit():
        raise CheckpointError("Resume checkpoint name must identify its training step")
    if any((artifact_dir / name).exists() for name in ("metrics.json", "predictions.jsonl", "model_manifest.json")):
        raise CheckpointError("A finalized artifact cannot be resumed or evaluated again")
    if any(path.exists() for path in (
        training_root / "final-test.lock", artifact_dir / "_evaluation" / "final-test.lock",
    )):
        raise CheckpointError("Final test evaluation was claimed; interruption review is required")

    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "config.json"):
        if not _nonempty_local_file(checkpoint / name, checkpoint):
            raise CheckpointError(f"Resume checkpoint is missing a local nonempty {name}")
    if not any(_nonempty_local_file(path, checkpoint) for path in checkpoint.glob("rng_state*.pth")):
        raise CheckpointError("Resume checkpoint is missing RNG state")
    weights = list(checkpoint.glob("model*.safetensors")) + list(checkpoint.glob("pytorch_model*.bin"))
    if not weights or not all(_nonempty_local_file(path, checkpoint) for path in weights):
        raise CheckpointError("Resume checkpoint is missing local model weights")

    model_config = _read_json(checkpoint / "config.json")
    if (
        model_config.get("label2id") != LABEL_MAPPING
        or model_config.get("id2label") != {str(value): key for key, value in LABEL_MAPPING.items()}
        or model_config.get("model_type") != "bert"
    ):
        raise CheckpointError("Resume model architecture or canonical labels are incompatible")
    state = _read_json(checkpoint / "trainer_state.json")
    step = state.get("global_step")
    epoch = state.get("epoch")
    if type(step) is not int or step <= 0 or step != int(checkpoint.name.removeprefix("checkpoint-")):
        raise CheckpointError("Resume trainer step differs from the checkpoint directory")
    if (
        type(epoch) not in (int, float) or not math.isfinite(epoch)
        or epoch <= 0 or epoch > config.num_epochs
        or state.get("num_train_epochs") != config.num_epochs
        or state.get("train_batch_size") != config.train_batch_size
    ):
        raise CheckpointError("Resume trainer epoch or batch configuration is incompatible")

    run_paths = sorted(training_root.glob("run-*.json"))
    if not run_paths:
        raise CheckpointError("Resume requires recorded immutable training configuration")
    for run_path in run_paths:
        if _read_json(run_path).get("training_config") != config.as_dict():
            raise CheckpointError("Resume training configuration differs from the recorded experiment")
    arguments = _read_json(training_root / "effective_training_arguments.json")
    expected = {
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_epochs,
        "weight_decay": config.weight_decay,
        "seed": config.random_seed,
        "data_seed": config.random_seed,
        "metric_for_best_model": "eval_macro_f1",
        "greater_is_better": True,
        "load_best_model_at_end": True,
    }
    if any(arguments.get(key) != value for key, value in expected.items()):
        raise CheckpointError("Resume effective arguments differ from validation-based training")
    return checkpoint
