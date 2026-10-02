"""Centralized MuRIL training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "google/muril-base-cased"
    max_length: int = 256
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: float = 3.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    random_seed: int = 42
    early_stopping_patience: int = 2
    smoke_test: bool = False
    smoke_max_steps: int = 2
    smoke_max_train_samples: int = 12
    smoke_max_eval_samples: int = 6

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def smoke_config(base: TrainingConfig | None = None) -> TrainingConfig:
    config = base or TrainingConfig()
    return TrainingConfig(**{**config.as_dict(), "smoke_test": True, "num_epochs": 1.0})
