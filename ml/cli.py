"""Command-line entry points for MahaPulse dataset and model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import TrainingConfig, smoke_config
from .dataset import DatasetPreparationError, acquire_official_dataset, prepare_dataset
from .evaluate import evaluate_artifact
from .train import TrainingError, train_model


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MahaPulse Phase 3 ML pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="acquire or validate MahaSent-MD")
    prepare.add_argument("--local-path", type=_path, help="explicit local CSV, dataset directory, or official clone")
    prepare.add_argument("--raw-dir", type=_path, default=Path("ml/data/raw"))
    prepare.add_argument("--output-dir", type=_path, default=Path("ml/data/processed/mahasent-md"))
    prepare.add_argument("--revision", default="main", help="official Git branch or tag to acquire")
    prepare.add_argument("--seed", type=int, default=42)

    train_parser = subparsers.add_parser("train", help="fine-tune MuRIL")
    mode_group = train_parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--smoke", action="store_true", help="run the tiny deterministic smoke configuration")
    mode_group.add_argument("--full", action="store_true", help="run complete training with validation-based selection")
    train_parser.add_argument("--model-version", default="muril-mahasent-md-smoke")
    train_parser.add_argument("--processed-dir", type=_path, default=Path("ml/data/processed/mahasent-md"))
    train_parser.add_argument("--artifact-root", type=_path, default=Path("ml/artifacts"))
    train_parser.add_argument("--model-name", default="google/muril-base-cased")
    train_parser.add_argument("--max-length", type=int, default=256)
    train_parser.add_argument("--train-batch-size", type=int, default=16)
    train_parser.add_argument("--eval-batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--num-epochs", type=float, default=3.0)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.1)
    train_parser.add_argument("--random-seed", type=int, default=42)
    train_parser.add_argument("--early-stopping-patience", type=int, default=2)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved artifact on held-out test data")
    evaluate.add_argument("--artifact-dir", type=_path, required=True)
    evaluate.add_argument("--processed-dir", type=_path, default=Path("ml/data/processed/mahasent-md"))

    inspect = subparsers.add_parser("inspect-artifact", help="print an artifact manifest")
    inspect.add_argument("artifact_dir", type=_path)
    return parser


def _run(args: argparse.Namespace) -> object:
    if args.command == "prepare":
        if args.local_path:
            report = prepare_dataset(args.local_path, args.output_dir, seed=args.seed)
        else:
            source_path, revision = acquire_official_dataset(args.raw_dir, args.revision)
            report = prepare_dataset(
                source_path,
                args.output_dir,
                seed=args.seed,
                source_url="https://github.com/l3cube-pune/MarathiNLP.git",
                source_revision=revision,
            )
        return report
    if args.command == "train":
        smoke = args.smoke
        config = TrainingConfig(
            model_name=args.model_name,
            max_length=args.max_length,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            random_seed=args.random_seed,
            early_stopping_patience=args.early_stopping_patience,
        )
        if smoke:
            config = smoke_config(config)
        version = args.model_version
        if not smoke and version == "muril-mahasent-md-smoke":
            version = "muril-mahasent-md-full"
        return {"artifact_dir": str(train_model(args.processed_dir, args.artifact_root, config, version))}
    if args.command == "evaluate":
        return evaluate_artifact(args.artifact_dir, args.processed_dir)
    if args.command == "inspect-artifact":
        return json.loads((args.artifact_dir / "model_manifest.json").read_text(encoding="utf-8"))
    raise ValueError(f"Unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = _run(args)
    except (DatasetPreparationError, TrainingError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
