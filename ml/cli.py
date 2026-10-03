"""Command-line entry points for MahaPulse dataset and model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import TrainingConfig, smoke_config
from .dataset import DatasetPreparationError, acquire_official_dataset, prepare_dataset
from .evaluate import evaluate_artifact
from .promotion import PromotionError, promote_artifact
from .train import TrainingError, train_model
from .topics import TopicTrainingConfig, TopicTrainingError, inspect_topic_artifact, train_topic_model


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
    train_parser.add_argument(
        "--resume-from-checkpoint",
        type=_path,
        help="resume an interrupted run from a validated Trainer checkpoint",
    )

    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved artifact on held-out test data")
    evaluate.add_argument("--artifact-dir", type=_path, required=True)
    evaluate.add_argument("--processed-dir", type=_path, default=Path("ml/data/processed/mahasent-md"))

    inspect = subparsers.add_parser("inspect-artifact", help="print an artifact manifest")
    inspect.add_argument("artifact_dir", type=_path)

    promote = subparsers.add_parser(
        "promote-artifact", help="verify and promote a full artifact after real API validation"
    )
    promote.add_argument("--artifact-dir", type=_path, required=True)
    promote.add_argument(
        "--validation-report",
        type=_path,
        required=True,
        help="passing prepromotion report bound to this artifact and its SHA-256 metadata",
    )
    promote.add_argument(
        "--api-integration-validated",
        action="store_true",
        help="attest that real FastAPI integration passed before promotion",
    )

    topics = subparsers.add_parser("topics", help="offline BERTopic artifact operations")
    topic_commands = topics.add_subparsers(dest="topics_command", required=True)
    topic_train = topic_commands.add_parser("train", help="train BERTopic on a prepared corpus")
    topic_train.add_argument("--processed-dir", type=_path, default=Path("ml/data/processed/mahasent-md"))
    topic_train.add_argument("--artifact-root", type=_path, default=Path("ml/artifacts"))
    topic_train.add_argument("--topic-version", default="bertopic-mahasent-md-v1")
    topic_train.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    topic_train.add_argument("--min-topic-size", type=int, default=10)
    topic_train.add_argument("--nr-topics", type=int)
    topic_train.add_argument("--random-seed", type=int, default=42)
    topic_train.add_argument("--smoke", action="store_true")
    topic_inspect = topic_commands.add_parser("inspect", help="print a topic artifact manifest")
    topic_inspect.add_argument("artifact_dir", type=_path)
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
        return {
            "artifact_dir": str(
                train_model(
                    args.processed_dir,
                    args.artifact_root,
                    config,
                    version,
                    resume_from_checkpoint=args.resume_from_checkpoint,
                )
            )
        }
    if args.command == "evaluate":
        return evaluate_artifact(args.artifact_dir, args.processed_dir)
    if args.command == "inspect-artifact":
        return json.loads((args.artifact_dir / "model_manifest.json").read_text(encoding="utf-8"))
    if args.command == "promote-artifact":
        return promote_artifact(
            args.artifact_dir,
            api_integration_validated=args.api_integration_validated,
            validation_report=args.validation_report,
        )
    if args.command == "topics":
        if args.topics_command == "inspect":
            return inspect_topic_artifact(args.artifact_dir)
        config = TopicTrainingConfig(
            embedding_model=args.embedding_model,
            min_topic_size=args.min_topic_size,
            nr_topics=args.nr_topics,
            random_seed=args.random_seed,
            smoke_test=args.smoke,
        )
        return {
            "artifact_dir": str(
                train_topic_model(
                    args.processed_dir,
                    args.artifact_root,
                    config,
                    args.topic_version,
                )
            )
        }
    raise ValueError(f"Unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = _run(args)
    except (
        DatasetPreparationError,
        TrainingError,
        TopicTrainingError,
        PromotionError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
