"""Verified MahaSent-MD acquisition, discovery, validation, and splitting."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
import subprocess
from typing import Iterable

from .preprocessing import PREPROCESSING_VERSION, normalize_model_text


OFFICIAL_REPO_URL = "https://github.com/l3cube-pune/MarathiNLP.git"
OFFICIAL_REPO_PAGE = "https://github.com/l3cube-pune/MarathiNLP"
OFFICIAL_DATASET_SUBDIR = "L3Cube-MahaSent-MD/MahaSent_All"
OFFICIAL_LICENSE = "CC BY-NC-SA 4.0"
OFFICIAL_RESEARCH_NOTE = (
    "L3Cube states the dataset is released for research purposes only; "
    "non-commercial share-alike attribution terms apply."
)
LABEL_MAPPING = {"negative": 0, "neutral": 1, "positive": 2}
RAW_LABEL_MAPPING = {
    "-1": "negative",
    "0": "neutral",
    "1": "positive",
    "negative": "negative",
    "neg": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "positive": "positive",
    "pos": "positive",
}


class DatasetPreparationError(RuntimeError):
    """Raised for unsafe, ambiguous, or invalid dataset inputs."""


@dataclass(frozen=True)
class SchemaMapping:
    text_column: str
    label_column: str
    domain_column: str | None
    split_column: str | None
    split: str | None


@dataclass(frozen=True)
class DatasetRecord:
    record_id: str
    text: str
    model_text: str
    label: int
    label_name: str
    domain: str | None
    source_file: str
    source_row: int
    split: str


@dataclass(frozen=True)
class DatasetSource:
    source_url: str
    source_revision: str | None
    source_filenames: tuple[str, ...]
    license: str
    research_use: str
    preparation_timestamp: str
    source_type: str


TEXT_COLUMN_NAMES = {
    "text",
    "tweet",
    "marathitext",
    "marathisentence",
    "sentence",
    "content",
    "review",
}
LABEL_COLUMN_NAMES = {"label", "sentiment", "polarity", "target", "class"}
DOMAIN_COLUMN_NAMES = {"domain", "sourcedomain", "category"}
SPLIT_COLUMN_NAMES = {"split", "set", "partition", "subset"}


def _normalized_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _split_from_filename(path: Path) -> str | None:
    name = path.stem.casefold()
    if re.search(r"(^|[-_])train($|[-_])", name):
        return "train"
    if re.search(r"(^|[-_])(valid|validation|val)($|[-_])", name):
        return "validation"
    if re.search(r"(^|[-_])test($|[-_])", name):
        return "test"
    return None


def resolve_schema(fieldnames: Iterable[str] | None, path: Path) -> SchemaMapping:
    """Resolve actual columns from a file rather than assuming one schema."""

    headers = [field for field in (fieldnames or []) if field]
    normalized = {_normalized_header(header): header for header in headers}
    text_column = next(
        (normalized[name] for name in TEXT_COLUMN_NAMES if name in normalized), None
    )
    label_column = next(
        (normalized[name] for name in LABEL_COLUMN_NAMES if name in normalized), None
    )
    if text_column is None:
        non_label = [header for header in headers if _normalized_header(header) not in LABEL_COLUMN_NAMES]
        if len(non_label) == 1:
            text_column = non_label[0]
    if label_column is None or text_column is None:
        raise DatasetPreparationError(
            f"Could not resolve text/label columns in {path}; headers={headers!r}"
        )
    domain_column = next(
        (normalized[name] for name in DOMAIN_COLUMN_NAMES if name in normalized), None
    )
    split_column = next(
        (normalized[name] for name in SPLIT_COLUMN_NAMES if name in normalized), None
    )
    return SchemaMapping(
        text_column=text_column,
        label_column=label_column,
        domain_column=domain_column,
        split_column=split_column,
        split=_split_from_filename(path),
    )


def _split_from_value(value: object) -> str | None:
    normalized = str(value).strip().casefold()
    if normalized == "train":
        return "train"
    if normalized in {"val", "valid", "validation", "dev"}:
        return "validation"
    if normalized == "test":
        return "test"
    return None


def normalize_label(value: object) -> tuple[int, str]:
    """Map upstream -1/0/1 or textual labels to negative=0..positive=2."""

    raw = str(value).strip().casefold()
    label_name = RAW_LABEL_MAPPING.get(raw)
    if label_name is None:
        raise DatasetPreparationError(f"unknown sentiment label: {value!r}")
    return LABEL_MAPPING[label_name], label_name


def _domain_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        normalized = _normalized_header(part)
        if normalized.startswith("mahasent") and normalized not in {"mahasentall"}:
            return part
    return None


def discover_dataset_files(source_path: Path) -> list[Path]:
    """Find only the intended dataset view, refusing ambiguous repository roots."""

    source_path = source_path.resolve()
    if source_path.is_file():
        return [source_path]
    if not source_path.is_dir():
        raise DatasetPreparationError(f"Dataset path does not exist: {source_path}")

    preferred = source_path / OFFICIAL_DATASET_SUBDIR
    if preferred.is_dir():
        return sorted(preferred.glob("*.csv"))

    direct_csv = sorted(source_path.glob("*.csv"))
    if direct_csv:
        return direct_csv

    candidates = sorted(source_path.rglob("*.csv"))
    if not candidates:
        raise DatasetPreparationError(f"No CSV files found under {source_path}")
    parent_dirs = {path.parent for path in candidates}
    if len(parent_dirs) > 1:
        raise DatasetPreparationError(
            "Dataset path contains multiple CSV groups. Pass the explicit "
            "MahaSent_All directory or a single prepared dataset directory."
        )
    return candidates


def _script_statistics(texts: Iterable[str]) -> dict[str, float | int]:
    total_letters = 0
    devanagari_letters = 0
    latin_letters = 0
    rows_with_devanagari = 0
    rows_with_latin = 0
    code_mixed_rows = 0
    row_count = 0
    for text in texts:
        row_count += 1
        has_devanagari = False
        has_latin = False
        for character in text:
            if not character.isalpha():
                continue
            total_letters += 1
            if "\u0900" <= character <= "\u097f":
                devanagari_letters += 1
                has_devanagari = True
            elif ("A" <= character <= "Z") or ("a" <= character <= "z"):
                latin_letters += 1
                has_latin = True
        rows_with_devanagari += int(has_devanagari)
        rows_with_latin += int(has_latin)
        code_mixed_rows += int(has_devanagari and has_latin)
    denominator = max(1, total_letters)
    return {
        "rows": row_count,
        "total_letters": total_letters,
        "devanagari_letters": devanagari_letters,
        "latin_letters": latin_letters,
        "devanagari_ratio": round(devanagari_letters / denominator, 6),
        "latin_ratio": round(latin_letters / denominator, 6),
        "rows_with_devanagari": rows_with_devanagari,
        "rows_with_latin": rows_with_latin,
        "code_mixed_rows": code_mixed_rows,
    }


def _read_records(
    files: list[Path], source_root: Path
) -> tuple[list[DatasetRecord], Counter[str], list[dict[str, str | None]]]:
    valid: list[DatasetRecord] = []
    removed: Counter[str] = Counter()
    mappings: list[dict[str, str | None]] = []
    seen_text: dict[str, DatasetRecord] = {}
    for path in files:
        try:
            decoded = path.read_bytes().decode("utf-8-sig")
        except UnicodeDecodeError:
            removed["invalid_utf8"] += 1
            continue
        except OSError as exc:
            raise DatasetPreparationError(f"Could not read dataset file {path}: {exc}") from exc
        try:
            reader = csv.DictReader(io.StringIO(decoded), strict=True)
            try:
                mapping = resolve_schema(reader.fieldnames, path)
            except csv.Error as exc:
                raise DatasetPreparationError(f"Malformed CSV header in {path}: {exc}") from exc
            mappings.append(
                {
                    "file": str(path.relative_to(source_root)),
                    "text_column": mapping.text_column,
                    "label_column": mapping.label_column,
                    "domain_column": mapping.domain_column,
                    "split_column": mapping.split_column,
                    "split": mapping.split,
                }
            )
            for source_row, row in enumerate(reader, start=2):
                if None in row:
                    removed["malformed_record"] += 1
                    continue
                raw_text = row.get(mapping.text_column)
                if raw_text is None:
                    removed["missing_text"] += 1
                    continue
                if not raw_text.strip():
                    removed["empty_or_whitespace_text"] += 1
                    continue
                try:
                    label, label_name = normalize_label(row.get(mapping.label_column))
                except DatasetPreparationError:
                    removed["unknown_label"] += 1
                    continue
                try:
                    model_text = normalize_model_text(raw_text)
                except (TypeError, UnicodeError):
                    removed["malformed_record"] += 1
                    continue
                duplicate_key = model_text
                if duplicate_key in seen_text:
                    if seen_text[duplicate_key].label != label:
                        removed["duplicate_label_conflict"] += 1
                    else:
                        removed["duplicate_text"] += 1
                    continue
                domain = row.get(mapping.domain_column) if mapping.domain_column else _domain_from_path(path)
                relative_file = str(path.relative_to(source_root))
                record_split = (
                    _split_from_value(row.get(mapping.split_column))
                    if mapping.split_column
                    else mapping.split
                ) or ""
                record = DatasetRecord(
                    record_id=f"{relative_file}:{source_row}",
                    text=raw_text,
                    model_text=model_text,
                    label=label,
                    label_name=label_name,
                    domain=domain.strip() if isinstance(domain, str) and domain.strip() else None,
                    source_file=relative_file,
                    source_row=source_row,
                    split=record_split,
                )
                seen_text[duplicate_key] = record
                valid.append(record)
        except csv.Error as exc:
            raise DatasetPreparationError(f"Malformed CSV records in {path}: {exc}") from exc
    if not valid:
        raise DatasetPreparationError("No valid labeled records were found")
    return valid, removed, mappings


def _stratified_split(records: list[DatasetRecord], seed: int) -> list[DatasetRecord]:
    by_label: dict[int, list[DatasetRecord]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)
    rng = random.Random(seed)
    assigned: list[DatasetRecord] = []
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        train_count = int(len(group) * 0.8)
        validation_count = int(len(group) * 0.1)
        for index, record in enumerate(group):
            split = "train" if index < train_count else "validation" if index < train_count + validation_count else "test"
            assigned.append(DatasetRecord(**{**asdict(record), "split": split}))
    return assigned


def _assign_splits(records: list[DatasetRecord], seed: int) -> tuple[list[DatasetRecord], str]:
    supplied = {record.split for record in records}
    if supplied == {"train", "validation", "test"}:
        return records, "official_upstream"
    return _stratified_split(records, seed), f"stratified_80_10_10_seed_{seed}"


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_remote(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "config", "--get", "remote.origin.url"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def acquire_official_dataset(raw_dir: Path, revision: str = "main") -> tuple[Path, str]:
    """Clone only the verified upstream dataset view and return resolved SHA."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = raw_dir / "l3cube-marathinlp"
    if clone_dir.exists():
        raise DatasetPreparationError(
            f"Raw clone already exists at {clone_dir}; remove it or choose another raw directory"
        )
    try:
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", "--branch", revision, OFFICIAL_REPO_URL, str(clone_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(clone_dir), "sparse-checkout", "init", "--cone"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(clone_dir), "sparse-checkout", "set", OFFICIAL_DATASET_SUBDIR],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise DatasetPreparationError(f"Could not acquire official MahaSent-MD source: {detail}") from exc
    resolved = _git_revision(clone_dir)
    if resolved is None:
        raise DatasetPreparationError("Official clone did not expose a Git revision")
    return clone_dir / OFFICIAL_DATASET_SUBDIR, resolved


def prepare_dataset(
    source_path: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    source_url: str | None = None,
    source_revision: str | None = None,
) -> dict[str, object]:
    """Prepare normalized records, split manifests, and a provenance report."""

    files = discover_dataset_files(source_path)
    resolved_source = source_path.resolve()
    source_root = resolved_source.parent if resolved_source.is_file() else resolved_source
    records, removed, mappings = _read_records(files, source_root)
    records, split_strategy = _assign_splits(records, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_rows = [asdict(record) for record in records]
    _write_jsonl(output_dir / "records.jsonl", record_rows)
    for split in ("train", "validation", "test"):
        split_rows = [
            {
                "record_id": record.record_id,
                "label": record.label,
                "label_name": record.label_name,
            }
            for record in records
            if record.split == split
        ]
        _write_jsonl(output_dir / "split_manifests" / f"{split}.jsonl", split_rows)
    resolved_revision = source_revision
    resolved_url = source_url or f"local://{resolved_source}"
    source = DatasetSource(
        source_url=resolved_url,
        source_revision=resolved_revision,
        source_filenames=tuple(sorted({record.source_file for record in records})),
        license=OFFICIAL_LICENSE if resolved_url.rstrip("/").endswith("MarathiNLP.git") else "unspecified-local-source",
        research_use=OFFICIAL_RESEARCH_NOTE if resolved_url.rstrip("/").endswith("MarathiNLP.git") else "Local source; verify licensing before use.",
        preparation_timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        source_type="official_git" if resolved_url.rstrip("/").endswith("MarathiNLP.git") else "local_path",
    )
    labels = Counter(record.label_name for record in records)
    split_counts = Counter(record.split for record in records)
    source_metadata = asdict(source)
    source_metadata["source_filenames"] = list(source.source_filenames)
    report = {
        "dataset_name": "L3Cube-MahaSent-MD/MahaSent_All",
        "total_rows": sum(labels.values()) + sum(removed.values()),
        "valid_rows": len(records),
        "removed_rows_by_reason": dict(sorted(removed.items())),
        "class_distribution": dict(sorted(labels.items())),
        "split_distribution": dict(sorted(split_counts.items())),
        "script_statistics": _script_statistics(record.model_text for record in records),
        "duplicate_count": removed.get("duplicate_text", 0) + removed.get("duplicate_label_conflict", 0),
        "schema_mappings": mappings,
        "split_strategy": split_strategy,
        "seed": seed,
        "label_mapping": LABEL_MAPPING,
        "raw_label_mapping": RAW_LABEL_MAPPING,
        "preprocessing_version": PREPROCESSING_VERSION,
        "source_metadata": source_metadata,
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "label_mapping.json").write_text(
        json.dumps(
            {
                "canonical_to_id": LABEL_MAPPING,
                "id_to_canonical": {str(value): key for key, value in LABEL_MAPPING.items()},
                "raw_to_canonical": RAW_LABEL_MAPPING,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def load_prepared_records(processed_dir: Path) -> dict[str, DatasetRecord]:
    path = processed_dir / "records.jsonl"
    if not path.is_file():
        raise DatasetPreparationError(f"Prepared records not found: {path}")
    records: dict[str, DatasetRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record = DatasetRecord(**row)
            records[record.record_id] = record
    return records


def load_split_records(processed_dir: Path, split: str) -> list[DatasetRecord]:
    records = load_prepared_records(processed_dir)
    manifest = processed_dir / "split_manifests" / f"{split}.jsonl"
    if not manifest.is_file():
        raise DatasetPreparationError(f"Split manifest not found: {manifest}")
    selected: list[DatasetRecord] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            record_id = json.loads(line)["record_id"]
            try:
                selected.append(records[record_id])
            except KeyError as exc:
                raise DatasetPreparationError(
                    f"Split manifest references unknown record: {record_id}"
                ) from exc
    return selected


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_model_text(text).encode("utf-8")).hexdigest()
