"""Validate an already trained local MuRIL artifact and the complete API flow.

Run after training and the one final held-out evaluation have completed:
    python scripts/validate-production-model.py --artifact-dir PATH --device cuda
Run again after explicit promotion with --expect-promoted. This script does not
train, select checkpoints, read dataset splits, or recompute held-out metrics.
All model loads are local-only; evidence is saved separately from model files.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import threading
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LABEL_MAPPING = {"negative": 0, "neutral": 1, "positive": 2}
DATASET_REVISION = "8ee29fa1329d6a841030eb46659d3c10614b5e59"
EXAMPLES = (
    ("positive", "हा मोबाईल खूप चांगला आहे."),
    ("negative", "ही सेवा अत्यंत खराब आहे."),
    ("neutral", "आज दुकान सकाळी दहा वाजता उघडले."),
    ("code_mixed", "हा phone चांगला आहे पण battery backup खराब आहे."),
)


class MemorySampler:
    """Measure this validator's resident memory without requiring psutil in CI."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_bytes = 0
        self._process = None

    def start(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        self._process = psutil.Process()

        def sample() -> None:
            while not self._stop.is_set():
                try:
                    self._peak_bytes = max(self._peak_bytes, self._process.memory_info().rss)
                except OSError:
                    return
                self._stop.wait(0.05)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        result: dict[str, Any] = {"rss_sample_interval_seconds": 0.05}
        if self._process is not None:
            current = self._process.memory_info().rss
            result.update({"process_rss_peak_mib": round(max(current, self._peak_bytes) / 1048576, 2),
                           "process_rss_final_mib": round(current / 1048576, 2)})
        else:
            result["process_rss"] = "unavailable: optional psutil dependency is absent"
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            result.update({"cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1048576, 2),
                           "cuda_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1048576, 2),
                           "gpu": torch.cuda.get_device_name()})
        return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_probabilities(sentiment: dict[str, Any]) -> None:
    probabilities = sentiment.get("probabilities")
    require(isinstance(probabilities, dict), "three sentiment probabilities are required")
    require(set(probabilities) == set(LABEL_MAPPING), "probabilities use noncanonical labels")
    values = list(probabilities.values())
    require(
        all(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1 for value in values),
        "sentiment probabilities must be finite values between zero and one",
    )
    require(math.isclose(sum(values), 1, abs_tol=1e-6), "probabilities do not sum to one")
    label = sentiment.get("label")
    require(label in LABEL_MAPPING, "prediction label is not canonical")
    require(probabilities[label] == max(values), "prediction label disagrees with probabilities")
    require(
        isinstance(sentiment.get("confidence"), (int, float))
        and math.isclose(sentiment["confidence"], probabilities[label], abs_tol=1e-6),
        "prediction confidence disagrees with probabilities",
    )


def verify_artifact(artifact: Path, expect_promoted: bool) -> dict[str, Any]:
    from backend.app.services.sentiment.muril import _validate_artifact
    from ml.preprocessing import PREPROCESSING_VERSION
    from ml.artifact_validation import validate_production_evidence
    from ml.promotion import REQUIRED_FILES
    from ml.train import _artifact_integrity

    require(artifact.is_dir(), "artifact directory does not exist")
    for filename in REQUIRED_FILES:
        require((artifact / filename).is_file(), f"missing artifact file: {filename}")
    metadata = _validate_artifact(artifact, allow_smoke_model=False)
    manifest = json.loads((artifact / "model_manifest.json").read_text(encoding="utf-8"))
    training = json.loads((artifact / "training_config.json").read_text(encoding="utf-8"))
    dataset = json.loads((artifact / "dataset_report.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
    require(metadata.smoke_test is False and training.get("smoke_test") is False,
            "full validation rejects smoke artifacts")
    require(metadata.production_ready is expect_promoted, "unexpected production_ready flag")
    require(manifest.get("project") == "MahaPulse", "wrong artifact project")
    require(manifest.get("base_model") == "google/muril-base-cased", "wrong base model")
    require(manifest.get("dataset_revision") == DATASET_REVISION, "unexpected dataset revision")
    require(manifest.get("preprocessing_version") == PREPROCESSING_VERSION,
            "training and inference preprocessing versions differ")
    require(manifest.get("training_config") == training, "training configuration evidence differs")
    require(manifest.get("lifecycle_validation", {}).get("training_completed") is True,
            "training completion has not been recorded")
    require(isinstance(metrics.get("validation"), dict) and isinstance(metrics.get("test"), dict),
            "final validation and held-out metrics must already exist")
    require(isinstance(manifest.get("evaluation_summary"), dict), "final test summary is missing")
    counts = dataset.get("split_distribution", {})
    require(counts == {"train": 47730, "validation": 5922, "test": 6744},
            "prepared dataset counts differ from audited full corpus")
    held_out_count = validate_production_evidence(artifact, manifest)
    actual_integrity = _artifact_integrity(artifact)
    require(manifest.get("artifact_integrity") == actual_integrity,
            "SHA-256 artifact integrity metadata does not match all artifact files")
    return {
        "model_version": metadata.model_version,
        "base_model": metadata.base_model,
        "smoke_test": metadata.smoke_test,
        "production_ready": metadata.production_ready,
        "dataset_revision": DATASET_REVISION,
        "split_distribution": counts,
        "preprocessing_version": PREPROCESSING_VERSION,
        "integrity_verified": True,
        "artifact_integrity": actual_integrity,
        "manifest_sha256": sha256(artifact / "model_manifest.json"),
        "held_out_prediction_records_verified": held_out_count,
    }


def independently_reload(artifact: Path, device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from ml.preprocessing import normalize_model_text

    started = perf_counter()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    require(device != "cuda" or torch.cuda.is_available(), "requested CUDA is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(artifact, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(artifact, local_files_only=True)
    require(model.config.label2id == LABEL_MAPPING, "reloaded model has incompatible labels")
    require(model.config.num_labels == 3, "reloaded model does not have three classes")
    model.to(device)
    model.eval()
    manifest = json.loads((artifact / "model_manifest.json").read_text(encoding="utf-8"))
    examples = []
    for category, text in EXAMPLES:
        model_text = normalize_model_text(text)
        encoded = tokenizer(model_text, truncation=True, max_length=manifest["max_length"],
                            padding=False, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(model(**encoded).logits, dim=-1).cpu().tolist()[0]
        by_label = {label: float(probabilities[index]) for label, index in LABEL_MAPPING.items()}
        label = max(by_label, key=by_label.__getitem__)
        sentiment = {"label": label, "confidence": by_label[label], "probabilities": by_label}
        validate_probabilities(sentiment)
        examples.append({"integration_example": category, "original_text": text,
                         "model_text": model_text, "sentiment": sentiment})
    del model, tokenizer, encoded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"passed": True, "local_files_only": True, "device": device,
            "elapsed_seconds": round(perf_counter() - started, 3), "examples": examples}


@contextmanager
def database_environment(url: str):
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = url
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous


def migrate_database(path: Path) -> str:
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, text

    url = "sqlite:///" + path.resolve().as_posix()
    config = Config(str(ROOT / "backend" / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "backend" / "alembic"))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    with database_environment(url):
        command.upgrade(config, "head")
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            version = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
    finally:
        engine.dispose()
    require(bool(version), "SQLite migration revision is missing")
    return str(version)


def batch_csv() -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["text"])
    writer.writerows([[text] for _, text in EXAMPLES] + [[""]])
    return buffer.getvalue().encode("utf-8")


def validate_application(settings: Any, output: Path, independent: dict[str, Any],
                         expect_promoted: bool, keywords: str) -> dict[str, Any]:
    from fastapi.testclient import TestClient
    from ml.preprocessing import normalize_model_text
    from backend.app.main import create_app
    from backend.app.services.enrichment.disabled import DisabledKeywordService

    app = create_app(settings)
    keyword_selection = {"requested": keywords, "selected": settings.keyword_backend}
    if keywords == "auto":
        state = app.state.orchestrator.keyword_service.metadata().state
        keyword_selection["cached_keybert_state"] = state
        if state != "ready":
            app.state.orchestrator.keyword_service = DisabledKeywordService()
            keyword_selection["selected"] = "disabled"
            keyword_selection["reason"] = "cached KeyBERT dependencies or embedding model unavailable"
    elif keywords == "keybert":
        require(app.state.orchestrator.keyword_service.metadata().state == "ready",
                "explicitly requested cached KeyBERT is unavailable")

    operations: dict[str, Any] = {}
    examples = []
    latencies = {}
    with TestClient(app) as client:
        def call(method: str, path: str, expected: int = 200, **kwargs):
            started = perf_counter()
            response = client.request(method, path, **kwargs)
            elapsed = round((perf_counter() - started) * 1000, 3)
            latencies.setdefault(path, []).append(elapsed)
            require(response.status_code == expected,
                    f"{method} {path}: expected HTTP {expected}, got {response.status_code}")
            return response

        for endpoint in ("/health", "/ready", "/v1/model-info"):
            operations[endpoint] = call("GET", endpoint).json()
        require(operations["/health"]["status"] == "ok", "API is not live")
        ready = operations["/ready"]["services"]
        require(ready["sentiment"]["state"] == "ready", "real sentiment service did not load")
        require(ready["sentiment"]["smoke_test"] is False, "API loaded a smoke model")
        require(ready["sentiment"]["production_ready"] is expect_promoted,
                "sentiment readiness does not match promotion state")
        require(ready["database"]["state"] == "ready", "migrated SQLite database is not ready")
        model_info = operations["/v1/model-info"]
        sentiment_info = model_info["sentiment_model"]
        require(sentiment_info["backend"] == "muril" and sentiment_info["state"] == "ready",
                "API did not use the real MuRIL backend")
        require(sentiment_info["smoke_test"] is False, "model-info reported a smoke model")
        require(sentiment_info["production_ready"] is expect_promoted,
                "model-info does not match artifact promotion state")
        require(model_info["summary_service"]["provider"] == "extractive",
                "extractive summary is not enabled")
        require(model_info["topic_service"]["state"] == "disabled", "optional topics must be disabled")

        for (category, text), reloaded in zip(EXAMPLES, independent["examples"], strict=True):
            response = call("POST", "/v1/analyze", json={"text": text},
                            headers={"X-Request-ID": f"full-model-{category}"})
            body = response.json()
            require(body["original_text"] == text, "API corrupted UTF-8 original text")
            require(body["model_text"] == normalize_model_text(text) == reloaded["model_text"],
                    "API model_text differs from training preprocessing")
            require(body["request_id"] == response.headers["X-Request-ID"] == f"full-model-{category}",
                    "request ID was not propagated")
            require(body["language"]["is_code_mixed"] is (category == "code_mixed"),
                    "Unicode/code-mixed language regression")
            require(body["meta"]["model_version"] == sentiment_info["version"],
                    "analysis model version differs from the real loaded artifact")
            require(not any("mock" in warning.lower() or "smoke artifact" in warning.lower()
                            for warning in body["meta"]["warnings"]),
                    "real inference unexpectedly used mock or smoke output")
            validate_probabilities(body["sentiment"])
            for label in LABEL_MAPPING:
                require(math.isclose(body["sentiment"]["probabilities"][label],
                                     reloaded["sentiment"]["probabilities"][label], abs_tol=1e-4),
                        "API probabilities differ from independent artifact inference")
            examples.append({"integration_example": category, "response": body})

        fixture = batch_csv()
        (output / "integration-marathi.csv").write_bytes(fixture)
        batch = call("POST", "/v1/analyze/batch", params={"text_column": "text"},
                     files={"file": ("integration-marathi.csv", fixture, "text/csv")}).json()
        require((batch["total_documents"], batch["successful_documents"], batch["failed_documents"],
                 batch["status"]) == (5, 4, 1, "partial"), "batch did not return four successes and one failure")
        require(batch["model_version"] == sentiment_info["version"], "batch used a different artifact")
        session_id = batch["session_id"]
        documents = []
        for offset in (0, 2, 4):
            page = call("GET", f"/v1/analyses/{session_id}", params={"limit": 2, "offset": offset}).json()
            require(page["total_documents"] == 5 and page["offset"] == offset and page["limit"] == 2,
                    "session pagination metadata is incorrect")
            require(len(page["documents"]) == min(2, 5 - offset), "pagination returned wrong row count")
            documents.extend(page["documents"])
        require(len({item["id"] for item in documents}) == 5, "pagination duplicated stored rows")
        require([item["row_index"] for item in documents] == list(range(5)), "stored row order changed")
        for index, (_, text) in enumerate(EXAMPLES):
            document = documents[index]
            require(document["status"] == "success" and document["original_text"] == text,
                    "stored Marathi example is missing or corrupted")
            require(document["model_text"] == normalize_model_text(text), "stored model_text differs")
            validate_probabilities(document["sentiment"])
        require(documents[4]["status"] == "failed" and documents[4]["error_code"] == "invalid_input",
                "blank row did not produce the expected isolated failure")
        require("Traceback" not in json.dumps(documents[4]), "failed row leaked internal traceback")

        analytics = call("GET", f"/v1/analyses/{session_id}/analytics").json()
        require((analytics["total"], analytics["successful"], analytics["failed"]) == (5, 4, 1),
                "analytics document counts are incorrect")
        require(sum(item["count"] for item in analytics["sentiment"].values()) == 4,
                "analytics sentiment counts are incorrect")
        require(math.isclose(sum(item["percentage"] for item in analytics["sentiment"].values()),
                             100, abs_tol=1e-6), "analytics sentiment percentages are incorrect")
        require(analytics["language"]["code_mixed_count"] == 1, "analytics lost the code-mixed row")
        require(analytics["null_topic_count"] == 4 and analytics["summary"] is None,
                "optional topics or analytics summary behavior changed")

        json_export = call("GET", f"/v1/analyses/{session_id}/export", params={"format": "json"})
        csv_export = call("GET", f"/v1/analyses/{session_id}/export", params={"format": "csv"})
        json_rows = json_export.json()["documents"]
        csv_rows = list(csv.DictReader(io.StringIO(csv_export.content.decode("utf-8-sig"))))
        require(len(json_rows) == len(csv_rows) == 5, "exports do not include all five rows")
        require("text/csv" in csv_export.headers["content-type"], "CSV export has wrong media type")
        require("application/json" in json_export.headers["content-type"], "JSON export has wrong media type")
        for index, (_, text) in enumerate(EXAMPLES):
            require(json_rows[index]["original_text"] == csv_rows[index]["original_text"] == text,
                    "CSV/JSON export corrupted Marathi text")
        for response, extension in ((json_export, "json"), (csv_export, "csv")):
            require(response.headers["content-disposition"].endswith(f'{session_id}.{extension}"'),
                    "export filename is incorrect")
            (output / f"integration-export.{extension}").write_bytes(response.content)

        openapi = call("GET", "/openapi.json").json()
        expected_paths = {"/health", "/ready", "/v1/model-info", "/v1/analyze", "/v1/analyze/batch",
                          "/v1/analyses/{session_id}", "/v1/analyses/{session_id}/analytics",
                          "/v1/analyses/{session_id}/export"}
        require(expected_paths.issubset(openapi["paths"]), "OpenAPI is missing an integration endpoint")
        require(openapi["paths"]["/v1/analyze"]["post"]["responses"]["200"]["content"]["application/json"]
                ["schema"]["$ref"].endswith("/AnalysisResponse"), "OpenAPI analysis schema changed")
        security = {}
        for origin, expected in ((settings.allowed_origins[0], 200), ("https://untrusted.invalid", 400)):
            response = call("OPTIONS", "/v1/analyze", expected=expected,
                            headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
            require((response.headers.get("access-control-allow-origin") == origin) is (expected == 200),
                    "CORS allowed an unexpected origin")
            security[origin] = response.status_code
        for payload in ({"text": " "}, {"text": EXAMPLES[0][1], "unexpected": True}):
            response = call("POST", "/v1/analyze", expected=422, json=payload)
            require("Traceback" not in response.text, "validation error leaked traceback")
        call("GET", "/v1/analyses/does-not-exist", expected=404)
        call("GET", f"/v1/analyses/{session_id}", expected=422, params={"limit": 1001})
        call("GET", f"/v1/analyses/{session_id}/export", expected=422, params={"format": "xml"})
        response = call("POST", "/v1/analyze", expected=413, content=b"{" + b"a" * 23000,
                        headers={"Content-Type": "application/json", "X-Request-ID": "oversized-real-validation"})
        require(response.json()["error"]["code"] == "request_too_large", "request body limit is missing")
        security["request_body_limit"] = response.status_code

    return {"passed": True, "transport": "FastAPI TestClient ASGI lifespan", "operations": operations,
            "keyword_selection": keyword_selection, "examples": examples, "batch": batch,
            "pagination": {"pages": 3, "documents": len(documents), "documents_unique": True},
            "stored_documents": documents, "analytics": analytics,
            "exports": {"csv_rows": len(csv_rows), "json_documents": len(json_rows),
                        "csv_sha256": sha256(output / "integration-export.csv"),
                        "json_sha256": sha256(output / "integration-export.json")},
            "openapi_paths_verified": sorted(expected_paths), "security": security,
            "endpoint_latency_ms": latencies}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--keywords", choices=("auto", "keybert", "disabled"), default="auto")
    parser.add_argument("--expect-promoted", action="store_true")
    args = parser.parse_args(argv)
    artifact = args.artifact_dir.resolve()
    phase = "promoted" if args.expect_promoted else "prepromotion"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or artifact.parent.parent / "validation" / artifact.name / phase / timestamp
    output = output.resolve()
    require(output != artifact and artifact not in output.parents,
            "validation evidence must be written outside the immutable model directory")
    output.mkdir(parents=True, exist_ok=True)
    # Set before importing ML libraries; both independent and service loads also
    # pass local_files_only=True explicitly.
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[variable] = "1"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    started = perf_counter()
    memory = MemorySampler()
    memory.start()
    report: dict[str, Any] = {"schema_version": 1, "artifact_dir": str(artifact), "phase": phase,
                              "timestamp_utc": timestamp, "offline": True,
                              "integration_examples_are_metrics": False, "status": "running"}
    stage = "artifact_integrity"
    try:
        report["artifact"] = verify_artifact(artifact, args.expect_promoted)
        stage = "independent_reload_and_inference"
        report["independent_reload"] = independently_reload(artifact, args.device)
        stage = "sqlite_migration"
        database_path = output / "integration.db"
        report["database"] = {"engine": "SQLite", "path": str(database_path),
                              "alembic_revision": migrate_database(database_path)}
        stage = "real_api_integration"
        from backend.app.config import Settings

        settings = Settings(_env_file=None, app_env="validation", sentiment_backend="muril",
                            sentiment_model_path=str(artifact), allow_smoke_model=False,
                            model_device=report["independent_reload"]["device"],
                            keyword_backend="disabled" if args.keywords == "disabled" else "keybert",
                            topic_backend="disabled", summary_backend="extractive",
                            database_url="sqlite:///" + database_path.as_posix(), store_raw_text=True,
                            persist_single_analysis=False, allowed_origins=["http://localhost:5500"],
                            max_text_length=1024, max_upload_bytes=100000, max_batch_rows=10)
        report["application"] = validate_application(settings, output, report["independent_reload"],
                                                      args.expect_promoted, args.keywords)
        stage = "artifact_immutability"
        from ml.train import _artifact_integrity

        require(sha256(artifact / "model_manifest.json") == report["artifact"]["manifest_sha256"],
                "validation changed the model manifest")
        require(_artifact_integrity(artifact) == report["artifact"]["artifact_integrity"],
                "validation changed artifact files")
        report["artifact"]["unchanged_by_validation"] = True
        report["status"] = "passed"
    except Exception as exc:  # retain evidence even when an integration gate fails
        report["status"] = "failed"
        report["failure"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
    finally:
        report["resources"] = memory.finish()
        report["duration_seconds"] = round(perf_counter() - started, 3)
        report_path = output / "validation_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "report": str(report_path),
                          "duration_seconds": report["duration_seconds"], "failure": report.get("failure")},
                         ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
