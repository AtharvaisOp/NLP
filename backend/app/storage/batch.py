"""Bounded synchronous CSV batch processing over the shared orchestrator."""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from time import perf_counter

from ..config import Settings
from ..exceptions import APIError, InputValidationError
from ..logging_config import LOGGER_NAME
from ..services.orchestrator import AnalysisOrchestrator
from .persistence import PersistenceService


class BatchProcessor:
    def __init__(self, settings: Settings, persistence: PersistenceService) -> None:
        self.settings = settings
        self.persistence = persistence

    def process(
        self,
        payload: bytes,
        *,
        filename: str | None,
        text_column: str | None,
        orchestrator: AnalysisOrchestrator,
    ):
        rows = self._parse(payload, filename=filename, text_column=text_column)
        started = perf_counter()
        session_id = self.persistence.create_session(
            orchestrator,
            source_type="batch",
            filename=Path(filename).name if filename else None,
        )
        successful = 0
        failed = 0
        for row_index, text in rows:
            try:
                response = orchestrator.analyze(text, f"{session_id}-row-{row_index}")
                self.persistence.persist_success(
                    session_id,
                    row_index=row_index,
                    original_text=text,
                    response=response,
                    orchestrator=orchestrator,
                )
                successful += 1
            except APIError as exc:
                self.persistence.persist_failure(
                    session_id,
                    row_index=row_index,
                    original_text=text,
                    error_code=exc.code,
                    error_message=exc.message,
                )
                failed += 1
            except Exception as exc:  # noqa: BLE001 - isolate one row safely
                logging.getLogger(LOGGER_NAME).exception(
                    "batch row failed",
                    extra={
                        "event": "batch.row_failed",
                        "fields": {"session_id": session_id, "row_index": row_index},
                    },
                    exc_info=exc,
                )
                self.persistence.persist_failure(
                    session_id,
                    row_index=row_index,
                    original_text=text,
                    error_code="internal_error",
                    error_message="Row analysis failed",
                )
                failed += 1
        self.persistence.finish_session(
            session_id, total=len(rows), successful=successful, failed=failed
        )
        metadata = orchestrator.sentiment_service.metadata()
        return {
            "session_id": session_id,
            "status": "completed" if failed == 0 else "partial" if successful else "failed",
            "total_documents": len(rows),
            "successful_documents": successful,
            "failed_documents": failed,
            "processing_ms": max(0, round((perf_counter() - started) * 1000)),
            "model_version": metadata.version,
        }

    def _parse(
        self, payload: bytes, *, filename: str | None, text_column: str | None
    ) -> list[tuple[int, str]]:
        if not filename or Path(filename).suffix.casefold() != ".csv":
            raise InputValidationError("upload must be a CSV file")
        if len(payload) > self.settings.max_upload_bytes:
            raise InputValidationError(
                "upload exceeds the configured maximum size",
                details={"max_upload_bytes": self.settings.max_upload_bytes},
            )
        if not payload:
            raise InputValidationError("CSV upload is empty")
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InputValidationError("CSV upload must be valid UTF-8") from exc
        try:
            reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
            headers = reader.fieldnames
            if not headers:
                raise InputValidationError("CSV header is missing")
            headers = [header.strip() if header else "" for header in headers]
            selected = (text_column or self.settings.default_text_column).strip()
            if selected not in headers:
                raise InputValidationError(
                    "configured text column is missing",
                    details={"text_column": selected, "columns": headers},
                )
            rows = []
            for row_index, row in enumerate(reader):
                if row_index >= self.settings.max_batch_rows:
                    raise InputValidationError(
                        "CSV row count exceeds the configured maximum",
                        details={"max_batch_rows": self.settings.max_batch_rows},
                    )
                normalized = {
                    (key.strip() if key else ""): value for key, value in row.items()
                }
                rows.append((row_index, (normalized.get(selected) or "").strip()))
        except csv.Error as exc:
            raise InputValidationError("CSV content is malformed") from exc
        if not rows:
            raise InputValidationError("CSV contains no data rows")
        return rows
