"""Persistence orchestration kept separate from HTTP routes."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from ..config import Settings
from ..db.session import DatabaseManager
from ..exceptions import ResourceNotFound
from ..schemas import (
    AnalysisResponse,
    AnalysisSessionResponse,
    AnalyticsResponse,
    ConfidenceAnalytics,
    KeywordAggregate,
    KeywordInfo,
    LanguageAnalytics,
    LanguageInfo,
    PersistedDocumentResponse,
    SentimentAggregate,
    SentimentInfo,
    SessionInfo,
    SummaryInfo,
    TopicAggregate,
    TopicInfo,
)
from ..services.orchestrator import AnalysisOrchestrator
from .repository import AnalysisDocument
from .repository import AnalysisRepository


class StorageError(RuntimeError):
    """Raised when a configured database cannot persist or read results."""


class PersistenceService:
    def __init__(
        self,
        database: DatabaseManager,
        settings: Settings,
        repository: AnalysisRepository | None = None,
    ) -> None:
        self.database = database
        self.settings = settings
        self.repository = repository or AnalysisRepository()

    def require_database(self) -> None:
        if not self.database.enabled:
            raise StorageError("database persistence is not configured")

    def create_session(
        self,
        orchestrator: AnalysisOrchestrator,
        *,
        source_type: str,
        filename: str | None,
    ) -> str:
        self.require_database()
        metadata = self._service_metadata(orchestrator)
        try:
            with self.database.session() as db:
                item = self.repository.create_session(
                    db,
                    source_type=source_type,
                    filename=filename,
                    model_version=metadata["sentiment_model_version"],
                    metadata=metadata,
                )
                return item.id
        except Exception as exc:  # noqa: BLE001 - safe storage boundary
            raise StorageError("could not create analysis session") from exc

    def persist_success(
        self,
        session_id: str,
        *,
        row_index: int,
        original_text: str,
        response: AnalysisResponse,
        orchestrator: AnalysisOrchestrator,
    ) -> None:
        self.require_database()
        try:
            with self.database.session() as db:
                self.repository.add_success_document(
                    db,
                    session_id=session_id,
                    row_index=row_index,
                    original_text=original_text if self.settings.store_raw_text else None,
                    response=response,
                    topic_model_version=orchestrator.topic_service.metadata().version,
                )
        except Exception as exc:  # noqa: BLE001 - safe storage boundary
            raise StorageError("could not persist analysis result") from exc

    def persist_failure(
        self,
        session_id: str,
        *,
        row_index: int,
        original_text: str,
        error_code: str,
        error_message: str,
    ) -> None:
        self.require_database()
        try:
            with self.database.session() as db:
                self.repository.add_failed_document(
                    db,
                    session_id=session_id,
                    row_index=row_index,
                    original_text=original_text if self.settings.store_raw_text else None,
                    error_code=error_code,
                    error_message=error_message,
                )
        except Exception as exc:  # noqa: BLE001 - safe storage boundary
            raise StorageError("could not persist failed analysis result") from exc

    def finish_session(
        self, session_id: str, *, total: int, successful: int, failed: int
    ) -> None:
        self.require_database()
        try:
            with self.database.session() as db:
                self.repository.complete_session(
                    db,
                    session_id=session_id,
                    total=total,
                    successful=successful,
                    failed=failed,
                )
        except Exception as exc:  # noqa: BLE001 - safe storage boundary
            raise StorageError("could not finalize analysis session") from exc

    def _service_metadata(self, orchestrator: AnalysisOrchestrator) -> dict[str, Any]:
        sentiment = orchestrator.sentiment_service.metadata()
        keyword = orchestrator.keyword_service.metadata()
        topic = orchestrator.topic_service.metadata()
        summary = orchestrator.summary_service.metadata()
        return {
            "sentiment_model_version": sentiment.version,
            "sentiment_backend": sentiment.backend,
            "keyword_backend": keyword.backend,
            "keyword_model": keyword.embedding_model or keyword.version,
            "topic_backend": topic.backend,
            "topic_model_version": topic.version,
            "summary_backend": summary.backend,
            "summary_provider": summary.provider,
        }

    def get_session_response(
        self, session_id: str, *, limit: int, offset: int
    ) -> AnalysisSessionResponse:
        self.require_database()
        try:
            with self.database.session() as db:
                session = self.repository.get_session(db, session_id)
                if session is None:
                    raise ResourceNotFound("analysis session was not found")
                total = self.repository.count_documents(db, session_id)
                documents = self.repository.list_documents(
                    db, session_id, limit=limit, offset=offset
                )
                return AnalysisSessionResponse(
                    session=_session_info(session),
                    documents=[_document_response(item) for item in documents],
                    total_documents=total,
                    limit=limit,
                    offset=offset,
                )
        except ResourceNotFound:
            raise
        except Exception as exc:  # noqa: BLE001 - safe storage boundary
            raise StorageError("could not retrieve analysis session") from exc

    def analytics(self, session_id: str) -> AnalyticsResponse:
        self.require_database()
        try:
            with self.database.session() as db:
                session = self.repository.get_session(db, session_id)
                if session is None:
                    raise ResourceNotFound("analysis session was not found")
                documents = self.repository.all_successful_documents(db, session_id)
                return _analytics(session_id, session, documents)
        except ResourceNotFound:
            raise
        except Exception as exc:  # noqa: BLE001 - safe storage boundary
            raise StorageError("could not calculate analysis analytics") from exc

    def export(self, session_id: str, format_name: str) -> tuple[bytes, str, str]:
        self.require_database()
        try:
            with self.database.session() as db:
                session = self.repository.get_session(db, session_id)
                if session is None:
                    raise ResourceNotFound("analysis session was not found")
                documents = self.repository.list_documents(
                    db, session_id, limit=max(session.total_documents, 1), offset=0
                )
                rows = [_export_row(item) for item in documents]
                if format_name == "json":
                    content = json.dumps(
                        {
                            "session": _session_info(session).model_dump(mode="json"),
                            "documents": rows,
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                    return content, "application/json; charset=utf-8", f"mahapulse-{session_id}.json"
                if format_name == "csv":
                    return _csv_export(rows), "text/csv; charset=utf-8", f"mahapulse-{session_id}.csv"
                raise ValueError("unsupported export format")
        except (ResourceNotFound, ValueError):
            raise
        except Exception as exc:  # noqa: BLE001 - safe storage boundary
            raise StorageError("could not export analysis results") from exc


def _session_info(item) -> SessionInfo:
    return SessionInfo(
        id=item.id,
        source_type=item.source_type,
        filename=item.filename,
        created_at=item.created_at.isoformat(),
        completed_at=item.completed_at.isoformat() if item.completed_at else None,
        status=item.status,
        total_documents=item.total_documents,
        successful_documents=item.successful_documents,
        failed_documents=item.failed_documents,
        model_version=item.model_version,
    )


def _document_response(item: AnalysisDocument) -> PersistedDocumentResponse:
    language = None
    if item.primary_language is not None:
        language = LanguageInfo(
            primary=item.primary_language,
            devanagari_ratio=item.devanagari_ratio or 0,
            latin_ratio=item.latin_ratio or 0,
            is_code_mixed=bool(item.is_code_mixed),
        )
    sentiment = None
    if item.sentiment is not None:
        sentiment = SentimentInfo(
            label=item.sentiment.label,
            confidence=item.sentiment.confidence,
            probabilities={
                "positive": item.sentiment.positive_probability,
                "negative": item.sentiment.negative_probability,
                "neutral": item.sentiment.neutral_probability,
            },
            low_confidence=item.sentiment.low_confidence,
        )
    return PersistedDocumentResponse(
        id=item.id,
        row_index=item.row_index,
        status=item.status,
        original_text=item.original_text,
        model_text=item.model_text,
        analysis_text=item.analysis_text,
        language=language,
        sentiment=sentiment,
        keywords=[KeywordInfo(text=value.keyword, score=value.score) for value in item.keywords],
        topic=TopicInfo(
            id=item.topic.topic_id if item.topic else None,
            label=item.topic.topic_label if item.topic else None,
            probability=item.topic.probability if item.topic else None,
        ),
        summary=SummaryInfo(
            text=item.summary.text if item.summary else None,
            provider=item.summary.provider if item.summary else None,
        ),
        processing_ms=item.processing_ms,
        warnings=item.warnings or [],
        error_code=item.error_code,
        error_message=item.error_message,
    )


def _analytics(session_id: str, session, documents: list[AnalysisDocument]) -> AnalyticsResponse:
    successful = len(documents)
    denominator = successful or 1
    labels = {label: 0 for label in ("positive", "negative", "neutral")}
    confidences: list[float] = []
    low_confidence_count = 0
    code_mixed_count = 0
    keyword_scores: dict[str, list[float]] = {}
    topic_values: dict[tuple[int, str | None], int] = {}
    null_topic_count = 0
    for document in documents:
        if document.sentiment is not None:
            labels[document.sentiment.label] = labels.get(document.sentiment.label, 0) + 1
            confidences.append(document.sentiment.confidence)
            low_confidence_count += int(document.sentiment.low_confidence)
        code_mixed_count += int(bool(document.is_code_mixed))
        for keyword in document.keywords:
            keyword_scores.setdefault(keyword.keyword, []).append(keyword.score)
        if document.topic is None or document.topic.topic_id is None:
            null_topic_count += 1
        else:
            key = (document.topic.topic_id, document.topic.topic_label)
            topic_values[key] = topic_values.get(key, 0) + 1
    return AnalyticsResponse(
        session_id=session_id,
        total=session.total_documents,
        successful=session.successful_documents,
        failed=session.failed_documents,
        sentiment={
            label: SentimentAggregate(count=count, percentage=count * 100 / denominator)
            for label, count in labels.items()
        },
        confidence=ConfidenceAnalytics(
            average=sum(confidences) / len(confidences) if confidences else None,
            minimum=min(confidences) if confidences else None,
            maximum=max(confidences) if confidences else None,
            low_confidence_count=low_confidence_count,
        ),
        language=LanguageAnalytics(
            code_mixed_count=code_mixed_count,
            code_mixed_percentage=code_mixed_count * 100 / denominator,
        ),
        keywords=[
            KeywordAggregate(
                text=text,
                count=len(scores),
                average_score=sum(scores) / len(scores),
            )
            for text, scores in sorted(
                keyword_scores.items(), key=lambda pair: (-len(pair[1]), pair[0])
            )[:50]
        ],
        topics=[
            TopicAggregate(
                id=topic_id,
                label=label,
                count=count,
                percentage=count * 100 / denominator,
            )
            for (topic_id, label), count in sorted(
                topic_values.items(), key=lambda pair: (-pair[1], pair[0][0])
            )
        ],
        null_topic_count=null_topic_count,
        summary=None,
    )


def _export_row(item: AnalysisDocument) -> dict[str, Any]:
    return {
        "document_id": item.id,
        "session_id": item.session_id,
        "row_index": item.row_index,
        "status": item.status,
        "original_text": item.original_text,
        "model_text": item.model_text,
        "analysis_text": item.analysis_text,
        "sentiment": item.sentiment.label if item.sentiment else None,
        "confidence": item.sentiment.confidence if item.sentiment else None,
        "positive_probability": item.sentiment.positive_probability if item.sentiment else None,
        "negative_probability": item.sentiment.negative_probability if item.sentiment else None,
        "neutral_probability": item.sentiment.neutral_probability if item.sentiment else None,
        "low_confidence": item.sentiment.low_confidence if item.sentiment else None,
        "keywords": "; ".join(
            f"{keyword.keyword} ({keyword.score:.6f})" for keyword in item.keywords
        ),
        "topic_id": item.topic.topic_id if item.topic else None,
        "topic_label": item.topic.topic_label if item.topic else None,
        "topic_probability": item.topic.probability if item.topic else None,
        "summary": item.summary.text if item.summary else None,
        "language": item.primary_language,
        "is_code_mixed": item.is_code_mixed,
        "processing_ms": item.processing_ms,
        "error_code": item.error_code,
        "error_message": item.error_message,
    }


def _csv_export(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    fieldnames = list(rows[0].keys()) if rows else ["document_id", "status"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _safe_csv_cell(value) for key, value in row.items()})
    return output.getvalue().encode("utf-8-sig")


def _safe_csv_cell(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in {"=", "+", "-", "@"}:
        return "'" + value
    return value
