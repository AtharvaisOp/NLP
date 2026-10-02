"""Repository methods for normalized analysis persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from ..db.models import (
    AnalysisDocument,
    AnalysisSession,
    KeywordResultModel,
    SentimentResultModel,
    SummaryResultModel,
    TopicResultModel,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisRepository:
    """SQLAlchemy-only data access; route handlers never issue SQL directly."""

    def create_session(
        self,
        db: Session,
        *,
        source_type: str,
        filename: str | None,
        model_version: str,
        metadata: dict[str, Any] | None,
    ) -> AnalysisSession:
        item = AnalysisSession(
            id=str(uuid4()),
            source_type=source_type,
            filename=filename,
            model_version=model_version,
            metadata_json=metadata,
        )
        db.add(item)
        db.flush()
        return item

    def add_success_document(
        self,
        db: Session,
        *,
        session_id: str,
        row_index: int,
        original_text: str | None,
        response,
        topic_model_version: str | None,
    ) -> AnalysisDocument:
        document = AnalysisDocument(
            id=str(uuid4()),
            session_id=session_id,
            row_index=row_index,
            original_text=original_text,
            model_text=response.model_text,
            analysis_text=response.analysis_text,
            primary_language=response.language.primary,
            devanagari_ratio=response.language.devanagari_ratio,
            latin_ratio=response.language.latin_ratio,
            is_code_mixed=response.language.is_code_mixed,
            processing_ms=response.meta.processing_ms,
            status="success",
            warnings=response.meta.warnings,
        )
        document.sentiment = SentimentResultModel(
            label=response.sentiment.label,
            confidence=response.sentiment.confidence,
            positive_probability=response.sentiment.probabilities["positive"],
            negative_probability=response.sentiment.probabilities["negative"],
            neutral_probability=response.sentiment.probabilities["neutral"],
            low_confidence=response.sentiment.low_confidence,
        )
        document.keywords = [
            KeywordResultModel(keyword=item.text, score=item.score, rank=rank)
            for rank, item in enumerate(response.keywords, start=1)
        ]
        document.topic = TopicResultModel(
            topic_id=response.topic.id,
            topic_label=response.topic.label,
            probability=response.topic.probability,
            topic_model_version=topic_model_version,
        )
        document.summary = SummaryResultModel(
            text=response.summary.text,
            provider=response.summary.provider,
        )
        db.add(document)
        return document

    def add_failed_document(
        self,
        db: Session,
        *,
        session_id: str,
        row_index: int,
        original_text: str | None,
        error_code: str,
        error_message: str,
    ) -> AnalysisDocument:
        document = AnalysisDocument(
            id=str(uuid4()),
            session_id=session_id,
            row_index=row_index,
            original_text=original_text,
            status="failed",
            error_code=error_code,
            error_message=error_message,
        )
        db.add(document)
        return document

    def complete_session(
        self,
        db: Session,
        *,
        session_id: str,
        total: int,
        successful: int,
        failed: int,
    ) -> AnalysisSession:
        item = db.get(AnalysisSession, session_id)
        if item is None:
            raise LookupError("analysis session was not found")
        item.total_documents = total
        item.successful_documents = successful
        item.failed_documents = failed
        item.status = "completed" if failed == 0 else "partial" if successful else "failed"
        item.completed_at = utcnow()
        return item

    def get_session(self, db: Session, session_id: str) -> AnalysisSession | None:
        return db.get(AnalysisSession, session_id)

    def count_documents(self, db: Session, session_id: str) -> int:
        return int(
            db.scalar(
                select(func.count(AnalysisDocument.id)).where(
                    AnalysisDocument.session_id == session_id
                )
            )
            or 0
        )

    def list_documents(
        self, db: Session, session_id: str, *, limit: int, offset: int
    ) -> list[AnalysisDocument]:
        statement = (
            select(AnalysisDocument)
            .where(AnalysisDocument.session_id == session_id)
            .options(
                selectinload(AnalysisDocument.sentiment),
                selectinload(AnalysisDocument.keywords),
                selectinload(AnalysisDocument.topic),
                selectinload(AnalysisDocument.summary),
            )
            .order_by(AnalysisDocument.row_index)
            .limit(limit)
            .offset(offset)
        )
        return list(db.scalars(statement).all())

    def all_successful_documents(self, db: Session, session_id: str) -> list[AnalysisDocument]:
        statement = (
            select(AnalysisDocument)
            .where(
                AnalysisDocument.session_id == session_id,
                AnalysisDocument.status == "success",
            )
            .options(
                selectinload(AnalysisDocument.sentiment),
                selectinload(AnalysisDocument.keywords),
                selectinload(AnalysisDocument.topic),
                selectinload(AnalysisDocument.summary),
            )
            .order_by(AnalysisDocument.row_index)
        )
        return list(db.scalars(statement).all())
