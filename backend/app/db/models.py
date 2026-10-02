"""Normalized SQLAlchemy 2.x persistence models for analysis results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def uuid_string() -> str:
    return str(uuid4())


class Base(DeclarativeBase):
    pass


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    source_type: Mapped[str] = mapped_column(String(16), nullable=False)
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="processing")
    total_documents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    successful_documents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_documents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    model_version: Mapped[str] = mapped_column(String(255), nullable=False)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    documents: Mapped[list["AnalysisDocument"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_analysis_sessions_created_at", "created_at"),)


class AnalysisDocument(Base):
    __tablename__ = "analysis_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    original_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    analysis_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    primary_language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    devanagari_ratio: Mapped[float | None] = mapped_column(nullable=True)
    latin_ratio: Mapped[float | None] = mapped_column(nullable=True)
    is_code_mixed: Mapped[bool | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    processing_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(255), nullable=True)
    warnings: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    session: Mapped[AnalysisSession] = relationship(back_populates="documents")
    sentiment: Mapped["SentimentResultModel | None"] = relationship(
        back_populates="document", cascade="all, delete-orphan", uselist=False
    )
    keywords: Mapped[list["KeywordResultModel"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="KeywordResultModel.rank"
    )
    topic: Mapped["TopicResultModel | None"] = relationship(
        back_populates="document", cascade="all, delete-orphan", uselist=False
    )
    summary: Mapped["SummaryResultModel | None"] = relationship(
        back_populates="document", cascade="all, delete-orphan", uselist=False
    )

    __table_args__ = (
        Index("ix_analysis_documents_session_row", "session_id", "row_index"),
        Index("ix_analysis_documents_created_at", "created_at"),
    )


class SentimentResultModel(Base):
    __tablename__ = "sentiment_results"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_documents.id", ondelete="CASCADE"), primary_key=True
    )
    label: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(nullable=False)
    positive_probability: Mapped[float] = mapped_column(nullable=False)
    negative_probability: Mapped[float] = mapped_column(nullable=False)
    neutral_probability: Mapped[float] = mapped_column(nullable=False)
    low_confidence: Mapped[bool] = mapped_column(nullable=False)

    document: Mapped[AnalysisDocument] = relationship(back_populates="sentiment")


class KeywordResultModel(Base):
    __tablename__ = "keyword_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    keyword: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float] = mapped_column(nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)

    document: Mapped[AnalysisDocument] = relationship(back_populates="keywords")


class TopicResultModel(Base):
    __tablename__ = "topic_results"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_documents.id", ondelete="CASCADE"), primary_key=True
    )
    topic_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    topic_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    probability: Mapped[float | None] = mapped_column(nullable=True)
    topic_model_version: Mapped[str | None] = mapped_column(String(255), nullable=True)

    document: Mapped[AnalysisDocument] = relationship(back_populates="topic")


class SummaryResultModel(Base):
    __tablename__ = "summary_results"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_documents.id", ondelete="CASCADE"), primary_key=True
    )
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider: Mapped[str | None] = mapped_column(String(64), nullable=True)

    document: Mapped[AnalysisDocument] = relationship(back_populates="summary")
