"""Initial normalized analysis persistence schema."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0001_initial_analysis_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "analysis_sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_type", sa.String(length=16), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("total_documents", sa.Integer(), nullable=False),
        sa.Column("successful_documents", sa.Integer(), nullable=False),
        sa.Column("failed_documents", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(length=255), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analysis_sessions_created_at", "analysis_sessions", ["created_at"])
    op.create_table(
        "analysis_documents",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("row_index", sa.Integer(), nullable=False),
        sa.Column("original_text", sa.Text(), nullable=True),
        sa.Column("model_text", sa.Text(), nullable=True),
        sa.Column("analysis_text", sa.Text(), nullable=True),
        sa.Column("primary_language", sa.String(length=16), nullable=True),
        sa.Column("devanagari_ratio", sa.Float(), nullable=True),
        sa.Column("latin_ratio", sa.Float(), nullable=True),
        sa.Column("is_code_mixed", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("processing_ms", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.String(length=255), nullable=True),
        sa.Column("warnings", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["analysis_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analysis_documents_session_id", "analysis_documents", ["session_id"])
    op.create_index("ix_analysis_documents_session_row", "analysis_documents", ["session_id", "row_index"])
    op.create_index("ix_analysis_documents_created_at", "analysis_documents", ["created_at"])
    op.create_table(
        "sentiment_results",
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("label", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("positive_probability", sa.Float(), nullable=False),
        sa.Column("negative_probability", sa.Float(), nullable=False),
        sa.Column("neutral_probability", sa.Float(), nullable=False),
        sa.Column("low_confidence", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["analysis_documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("document_id"),
    )
    op.create_index("ix_sentiment_results_label", "sentiment_results", ["label"])
    op.create_table(
        "keyword_results",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("keyword", sa.Text(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["analysis_documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_keyword_results_document_id", "keyword_results", ["document_id"])
    op.create_table(
        "topic_results",
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("topic_id", sa.Integer(), nullable=True),
        sa.Column("topic_label", sa.String(length=255), nullable=True),
        sa.Column("probability", sa.Float(), nullable=True),
        sa.Column("topic_model_version", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["document_id"], ["analysis_documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("document_id"),
    )
    op.create_index("ix_topic_results_topic_id", "topic_results", ["topic_id"])
    op.create_table(
        "summary_results",
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("provider", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(["document_id"], ["analysis_documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("document_id"),
    )


def downgrade() -> None:
    op.drop_table("summary_results")
    op.drop_index("ix_topic_results_topic_id", table_name="topic_results")
    op.drop_table("topic_results")
    op.drop_index("ix_keyword_results_document_id", table_name="keyword_results")
    op.drop_table("keyword_results")
    op.drop_index("ix_sentiment_results_label", table_name="sentiment_results")
    op.drop_table("sentiment_results")
    op.drop_index("ix_analysis_documents_created_at", table_name="analysis_documents")
    op.drop_index("ix_analysis_documents_session_row", table_name="analysis_documents")
    op.drop_index("ix_analysis_documents_session_id", table_name="analysis_documents")
    op.drop_table("analysis_documents")
    op.drop_index("ix_analysis_sessions_created_at", table_name="analysis_sessions")
    op.drop_table("analysis_sessions")
