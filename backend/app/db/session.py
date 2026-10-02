"""Configured SQLAlchemy engine and request-scoped synchronous sessions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..config import Settings
from .models import Base


class DatabaseManager:
    def __init__(self, settings: Settings) -> None:
        self.url = settings.database_url
        self.configured = bool(self.url)
        self.engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        self._initialization_error: Exception | None = None
        if self.url:
            try:
                connect_args = {}
                engine_kwargs = {"pool_pre_ping": True, "future": True}
                if self.url.startswith("sqlite"):
                    connect_args["check_same_thread"] = False
                    if ":memory:" in self.url:
                        engine_kwargs["poolclass"] = StaticPool
                self.engine = create_engine(self.url, connect_args=connect_args, **engine_kwargs)
                self._session_factory = sessionmaker(
                    self.engine, class_=Session, expire_on_commit=False
                )
            except Exception as exc:  # noqa: BLE001 - readiness reports safe state
                self._initialization_error = exc

    @property
    def enabled(self) -> bool:
        return self.engine is not None

    @contextmanager
    def session(self) -> Iterator[Session]:
        if self._session_factory is None:
            raise RuntimeError("database is not configured")
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def readiness(self) -> tuple[str, str]:
        if not self.configured:
            return "disabled", "Database is not required; DATABASE_URL is not configured"
        if self.engine is None:
            return "unavailable", "Configured database could not be initialized"
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                available_tables = set(inspect(connection).get_table_names())
                if not set(Base.metadata.tables).issubset(available_tables):
                    return "unavailable", "Database schema is unavailable; apply Alembic migrations"
        except Exception:
            return "unavailable", "Configured database is unavailable"
        return "ready", "Configured database connection is available"

    def dispose(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
