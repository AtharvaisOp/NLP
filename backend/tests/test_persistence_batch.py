from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.config import Settings
from backend.app.db.models import AnalysisSession
from backend.app.main import create_app


def make_client(tmp_path: Path, **overrides) -> TestClient:
    values = {
        "database_url": f"sqlite:///{tmp_path / 'mahapulse-test.db'}",
        "allowed_origins": [],
        "keyword_backend": "mock",
        "topic_backend": "mock",
        "summary_backend": "mock",
    }
    values.update(overrides)
    app = create_app(settings=Settings(**values))
    from backend.app.db.models import Base

    Base.metadata.create_all(app.state.database.engine)
    return TestClient(app)


def upload(client: TestClient, content: str, **params):
    return client.post(
        "/v1/analyze/batch",
        params=params,
        files={"file": ("reviews.csv", content.encode("utf-8"), "text/csv")},
    )


def test_batch_marathi_code_mixed_persistence_analytics_and_exports(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    fixture = Path(__file__).parent / "fixtures" / "phase6_marathi.csv"
    response = upload(client, fixture.read_text(encoding="utf-8"))

    assert response.status_code == 200
    batch = response.json()
    assert batch["status"] == "completed"
    assert batch["total_documents"] == 4
    assert batch["successful_documents"] == 4
    session_id = batch["session_id"]

    retrieved = client.get(f"/v1/analyses/{session_id}", params={"limit": 2})
    assert retrieved.status_code == 200
    body = retrieved.json()
    assert body["total_documents"] == 4
    assert len(body["documents"]) == 2
    assert body["documents"][0]["original_text"].startswith("हा मोबाईल")
    assert body["documents"][0]["language"]["primary"] == "mr"

    second_page = client.get(f"/v1/analyses/{session_id}", params={"limit": 2, "offset": 2})
    assert len(second_page.json()["documents"]) == 2
    assert second_page.json()["documents"][1]["language"]["is_code_mixed"] is True

    analytics = client.get(f"/v1/analyses/{session_id}/analytics")
    assert analytics.status_code == 200
    analytics_body = analytics.json()
    assert analytics_body["total"] == 4
    assert analytics_body["successful"] == 4
    assert sum(item["count"] for item in analytics_body["sentiment"].values()) == 4
    assert sum(item["percentage"] for item in analytics_body["sentiment"].values()) == pytest.approx(100)
    assert analytics_body["language"]["code_mixed_count"] == 1
    assert analytics_body["null_topic_count"] == 4

    json_export = client.get(f"/v1/analyses/{session_id}/export?format=json")
    assert json_export.status_code == 200
    assert json_export.headers["content-disposition"].endswith(f'{session_id}.json"')
    assert len(json_export.json()["documents"]) == 4

    csv_export = client.get(f"/v1/analyses/{session_id}/export?format=csv")
    assert csv_export.status_code == 200
    assert "text" in csv_export.content.decode("utf-8-sig")
    assert "हा मोबाईल" in csv_export.content.decode("utf-8-sig")


def test_batch_row_failure_isolated_and_session_partial(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = upload(client, "text\nचांगली सेवा\n \nखराब सेवा\n")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial"
    assert body["successful_documents"] == 2
    assert body["failed_documents"] == 1
    session = client.get(f"/v1/analyses/{body['session_id']}").json()
    failed = [item for item in session["documents"] if item["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["error_code"] == "invalid_input"
    assert "Traceback" not in json.dumps(failed)


def test_bom_custom_text_column_and_formula_injection_protection(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    content = "\ufeffreview,category\n=SUM(A1:A2),test\nहा phone चांगला आहे,review\n"
    response = client.post(
        "/v1/analyze/batch?text_column=review",
        files={"file": ("reviews.csv", content.encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200
    session_id = response.json()["session_id"]
    exported = client.get(f"/v1/analyses/{session_id}/export?format=csv")
    text = exported.content.decode("utf-8-sig")
    assert "'=SUM(A1:A2)" in text
    assert "हा phone" in text


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", "CSV upload is empty"),
        ("value\nhello\n", "configured text column is missing"),
        ("text\n\"unterminated\n", "CSV content is malformed"),
    ],
)
def test_invalid_csv_inputs_are_safe(
    tmp_path: Path, content: str, expected: str
) -> None:
    client = make_client(tmp_path)
    response = upload(client, content)
    assert response.status_code == 422
    assert expected in response.json()["error"]["message"]


def test_upload_size_and_row_limits(tmp_path: Path) -> None:
    client = make_client(tmp_path, max_upload_bytes=10, max_batch_rows=2)
    too_large = upload(client, "text\nहा मोबाईल\n")
    assert too_large.status_code == 422
    assert too_large.json()["error"]["code"] == "invalid_input"

    row_db_dir = tmp_path / "rows"
    row_db_dir.mkdir()
    client = make_client(row_db_dir, max_batch_rows=2)
    too_many = upload(client, "text\na\nb\nc\n")
    assert too_many.status_code == 422
    assert "row count" in too_many.json()["error"]["message"]


def test_store_raw_text_false_and_empty_analytics(tmp_path: Path) -> None:
    client = make_client(tmp_path, store_raw_text=False)
    response = upload(client, "text\n \n")
    assert response.status_code == 200
    session_id = response.json()["session_id"]
    retrieved = client.get(f"/v1/analyses/{session_id}").json()
    assert retrieved["documents"][0]["original_text"] is None
    assert retrieved["documents"][0]["status"] == "failed"
    analytics = client.get(f"/v1/analyses/{session_id}/analytics").json()
    assert analytics["successful"] == 0
    assert analytics["confidence"]["average"] is None
    assert analytics["language"]["code_mixed_percentage"] == 0
    assert analytics["keywords"] == []


def test_analyze_is_stateless_by_default_but_persists_when_enabled(tmp_path: Path) -> None:
    stateless = TestClient(create_app(settings=Settings(allowed_origins=[])))
    response = stateless.post("/v1/analyze", json={"text": "नमस्कार"})
    assert response.status_code == 200

    enabled = make_client(tmp_path, persist_single_analysis=True)
    response = enabled.post("/v1/analyze", json={"text": "नमस्कार"})
    assert response.status_code == 200
    with enabled.app.state.database.session() as db:
        session = db.scalar(select(AnalysisSession).where(AnalysisSession.source_type == "single"))
    assert session is not None
    retrieved = enabled.get(f"/v1/analyses/{session.id}")
    assert retrieved.status_code == 200
    assert retrieved.json()["session"]["total_documents"] == 1


def test_database_required_for_batch_and_history(tmp_path: Path) -> None:
    client = TestClient(create_app(settings=Settings(allowed_origins=[])))
    batch = upload(client, "text\nनमस्कार\n")
    assert batch.status_code == 503
    assert batch.json()["error"]["code"] == "service_unavailable"
    history = client.get("/v1/analyses/not-configured")
    assert history.status_code == 503

    configured_dir = tmp_path / "configured"
    configured_dir.mkdir()
    configured = make_client(configured_dir)
    missing = configured.get("/v1/analyses/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"


def test_ready_reports_database_not_required_when_unconfigured() -> None:
    client = TestClient(create_app(settings=Settings(allowed_origins=[])))
    body = client.get("/ready").json()
    assert body["services"]["database"]["state"] == "disabled"
