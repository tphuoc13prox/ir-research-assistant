from fastapi.testclient import TestClient

from src.api.app import app


def test_root_serves_chatbot_ui() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "IR Research Assistant" in response.text


def test_session_status_reports_no_active_topic() -> None:
    client = TestClient(app)

    response = client.get("/session/status")

    assert response.status_code == 200
    assert response.json() == {"ready": False}


def test_chat_requires_topic_session() -> None:
    client = TestClient(app)

    response = client.post("/chat", json={"question": "What is dense retrieval?"})

    assert response.status_code == 409
    assert "choose a research topic" in response.json()["detail"]
