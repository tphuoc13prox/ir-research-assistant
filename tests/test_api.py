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


def test_summarize_requires_topic_session() -> None:
    client = TestClient(app)

    response = client.post("/paper/summarize", json={"paper_id": "1234.5678"})

    assert response.status_code == 409
    assert "choose a research topic" in response.json()["detail"]


def test_update_settings_threshold() -> None:
    client = TestClient(app)
    response = client.post("/settings", json={
        "hybrid_enabled": True,
        "dense_top_k": 5,
        "sparse_top_k": 5,
        "fusion_top_k": 5,
        "rrf_k": 60,
        "dense_weight": 1.0,
        "sparse_weight": 1.0,
        "ranking_enabled": True,
        "base_model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "relevance_threshold": 0.45
    })
    assert response.status_code == 200
    assert response.json() == {"status": "updated"}
    
    from src.api.routes import session_manager
    assert session_manager.settings["relevance_threshold"] == 0.45
