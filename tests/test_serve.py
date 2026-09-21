import json
import math
import socket
import threading
import time

import pytest

from any2jev.serve import create_app

DEPARTMENT = {"returns": "Exchanges, refunds, wrong or damaged items", "shipping": "Delivery status, delays, lost packages",
              "billing": "Charges, invoices, payment problems"}


@pytest.fixture(scope="module")
def client(tiny_model):
    from fastapi.testclient import TestClient

    return TestClient(create_app(tiny_model))


def test_choice_roundtrip_matches_docs(client):
    r = client.post("/v1/systemone", json={"state": "My running shoes arrived in the wrong size.", "model": "jev-latest",
                                           "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": DEPARTMENT}}})
    assert r.status_code == 200
    body = r.json()
    a = body["answers"]["department"]
    assert a["type"] == "choice" and a["choice"] in DEPARTMENT and set(a["probabilities"]) == set(DEPARTMENT)
    assert math.isclose(sum(a["probabilities"].values()), 1.0, abs_tol=0.01) and 0 <= a["confidence"] <= 1
    assert a["choice"] == max(a["probabilities"], key=a["probabilities"].get)
    assert set(body["usage"]) == {"input_tokens", "output_tokens"} and body["model"] == "any2jev-latest"


def test_noul_score_and_object_state(client):
    r = client.post("/v1/systemone", json={"state": {"document": "I was charged twice. Please fix this ASAP."}, "model": "jev-latest",
                                           "questions": {"billing": {"type": "noul", "instructions": "Is this ticket about billing?", "criteria": {"true": "About charges", "false": "Not about charges"}},
                                                         "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["can wait", "this week", "today"]}}})
    assert r.status_code == 200
    n, s = r.json()["answers"]["billing"], r.json()["answers"]["urgency"]
    assert n == {"type": "noul", "noul": n["noul"]} and 0 <= n["noul"] <= 1
    assert s["type"] == "score" and 0 <= s["score"] <= 2 and s["legend"] == {"0": "can wait", "1": "this week", "2": "today"}
    assert math.isclose(s["score"], sum(int(k) * v for k, v in s["probabilities"].items()), abs_tol=0.01)


@pytest.mark.parametrize("body", [
    {"state": "x", "model": "m", "questions": {"q": {"type": "score", "instructions": "i", "criteria": ["only one"]}}},
    {"state": "x", "model": "m", "questions": {"q": {"type": "bogus", "instructions": "i"}}},
    {"state": "x", "model": "m", "questions": {}},
    {"state": "x", "model": "m", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {f"o{i}": None for i in range(256)}}}},
])
def test_validation_422(client, body):
    assert client.post("/v1/systemone", json=body).status_code == 422


def test_branch_limit_returns_422_and_server_recovers(tiny_model):
    from fastapi.testclient import TestClient

    with TestClient(create_app(tiny_model), raise_server_exceptions=False) as client:
        before = client.get("/health").json()["requests"]
        body = {"state": "short", "questions": {"q": {"type": "noul", "instructions": "word " * 1500}}}
        response = client.post("/v1/systemone", json=body)
        assert response.status_code == 422
        assert "branch limit" in response.json()["detail"]
        assert client.get("/health").json()["requests"] == before
        body["questions"]["q"]["instructions"] = "Is this short?"
        assert client.post("/v1/systemone", json=body).status_code == 200


def test_packed_equals_separate(client):
    qs = {"a": {"type": "noul", "instructions": "Is the weather described as nice?"},
          "b": {"type": "choice", "instructions": "Which season is it most likely?", "criteria": {"summer": None, "winter": None, "unknown": None}}}
    state = "The weather is nice today and the park is full of people."
    both = client.post("/v1/systemone", json={"state": state, "model": "m", "questions": qs}).json()["answers"]
    alone = client.post("/v1/systemone", json={"state": state, "model": "m", "questions": {"b": qs["b"]}}).json()["answers"]
    for k in both["b"]["probabilities"]:
        assert abs(both["b"]["probabilities"][k] - alone["b"]["probabilities"][k]) <= 0.002


def test_models_and_health(client):
    m = client.get("/v1/models").json()["models"]
    assert {x["name"] for x in m} >= {"any2jev-latest", "jev-latest"} and all({"name", "description", "release_date"} <= set(x) for x in m)
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["requests"] >= 1


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_official_sdk_against_local_server(tiny_model):
    pytest.importorskip("typesafe_sdk")
    uvicorn = pytest.importorskip("uvicorn")
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(tiny_model), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    try:
        with TypeSafeClient(api_key="local", base_url=f"http://127.0.0.1:{port}", model="any2jev-latest") as c:
            resp = c.system_one(state={"document": "I was charged twice. Please fix this ASAP."},
                                questions={"billing": Noul(instructions="Is this ticket about billing?"),
                                           "tone": Choice(instructions="What is the customer's tone?", criteria={"calm": None, "frustrated": None, "angry": None}),
                                           "urgency": Score(instructions="How urgent is this ticket?", criteria=["can wait", "this week", "today"])})
            assert 0 <= resp.nouls["billing"].noul <= 1
            assert resp.choices["tone"].choice in {"calm", "frustrated", "angry"}
            assert 0 <= resp.scores["urgency"].score <= 2 and resp.scores["urgency"].legend[2] == "today"
            assert resp.usage.input_tokens and resp.usage.input_tokens > 0
            models = c.models.list()
            assert any(m.name == "any2jev-latest" for m in models.models)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    json.dumps(resp.answers["billing"].model_dump())
