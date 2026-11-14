"""The local order service: auth, lookups, reschedule rules, idempotency, faults."""

from datetime import timedelta

from fastapi.testclient import TestClient

from voice_agent.demo_service import OrderStore, create_app, today

H = {"X-API-Key": "demo-key"}


def make():
    store = OrderStore()
    return TestClient(create_app(store)), store


def test_auth_required():
    c, _ = make()
    assert c.get("/orders/48213").status_code == 401


def test_lookup_and_phone_search():
    c, _ = make()
    r = c.get("/orders/48213", headers=H).json()
    assert r["status"] == "shipped" and r["item"] == "pressure cooker"
    assert c.get("/orders/11111", headers=H).status_code == 404
    ids = [o["order_id"] for o in c.get("/orders", params={"phone": "9876543210"}, headers=H).json()["orders"]]
    assert ids == ["48213", "51177"]


def test_reschedule_rules_and_idempotency():
    c, store = make()
    new = (today() + timedelta(days=7)).isoformat()
    hk = {**H, "Idempotency-Key": "a1"}
    r = c.post("/orders/48213/reschedule", json={"date": new}, headers=hk)
    assert r.status_code == 200 and r.json()["expected_delivery"] == new
    again = c.post("/orders/48213/reschedule", json={"date": "2000-01-01"}, headers=hk)
    assert again.status_code == 200 and again.json() == r.json() and store.writes == 1
    assert c.post("/orders/51177/reschedule", json={"date": new}, headers={**H, "Idempotency-Key": "b"}).status_code == 409
    past = today().isoformat()
    assert c.post("/orders/48213/reschedule", json={"date": past}, headers={**H, "Idempotency-Key": "c"}).status_code == 422
    assert c.post("/orders/48213/reschedule", json={"date": new}, headers=H).status_code == 400


def test_faults_count_down_and_reset():
    c, store = make()
    c.post("/_admin/faults", json={"fail_503": 1, "rate_limit_429": 1, "retry_after_s": 2, "malformed": 1}, headers=H)
    assert c.get("/orders/48213", headers=H).status_code == 503
    r = c.get("/orders/48213", headers=H)
    assert r.status_code == 429 and r.headers["retry-after"] == "2"
    assert c.get("/orders/48213", headers=H).text.startswith('{"order_id": "4821')
    assert c.get("/orders/48213", headers=H).json()["order_id"] == "48213"
    c.post("/_admin/reset", headers=H)
    assert store.faults["fail_503"] == 0
