"""REST client: retries, back-off, Retry-After, timeouts, idempotency and error mapping. No network."""

import httpx

from voice_agent.orders_client import OrdersClient


def client_for(handler, **kw):
    sleeps = []
    c = OrdersClient(base_url="http://svc", api_key="k", transport=httpx.MockTransport(handler),
                     sleep=sleeps.append, rng=lambda: 0.5, **kw)
    return c, sleeps


def test_success_records_meta_and_sends_key():
    seen = []

    def h(req):
        seen.append(req.headers["x-api-key"])
        return httpx.Response(200, json={"order_id": "48213", "status": "shipped"})

    c, sleeps = client_for(h)
    r = c.get_order("48213")
    assert r["status"] == "shipped" and r["_meta"]["attempts"] == 1 and r["_meta"]["statuses"] == [200]
    assert seen == ["k"] and sleeps == []


def test_retries_503_then_succeeds_with_exponential_backoff():
    codes = [503, 503, 200]

    def h(req):
        code = codes.pop(0)
        return httpx.Response(code, json={"order_id": "1"} if code == 200 else {"detail": "down"})

    c, sleeps = client_for(h)
    r = c.get_order("1")
    assert r["order_id"] == "1" and r["_meta"]["statuses"] == [503, 503, 200]
    assert sleeps == [0.2, 0.4]  # backoff_s * 2**n * (0.5 + rng)


def test_gives_up_after_two_retries():
    c, sleeps = client_for(lambda req: httpx.Response(503, json={"detail": "down"}))
    r = c.get_order("1")
    assert r["error"] == "unavailable" and r["attempts"] == 3 and len(sleeps) == 2


def test_429_honours_retry_after():
    codes = [429, 200]

    def h(req):
        code = codes.pop(0)
        return httpx.Response(code, headers={"Retry-After": "1"}, json={"order_id": "1"})

    c, sleeps = client_for(h)
    assert c.get_order("1")["order_id"] == "1" and sleeps == [1.0]


def test_retry_after_longer_than_budget_gives_up_immediately():
    c, sleeps = client_for(lambda req: httpx.Response(429, headers={"Retry-After": "30"}, json={}))
    r = c.get_order("1")
    assert r["error"] == "unavailable" and r["attempts"] == 1 and sleeps == []


def test_timeout_maps_to_timeout_error():
    def h(req):
        raise httpx.ReadTimeout("slow", request=req)

    c, _ = client_for(h)
    r = c.get_order("1")
    assert r["error"] == "timeout" and r["attempts"] == 3 and r["_meta"]["statuses"] == ["timeout"] * 3


def test_connect_error_maps_to_unavailable():
    def h(req):
        raise httpx.ConnectError("refused", request=req)

    c, _ = client_for(h, max_retries=0)
    assert c.get_order("1")["error"] == "unavailable"


def test_malformed_json_is_retried():
    bodies = [b'{"order_id": "4', b'{"order_id": "4"}']
    c, _ = client_for(lambda req: httpx.Response(200, content=bodies.pop(0)))
    assert c.get_order("4")["order_id"] == "4"


def test_client_errors_map_without_retry():
    for code, kind in ((404, "not_found"), (409, "rejected"), (422, "rejected"), (401, "unavailable")):
        c, sleeps = client_for(lambda req, code=code: httpx.Response(code, json={"detail": "x"}))
        r = c.get_order("1")
        assert r["error"] == kind and r["attempts"] == 1 and r["status"] == code and sleeps == []


def test_reschedule_reuses_one_idempotency_key_across_retries():
    keys, codes = [], [503, 200]

    def h(req):
        keys.append(req.headers["idempotency-key"])
        return httpx.Response(codes.pop(0), json={"order_id": "1", "rescheduled": True})

    c, _ = client_for(h)
    assert c.reschedule("1", "2030-01-01")["rescheduled"]
    assert len(keys) == 2 and keys[0] == keys[1]


def test_budget_caps_total_time():
    import time

    def h(req):
        time.sleep(0.15)
        raise httpx.ReadTimeout("slow", request=req)

    c = OrdersClient(base_url="http://svc", transport=httpx.MockTransport(h), budget_s=0.3, backoff_s=0.01)
    t = time.perf_counter()
    r = c.get_order("1")
    assert r["error"] == "timeout" and time.perf_counter() - t < 0.6
