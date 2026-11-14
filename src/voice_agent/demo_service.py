"""A small order-status service that stands in for a client's backend system, run locally.

It is a stand-in, not a real business system: seeded JSON data, an API-key header, and fault
injection so the agent's REST client can be tested against slow, failing and malformed responses.

    GET  /orders/{order_id}                 status, item, expected delivery date (404 if unknown)
    GET  /orders?phone=9876543210           orders for a phone number
    POST /orders/{order_id}/reschedule      {"date": "YYYY-MM-DD"} with an Idempotency-Key header
                                            200, 404, 409 (already delivered / cancelled), 422 (bad or past date);
                                            repeating a key returns the first response unchanged
    POST /_admin/reset                      restore the seed data, clear idempotency keys and faults
    POST /_admin/faults                     set faults (below)

Every route needs ``X-API-Key``. Faults apply to the /orders routes and count down per request:
``latency_ms`` (every request), ``fail_503`` (next n requests), ``rate_limit_429`` (next n,
with ``Retry-After: retry_after_s``), ``hang_s`` (every request sleeps this long, longer than the
client's timeout), ``malformed`` (next n bodies are not JSON).

    voice-agent demo-service --port 8090
"""

from __future__ import annotations

import asyncio
import copy
import json
import socket
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

DATA_PATH = Path(__file__).parent / "data" / "orders.json"
DEFAULT_KEY = "demo-key"
TZ = ZoneInfo("Asia/Kolkata")
FAULT_KEYS = ("latency_ms", "fail_503", "rate_limit_429", "retry_after_s", "hang_s", "malformed")


def today() -> date:
    return datetime.now(TZ).date()


class OrderStore:
    def __init__(self, path: Path = DATA_PATH):
        self.seed = json.loads(Path(path).read_text(encoding="utf-8"))
        self.reset()

    def reset(self) -> None:
        base = today()
        self.orders: dict[str, dict] = {}
        for o in copy.deepcopy(self.seed):
            o["expected_delivery"] = (base + timedelta(days=o.pop("delivery_offset_days"))).isoformat()
            self.orders[o["order_id"]] = o
        self.idempotency: dict[str, tuple[int, dict]] = {}
        self.faults: dict[str, float] = {k: 0 for k in FAULT_KEYS}
        self.writes = 0

    def public(self, o: dict) -> dict:
        return {k: o[k] for k in ("order_id", "status", "item", "expected_delivery", "customer")}


def create_app(store: OrderStore | None = None, api_key: str = DEFAULT_KEY, faults: dict | None = None) -> FastAPI:
    store = store or OrderStore()
    if faults:
        store.faults.update(faults)
    app = FastAPI(title="order service (local stand-in)")
    app.state.store = store

    def auth(key: str | None) -> Response | None:
        if key != api_key:
            return JSONResponse({"detail": "missing or wrong API key"}, status_code=401)
        return None

    async def fault() -> Response | None:
        f = store.faults
        if f["latency_ms"]:
            await asyncio.sleep(f["latency_ms"] / 1000)
        if f["hang_s"]:
            await asyncio.sleep(f["hang_s"])
        if f["fail_503"] > 0:
            f["fail_503"] -= 1
            return JSONResponse({"detail": "service unavailable"}, status_code=503)
        if f["rate_limit_429"] > 0:
            f["rate_limit_429"] -= 1
            return JSONResponse({"detail": "too many requests"}, status_code=429,
                                headers={"Retry-After": str(int(f["retry_after_s"] or 1))})
        if f["malformed"] > 0:
            f["malformed"] -= 1
            return PlainTextResponse('{"order_id": "4821', media_type="application/json")
        return None

    @app.post("/_admin/reset")
    def reset(x_api_key: str | None = Header(None)):
        if (r := auth(x_api_key)) is not None:
            return r
        store.reset()
        return {"ok": True}

    @app.post("/_admin/faults")
    async def set_faults(request: Request, x_api_key: str | None = Header(None)):
        if (r := auth(x_api_key)) is not None:
            return r
        body = await request.json()
        store.faults.update({k: v for k, v in body.items() if k in FAULT_KEYS})
        return store.faults

    @app.get("/orders/{order_id}")
    async def get_order(order_id: str, x_api_key: str | None = Header(None)):
        if (r := auth(x_api_key)) is not None or (r := await fault()) is not None:
            return r
        o = store.orders.get(order_id)
        if o is None:
            return JSONResponse({"detail": f"order {order_id} not found"}, status_code=404)
        return store.public(o)

    @app.get("/orders")
    async def find_orders(phone: str = "", x_api_key: str | None = Header(None)):
        if (r := auth(x_api_key)) is not None or (r := await fault()) is not None:
            return r
        return {"phone": phone, "orders": [store.public(o) for o in store.orders.values() if o["phone"] == phone]}

    @app.post("/orders/{order_id}/reschedule")
    async def reschedule(order_id: str, request: Request, x_api_key: str | None = Header(None),
                         idempotency_key: str | None = Header(None)):
        if (r := auth(x_api_key)) is not None or (r := await fault()) is not None:
            return r
        if not idempotency_key:
            return JSONResponse({"detail": "Idempotency-Key header required"}, status_code=400)
        if idempotency_key in store.idempotency:
            code, body = store.idempotency[idempotency_key]
            return JSONResponse(body, status_code=code)
        try:
            payload = await request.json()
            new = date.fromisoformat(str(payload.get("date", "")))
        except (ValueError, AttributeError):
            code, body = 422, {"detail": "date must be YYYY-MM-DD"}
        else:
            o = store.orders.get(order_id)
            if o is None:
                code, body = 404, {"detail": f"order {order_id} not found"}
            elif o["status"] in ("delivered", "cancelled"):
                code, body = 409, {"detail": f"order {order_id} is already {o['status']}", "status": o["status"]}
            elif new <= today():
                code, body = 422, {"detail": "the new date must be after today"}
            else:
                o["expected_delivery"] = new.isoformat()
                store.writes += 1
                code, body = 200, {**store.public(o), "rescheduled": True}
        store.idempotency[idempotency_key] = (code, body)
        return JSONResponse(body, status_code=code)

    return app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServiceThread:
    """Run the service with uvicorn in a background thread (for evaluations and the call suite)."""

    def __init__(self, store: OrderStore | None = None, api_key: str = DEFAULT_KEY, port: int | None = None):
        import uvicorn

        self.store = store or OrderStore()
        self.api_key = api_key
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        config = uvicorn.Config(create_app(self.store, api_key), host="127.0.0.1", port=self.port,
                                log_level="warning", lifespan="off")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, name="order-service", daemon=True)

    def __enter__(self) -> "ServiceThread":
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("order service did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


def serve(port: int = 8090, api_key: str = DEFAULT_KEY) -> None:
    import uvicorn

    uvicorn.run(create_app(api_key=api_key), host="127.0.0.1", port=port, log_level="info")
