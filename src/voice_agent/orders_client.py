"""REST client for the order service, built for a voice turn that must never hang.

* Timeouts: connect 1 s, read 2.5 s, and a total budget of 5 s per tool call, retries included.
* Retries: up to 2, exponential back-off with jitter, on connect errors, timeouts, 502/503/504,
  malformed JSON and 429 (``Retry-After`` honoured if it fits in the budget). GETs are safe to
  retry; the reschedule POST is retried only because it carries an ``Idempotency-Key`` that the
  server deduplicates, so a retried write cannot apply twice.
* Errors never raise into the pipeline. They come back as data the LLM can speak about:
  ``{"error": "not_found" | "timeout" | "unavailable" | "rejected", "message": ..., "attempts": n}``.
* Every call records attempts, status codes and seconds in ``_meta`` (stripped before the LLM
  sees the result and attached to the ``ToolEvent``).

Base URL and key: ``VOICE_AGENT_ORDERS_URL`` and ``VOICE_AGENT_ORDERS_KEY``.
"""

from __future__ import annotations

import os
import random
import time
import uuid
from typing import Any, Callable

import httpx

RETRY_STATUS = (502, 503, 504)


class OrdersClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        connect_timeout: float = 1.0,
        read_timeout: float = 2.5,
        budget_s: float = 5.0,
        max_retries: int = 2,
        backoff_s: float = 0.2,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ):
        self.base_url = (base_url or os.environ.get("VOICE_AGENT_ORDERS_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("VOICE_AGENT_ORDERS_KEY") or "demo-key"
        self.connect_timeout, self.read_timeout = connect_timeout, read_timeout
        self.budget_s, self.max_retries, self.backoff_s = budget_s, max_retries, backoff_s
        self.sleep, self.rng = sleep, rng
        self.http = httpx.Client(transport=transport, headers={"X-API-Key": self.api_key})

    # ------------------------------------------------------------ public calls

    def get_order(self, order_id: str) -> dict[str, Any]:
        return self._request("GET", f"/orders/{order_id}")

    def find_by_phone(self, phone: str) -> dict[str, Any]:
        return self._request("GET", "/orders", params={"phone": phone})

    def reschedule(self, order_id: str, new_date: str, idempotency_key: str | None = None) -> dict[str, Any]:
        key = idempotency_key or str(uuid.uuid4())  # one key for every retry of this call
        return self._request("POST", f"/orders/{order_id}/reschedule", json={"date": new_date},
                             headers={"Idempotency-Key": key})

    # ------------------------------------------------------------ core

    def _request(self, method: str, path: str, **kw) -> dict[str, Any]:
        t0 = time.perf_counter()
        deadline = t0 + self.budget_s
        statuses: list[int | str] = []
        attempts = 0
        last_kind, last_msg = "unavailable", "no attempt made"
        while attempts <= self.max_retries:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.05:
                last_kind, last_msg = "timeout", "time budget used up"
                break
            attempts += 1
            timeout = httpx.Timeout(min(self.read_timeout, remaining), connect=min(self.connect_timeout, remaining))
            retry_after: float | None = None
            try:
                resp = self.http.request(method, self.base_url + path, timeout=timeout, **kw)
            except httpx.TimeoutException as e:
                statuses.append("timeout")
                last_kind, last_msg = "timeout", f"no response in time ({type(e).__name__})"
            except httpx.HTTPError as e:
                statuses.append("connect_error")
                last_kind, last_msg = "unavailable", f"could not reach the order system ({type(e).__name__})"
            else:
                statuses.append(resp.status_code)
                code = resp.status_code
                if code == 200:
                    try:
                        body = resp.json()
                    except ValueError:
                        last_kind, last_msg = "unavailable", "the order system sent a malformed response"
                    else:
                        return self._done(body, attempts, statuses, t0)
                elif code in RETRY_STATUS:
                    last_kind, last_msg = "unavailable", f"the order system is unavailable (HTTP {code})"
                elif code == 429:
                    last_kind, last_msg = "unavailable", "the order system is busy (HTTP 429)"
                    ra = resp.headers.get("retry-after", "")
                    retry_after = float(ra) if ra.replace(".", "", 1).isdigit() else None
                else:
                    detail = _detail(resp)
                    kind = "not_found" if code == 404 else "unavailable" if code in (401, 403) else "rejected"
                    return self._done({"error": kind, "message": detail, "status": code}, attempts, statuses, t0)
            if attempts > self.max_retries:
                break
            wait = retry_after if retry_after is not None else self.backoff_s * 2 ** (attempts - 1) * (0.5 + self.rng())
            if time.perf_counter() + wait >= deadline - 0.05:
                break  # the wait alone would exhaust the budget: give up now and say so
            self.sleep(wait)
        return self._done({"error": last_kind, "message": last_msg}, attempts, statuses, t0)

    def _done(self, body: dict, attempts: int, statuses: list, t0: float) -> dict[str, Any]:
        seconds = time.perf_counter() - t0
        out = dict(body)
        if "error" in out:
            out["attempts"] = attempts
        out["_meta"] = {"attempts": attempts, "statuses": statuses, "seconds": round(seconds, 4),
                        "within_budget": seconds <= self.budget_s + 0.25}
        return out


def _detail(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("detail", resp.text))[:200]
    except ValueError:
        return resp.text[:200]
