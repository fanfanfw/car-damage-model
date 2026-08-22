"""Managed x-api-key client for the Car Damage Assessment API.

Client-side gateway to the Django API Key Management control plane,
mirroring the CapSolve integration pattern:

- AuthGateway validates a raw `x-api-key` value by calling the control plane
  introspection endpoint and caches only *positive* results, keyed by an
  HMAC of the raw key (the raw key never becomes a cache/dict key).
- UsageGateway aggregates request/error/latency statistics asynchronously and
  reports them in bounded batches; only non-sensitive dimensions are sent.

Fail-closed: any timeout, network error, credential rejection, throttling,
5xx, or malformed control-plane response surfaces as ControlPlaneUnavailable
(503). There is no local/legacy key fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from settings import Settings

_SENSITIVE = re.compile(r"(x-api-key|authorization|bearer|service_credential|api[_-]?key)\s*[:=]?\s*[^\s,;]+", re.I)


class _RedactingFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        return _SENSITIVE.sub("<redacted>", message)


logger = logging.getLogger("car_damage.control_plane")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(_RedactingFormatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)


class ControlPlaneUnavailable(Exception):
    """The control plane could not be reached or refused the request."""


RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 10_000
SUPPORTED_RATE_LIMIT_CATEGORIES = frozenset({"assess"})


@dataclass(frozen=True)
class RateLimit:
    requests: int
    window_seconds: int


@dataclass(frozen=True)
class AuthResult:
    active: bool
    key_id: str | None = None
    scopes: tuple = ()
    expires_at: str | None = None
    cache_ttl: int | None = None
    rate_limits: dict | None = None


@dataclass
class CacheEntry:
    result: AuthResult
    stored_at: float
    ttl: int


class IntrospectionCache:
    def __init__(self, max_entries: int = 4096):
        self._max = max_entries
        self._data: dict[str, CacheEntry] = {}

    @staticmethod
    def key(keying: bytes, service: str, method: str, endpoint_template: str, api_key: str) -> str:
        mac = hmac.new(keying, f"{service}|{method}|{endpoint_template}|{api_key}".encode(), hashlib.sha256)
        return mac.hexdigest()

    def get(self, hash_key: str) -> AuthResult | None:
        entry = self._data.get(hash_key)
        if entry is None:
            return None
        if time.monotonic() - entry.stored_at > entry.ttl:
            self._data.pop(hash_key, None)
            return None
        return entry.result

    def put(self, hash_key: str, result: AuthResult, ttl: int) -> None:
        if ttl <= 0:
            return
        if len(self._data) >= self._max and hash_key not in self._data:
            self._evict()
        self._data[hash_key] = CacheEntry(result, time.monotonic(), ttl)

    def _evict(self) -> None:
        oldest = min(self._data, key=lambda k: self._data[k].stored_at)
        self._data.pop(oldest, None)

    def clear(self) -> None:
        self._data.clear()


def _parse_rate_limit(value):
    if not isinstance(value, dict) or set(value) != {"requests", "window_seconds"}:
        raise ControlPlaneUnavailable
    requests = value.get("requests")
    window = value.get("window_seconds")
    if type(requests) is not int or type(window) is not int:
        raise ControlPlaneUnavailable
    if not 0 <= requests <= RATE_LIMIT_MAX_REQUESTS or window != RATE_LIMIT_WINDOW_SECONDS:
        raise ControlPlaneUnavailable
    return {"assess": RateLimit(requests=requests, window_seconds=window)}


class FixedWindowRateLimiter:
    """Bounded in-memory fixed-window counter keyed by managed key_id.

    Single-process only: a restart resets every counter and N API workers
    multiply the effective limit N-fold, so the shared Redis counter backend
    (loopback 127.0.0.1:6379, never public) must be enabled before scaling to
    multi-worker or multi-instance. Buckets are keyed by the introspection
    key_id UUID, never raw key material; a rotated/new key_id starts a fresh
    bucket. The clock is injectable (monotonic by default) so tests can advance
    time without sleeping. Stale buckets from older windows are removed on
    every check; at full capacity a new key fails closed (429).
    """

    def __init__(self, max_buckets: int = 10_000, clock=time.monotonic):
        self._max_buckets = max_buckets
        self._clock = clock
        self._buckets: dict[str, tuple[int, int]] = {}
        self._lock = asyncio.Lock()

    async def check_and_increment(self, key_id: str, limit: int, window_seconds: int) -> int | None:
        """Admit one request for key_id or deny with seconds until window reset.

        Returns None to admit, otherwise the 429 Retry-After value. A
        deny-all policy (requests=0) always denies and stores no bucket.
        """
        now = self._clock()
        window = int(now // window_seconds)
        retry = max(1, window_seconds - int(now % window_seconds))
        if limit <= 0:
            return retry
        async with self._lock:
            stale = [key for key, (bucket_window, _) in self._buckets.items() if bucket_window < window]
            for key in stale:
                self._buckets.pop(key, None)
            if key_id not in self._buckets:
                if len(self._buckets) >= self._max_buckets:
                    return retry
                self._buckets[key_id] = (window, 0)
            bucket_window, count = self._buckets[key_id]
            if bucket_window != window:
                self._buckets[key_id] = (window, 0)
                count = 0
            if count >= limit:
                return retry
            self._buckets[key_id] = (window, count + 1)
        return None


class AuthGateway:
    def __init__(self, settings: Settings, transport=None, cache: IntrospectionCache | None = None):
        self.settings = settings
        self.transport = transport
        self.cache = cache or IntrospectionCache(settings.auth_cache_max_entries)
        self._keying = settings.service_credential.encode()
        self._introspect_window_start = 0.0
        self._introspect_count = 0

    def _allow_introspection(self) -> bool:
        """Per-process token bucket so a key flood cannot exhaust the shared
        control-plane credential budget (which would 503 every service caller).
        Over-budget introspections fail closed as unavailable."""
        now = time.monotonic()
        if now - self._introspect_window_start >= 60.0:
            self._introspect_window_start = now
            self._introspect_count = 0
        if self._introspect_count >= self.settings.auth_max_introspections_per_minute:
            return False
        self._introspect_count += 1
        return True

    async def authenticate(self, api_key: str, method: str, endpoint_template: str) -> AuthResult:
        if not api_key or not api_key.isascii() or not 8 <= len(api_key) <= 512:
            return AuthResult(active=False)
        service = self.settings.service_id
        hash_key = IntrospectionCache.key(self._keying, service, method, endpoint_template, api_key)
        cached = self.cache.get(hash_key)
        if cached is not None:
            return cached
        if not self._allow_introspection():
            raise ControlPlaneUnavailable
        result = await self._introspect(api_key, service, method, endpoint_template)
        if result.active:
            self.cache.put(hash_key, result, self._effective_ttl(result))
        return result

    async def _introspect(self, api_key: str, service: str, method: str, endpoint_template: str) -> AuthResult:
        base = self.settings.control_plane_base_url
        if not base:
            raise ControlPlaneUnavailable
        headers = {"Authorization": f"Bearer {self.settings.service_credential}"}
        payload = {"api_key": api_key, "method": method, "endpoint_template": endpoint_template}
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.introspection_timeout_seconds),
                transport=self.transport,
            ) as client:
                response = await client.post(
                    f"{base}/internal/v2/keys/introspect",
                    json=payload,
                    headers=headers,
                )
        except httpx.HTTPError:
            raise ControlPlaneUnavailable from None
        if response.status_code in (401, 403):
            logger.warning(
                "control plane rejected introspection credential (status=%s); "
                "check SERVICE_CREDENTIAL expiry/revocation",
                response.status_code,
            )
            raise ControlPlaneUnavailable
        if response.status_code in (502, 503, 504, 429) or response.status_code >= 500:
            raise ControlPlaneUnavailable
        if response.status_code != 200:
            raise ControlPlaneUnavailable
        try:
            data = response.json()
        except ValueError:
            raise ControlPlaneUnavailable from None
        if not isinstance(data, dict) or not isinstance(data.get("active"), bool):
            raise ControlPlaneUnavailable
        cache_ttl = data.get("cache_ttl_seconds")
        if (
            cache_ttl is not None
            and (
                not isinstance(cache_ttl, int)
                or isinstance(cache_ttl, bool)
                or cache_ttl < 0
            )
        ):
            raise ControlPlaneUnavailable
        if not data["active"]:
            return AuthResult(active=False, cache_ttl=cache_ttl)
        key_id = data.get("key_id")
        scopes = data.get("scopes")
        expires_at = data.get("expires_at")
        try:
            uuid.UUID(key_id)
        except (ValueError, AttributeError, TypeError):
            raise ControlPlaneUnavailable from None
        if (
            not isinstance(scopes, list)
            or len(scopes) > 100
            or not all(
                isinstance(scope, str) and 1 <= len(scope) <= 128
                for scope in scopes
            )
        ):
            raise ControlPlaneUnavailable
        if expires_at is not None:
            if not isinstance(expires_at, str):
                raise ControlPlaneUnavailable
            try:
                parsed_expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            except ValueError:
                raise ControlPlaneUnavailable from None
            if parsed_expiry.tzinfo is None:
                raise ControlPlaneUnavailable
        return AuthResult(
            active=True,
            key_id=key_id,
            scopes=tuple(scopes),
            expires_at=expires_at,
            cache_ttl=cache_ttl,
            rate_limits=_parse_rate_limit(data.get("rate_limit")) if "rate_limit" in data else None,
        )

    def _effective_ttl(self, result: AuthResult) -> int:
        ttl = self.settings.auth_cache_max_ttl_seconds
        if result.cache_ttl is not None:
            try:
                ttl = min(ttl, max(0, int(result.cache_ttl)))
            except (TypeError, ValueError):
                pass
        if result.expires_at:
            try:
                expires = datetime.fromisoformat(result.expires_at.replace("Z", "+00:00"))
                remaining = (expires - datetime.now(UTC)).total_seconds()
                ttl = min(ttl, max(0, int(remaining)))
            except ValueError:
                pass
        return ttl


@dataclass
class _AggPoint:
    request_count: int = 0
    error_count: int = 0
    latency_sum_ms: int = 0
    latency_max_ms: int = 0


class UsageGateway:
    """Asynchronous, bounded usage reporting.

    Only non-sensitive dimensions are transmitted: bucket, key_id (UUID from
    introspection), endpoint template, method, status class, counts, latency.
    Never: filenames, image bytes, request bodies, raw URLs/queries, API keys,
    credentials, model results, bboxes, decisions, or assessment content.
    """

    def __init__(self, settings: Settings, transport=None):
        self.settings = settings
        self._transport = transport
        self._aggregates: dict[tuple, _AggPoint] = {}
        self._queue: deque = deque()
        self._lock = asyncio.Lock()
        self.drop_count = 0
        self.records_queued = 0
        self._task = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.flush_now()

    def record(
        self,
        key_id: str,
        method: str,
        endpoint_template: str,
        status_class: str,
        latency_ms: int,
    ) -> None:
        bucket = datetime.now(UTC).replace(second=0, microsecond=0)
        dim = (bucket, str(key_id), method, endpoint_template, status_class)
        point = self._aggregates.setdefault(dim, _AggPoint())
        point.request_count += 1
        point.error_count += 1 if status_class == "5xx" else 0
        point.latency_sum_ms += latency_ms
        point.latency_max_ms = max(point.latency_max_ms, latency_ms)

    async def flush_now(self) -> None:
        async with self._lock:
            if not self._aggregates and not self._queue:
                return
            records = self._drain_records()
            if records:
                self._enqueue_batch(records)
            await self._drain_queue()

    def _drain_records(self) -> list[dict]:
        records = []
        for dim, point in self._aggregates.items():
            bucket, key_id, method, template, status = dim
            records.append(
                {
                    "bucket_start": bucket.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "key_id": str(key_id),
                    "endpoint_template": template,
                    "method": method,
                    "status_class": status,
                    "request_count": point.request_count,
                    "error_count": point.error_count,
                    "latency_sum_ms": point.latency_sum_ms,
                    "latency_max_ms": point.latency_max_ms,
                }
            )
        self._aggregates.clear()
        return records

    def _enqueue_batch(self, records: list[dict]) -> None:
        limit = self.settings.usage_max_records_per_batch
        for start in range(0, len(records), limit):
            chunk = records[start : start + limit]
            self._queue.append({"batch_id": str(uuid.uuid4()), "records": chunk})
            self.records_queued += len(chunk)
        while (
            len(self._queue) > self.settings.usage_max_queue_batches
            or self.records_queued > self.settings.usage_max_queued_records
        ):
            dropped = self._queue.popleft()
            self.records_queued -= len(dropped["records"])
            self.drop_count += 1

    async def _drain_queue(self) -> None:
        pending = []
        while self._queue:
            pending.append(self._queue.popleft())
            self.records_queued -= len(pending[-1]["records"])
        for batch in pending:
            await self._send_with_retry(batch)

    async def _send_with_retry(self, batch: dict) -> None:
        base = self.settings.control_plane_base_url
        if not base:
            self.drop_count += 1
            return
        headers = {"Authorization": f"Bearer {self.settings.service_credential}"}
        attempts = self.settings.usage_retry_attempts
        for attempt in range(attempts):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self.settings.introspection_timeout_seconds),
                    transport=self._transport,
                ) as client:
                    response = await client.post(
                        f"{base}/internal/v1/usage/batches",
                        json=batch,
                        headers=headers,
                    )
                if response.status_code in (200, 202):
                    return
                if response.status_code in (429,) or response.status_code >= 500:
                    await asyncio.sleep(1.0 * (2**attempt))
                    continue
                return
            except httpx.HTTPError:
                await asyncio.sleep(1.0 * (2**attempt))
        self.drop_count += 1

    async def _loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.settings.usage_flush_interval_seconds)
            await self.flush_now()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()
