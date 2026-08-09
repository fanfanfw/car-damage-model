"""Managed-key assess rate-limit enforcement tests (Phase 4).

Covers the per-key_id fixed-window limiter, kill-switch semantics, policy
validation while enforcement is disabled, HTTP 429/Retry-After behavior, 4xx
usage recording, middleware ordering (limiter after auth, before upload and
inference), root-path consistency, key isolation, and capacity/cleanup bounds.
"""

import asyncio
import io

import httpx
import pytest
from fastapi.testclient import TestClient

from control_plane import (
    AuthGateway,
    AuthResult,
    ControlPlaneUnavailable,
    FixedWindowRateLimiter,
    RateLimit,
)

from .conftest import (
    ASSESS_SCOPE,
    KEY_ID,
    VALID_KEY,
    FakeAuthGateway,
    FakeUsageGateway,
    make_app,
    make_image,
    make_settings,
)

KEY_ID_2 = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class FakeClock:
    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def policy(requests, window_seconds=60):
    return {"assess": RateLimit(requests=requests, window_seconds=window_seconds)}


def _post(client, key=VALID_KEY, content=None, mime="image/jpeg", path="/assess"):
    files = {"image": ("car.jpg", io.BytesIO(content or make_image("JPEG")), mime)}
    headers = {"x-api-key": key} if key else {}
    return client.post(path, files=files, headers=headers)


class _PerKeyGateway:
    """Returns a distinct key_id per raw key, all sharing one policy."""

    def __init__(self, mapping, rate_limits=None):
        self.mapping = mapping
        self.rate_limits = rate_limits
        self.attempts = []

    async def authenticate(self, api_key, scope):
        self.attempts.append(api_key)
        key_id = self.mapping.get(api_key)
        if not key_id:
            return AuthResult(active=False)
        return AuthResult(active=True, key_id=key_id, scopes=(ASSESS_SCOPE,), rate_limits=self.rate_limits)


# ---------------------------------------------------------------------------
# Limiter unit tests (no HTTP, no sleeps, injectable clock)
# ---------------------------------------------------------------------------


class TestLimiterUnit:
    def test_n_accepted_then_n_plus_one_denied(self):
        limiter = FixedWindowRateLimiter(clock=FakeClock(0.0))

        async def run():
            admitted = [await limiter.check_and_increment("k", 3, 60) for _ in range(4)]
            assert admitted == [None, None, None, 60]

        asyncio.run(run())

    @pytest.mark.parametrize("position,expected", [(0.5, 60), (30.5, 30), (59.5, 1)])
    def test_retry_after_at_window_positions(self, position, expected):
        limiter = FixedWindowRateLimiter(clock=FakeClock(position))

        async def run():
            assert await limiter.check_and_increment("k", 1, 60) is None
            assert await limiter.check_and_increment("k", 1, 60) == expected

        asyncio.run(run())

    def test_deny_all_stores_no_bucket(self):
        limiter = FixedWindowRateLimiter(clock=FakeClock(0.0))

        async def run():
            assert await limiter.check_and_increment("k", 0, 60) == 60
            assert limiter._buckets == {}

        asyncio.run(run())

    def test_keys_are_isolated(self):
        limiter = FixedWindowRateLimiter(clock=FakeClock(0.0))

        async def run():
            assert await limiter.check_and_increment("a", 1, 60) is None
            assert await limiter.check_and_increment("a", 1, 60) == 60
            assert await limiter.check_and_increment("b", 1, 60) is None
            assert await limiter.check_and_increment("b", 1, 60) == 60

        asyncio.run(run())

    def test_new_key_id_gets_fresh_bucket(self):
        limiter = FixedWindowRateLimiter(clock=FakeClock(0.0))

        async def run():
            assert await limiter.check_and_increment("rotated", 1, 60) is None
            assert await limiter.check_and_increment("rotated", 1, 60) == 60
            assert await limiter.check_and_increment("fresh", 1, 60) is None

        asyncio.run(run())

    def test_window_resets_with_fake_clock(self):
        clock = FakeClock(0.0)
        limiter = FixedWindowRateLimiter(clock=clock)

        async def run():
            assert await limiter.check_and_increment("k", 1, 60) is None
            assert await limiter.check_and_increment("k", 1, 60) == 60
            clock.advance(60)
            assert await limiter.check_and_increment("k", 1, 60) is None

        asyncio.run(run())

    def test_stale_buckets_are_cleaned(self):
        clock = FakeClock(0.0)
        limiter = FixedWindowRateLimiter(clock=clock, max_buckets=1000)

        async def run():
            for i in range(50):
                assert await limiter.check_and_increment(f"key-{i}", 1, 60) is None
            assert len(limiter._buckets) == 50
            clock.advance(61)
            assert await limiter.check_and_increment("key-0", 1, 60) is None
            assert len(limiter._buckets) == 1

        asyncio.run(run())

    def test_capacity_fails_closed(self):
        limiter = FixedWindowRateLimiter(clock=FakeClock(0.0), max_buckets=2)

        async def run():
            assert await limiter.check_and_increment("a", 1, 60) is None
            assert await limiter.check_and_increment("b", 1, 60) is None
            assert await limiter.check_and_increment("c", 1, 60) == 60
            assert len(limiter._buckets) == 2

        asyncio.run(run())

    def test_concurrency_never_exceeds_limit(self):
        limiter = FixedWindowRateLimiter(clock=FakeClock(5.0))

        async def run():
            results = await asyncio.gather(
                *[limiter.check_and_increment("k", 2, 60) for _ in range(8)]
            )
            assert results.count(None) == 2

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Kill-switch semantics
# ---------------------------------------------------------------------------


class TestKillSwitch:
    def _client(self, enforcement, limit=1):
        settings = make_settings(API_KEY_RATE_LIMIT_ENFORCEMENT=enforcement)
        return TestClient(
            make_app(
                settings=settings,
                auth_gateway=FakeAuthGateway(rate_limits=policy(limit)),
                rate_limiter=FixedWindowRateLimiter(clock=FakeClock(0.0)),
            )
        )

    def test_absent_env_preserves_old_behavior(self):
        settings = make_settings()
        assert not settings.api_key_rate_limit_enforcement
        client = TestClient(
            make_app(
                settings=settings,
                auth_gateway=FakeAuthGateway(rate_limits=policy(1)),
                rate_limiter=FixedWindowRateLimiter(clock=FakeClock(0.0)),
            )
        )
        assert _post(client).status_code == 200
        assert _post(client).status_code == 200

    def test_explicit_false_disables(self):
        client = self._client("false")
        assert _post(client).status_code == 200
        assert _post(client).status_code == 200

    def test_explicit_true_enables(self):
        client = self._client("true")
        assert _post(client).status_code == 200
        assert _post(client).status_code == 429

    def test_invalid_boolean_rejected_at_config_time(self):
        with pytest.raises(ValueError):
            make_settings(API_KEY_RATE_LIMIT_ENFORCEMENT="maybe")


class TestPolicyValidationWhileDisabled:
    def _malformed_client(self, enforcement):
        def handler(request):
            return httpx.Response(
                200,
                json={
                    "active": True,
                    "key_id": KEY_ID,
                    "scopes": [ASSESS_SCOPE],
                    "rate_limits": {"assess": {"requests": 10}},
                },
            )

        kwargs = {} if enforcement is None else {"API_KEY_RATE_LIMIT_ENFORCEMENT": enforcement}
        settings = make_settings(**kwargs)
        gateway = AuthGateway(settings, transport=httpx.MockTransport(handler))
        return TestClient(
            make_app(
                settings=settings,
                auth_gateway=gateway,
                rate_limiter=FixedWindowRateLimiter(clock=FakeClock(0.0)),
            )
        )

    def test_malformed_policy_503_with_enforcement_disabled(self):
        client = self._malformed_client(None)
        assert _post(client).status_code == 503

    def test_malformed_policy_503_with_enforcement_enabled(self):
        client = self._malformed_client("true")
        assert _post(client).status_code == 503


# ---------------------------------------------------------------------------
# HTTP behavior
# ---------------------------------------------------------------------------


class TestHTTPEnforcement:
    def _client(self, usage=None, clock_start=0.0):
        clock = FakeClock(clock_start)
        return (
            TestClient(
                make_app(
                    settings=make_settings(API_KEY_RATE_LIMIT_ENFORCEMENT="true"),
                    auth_gateway=FakeAuthGateway(rate_limits=policy(1)),
                    rate_limiter=FixedWindowRateLimiter(clock=clock),
                    usage_gateway=usage or FakeUsageGateway(),
                )
            ),
            clock,
        )

    def test_429_body_and_retry_after(self):
        usage = FakeUsageGateway()
        client, _ = self._client(usage=usage)
        assert _post(client).status_code == 200
        response = _post(client)
        assert response.status_code == 429
        assert response.json() == {"detail": "Rate limit exceeded"}
        assert response.headers["retry-after"] == "60"
        assert usage.records[-1]["status_class"] == "4xx"
        assert usage.records[-1]["key_id"] == KEY_ID
        assert usage.records[-1]["endpoint_template"] == "/assess"

    def test_retry_after_mid_window(self):
        client, clock = self._client()
        assert _post(client).status_code == 200
        clock.advance(30.5)
        response = _post(client)
        assert response.status_code == 429
        assert response.headers["retry-after"] == "30"

    def test_window_reset_at_http_level(self):
        client, clock = self._client()
        assert _post(client).status_code == 200
        assert _post(client).status_code == 429
        clock.advance(60)
        assert _post(client).status_code == 200

    def test_deny_all_always_429(self):
        client = TestClient(
            make_app(
                settings=make_settings(API_KEY_RATE_LIMIT_ENFORCEMENT="true"),
                auth_gateway=FakeAuthGateway(rate_limits=policy(0)),
                rate_limiter=FixedWindowRateLimiter(clock=FakeClock(0.0)),
            )
        )
        assert _post(client).status_code == 429
        assert _post(client).status_code == 429

    def test_two_keys_are_isolated(self):
        gateway = _PerKeyGateway({VALID_KEY: KEY_ID, VALID_KEY + "-extra": KEY_ID_2}, rate_limits=policy(1))
        client = TestClient(
            make_app(
                settings=make_settings(API_KEY_RATE_LIMIT_ENFORCEMENT="true"),
                auth_gateway=gateway,
                rate_limiter=FixedWindowRateLimiter(clock=FakeClock(0.0)),
            )
        )
        assert _post(client, key=VALID_KEY).status_code == 200
        assert _post(client, key=VALID_KEY).status_code == 429
        assert _post(client, key=VALID_KEY + "-extra").status_code == 200

    def test_new_key_id_gets_fresh_bucket_at_http_level(self):
        gateway = _PerKeyGateway({VALID_KEY: KEY_ID}, rate_limits=policy(1))
        client = TestClient(
            make_app(
                settings=make_settings(API_KEY_RATE_LIMIT_ENFORCEMENT="true"),
                auth_gateway=gateway,
                rate_limiter=FixedWindowRateLimiter(clock=FakeClock(0.0)),
            )
        )
        assert _post(client).status_code == 200
        assert _post(client).status_code == 429

    def test_absent_policy_preserves_old_behavior(self):
        client = TestClient(
            make_app(
                settings=make_settings(API_KEY_RATE_LIMIT_ENFORCEMENT="true"),
                auth_gateway=FakeAuthGateway(),
                rate_limiter=FixedWindowRateLimiter(clock=FakeClock(0.0)),
            )
        )
        assert _post(client).status_code == 200
        assert _post(client).status_code == 200

    def test_invalid_key_still_401(self):
        client = TestClient(
            make_app(
                settings=make_settings(API_KEY_RATE_LIMIT_ENFORCEMENT="true"),
                auth_gateway=FakeAuthGateway(active=False, rate_limits=policy(1)),
                rate_limiter=FixedWindowRateLimiter(clock=FakeClock(0.0)),
            )
        )
        assert _post(client).status_code == 401

    def test_control_plane_outage_still_503(self):
        client = TestClient(
            make_app(
                settings=make_settings(API_KEY_RATE_LIMIT_ENFORCEMENT="true"),
                auth_gateway=FakeAuthGateway(raise_error=ControlPlaneUnavailable(), rate_limits=policy(1)),
                rate_limiter=FixedWindowRateLimiter(clock=FakeClock(0.0)),
            )
        )
        assert _post(client).status_code == 503

    def test_limiter_runs_before_upload_parsing(self):
        settings = make_settings(API_KEY_RATE_LIMIT_ENFORCEMENT="true", MAX_UPLOAD_BYTES="4096")
        client = TestClient(
            make_app(
                settings=settings,
                auth_gateway=FakeAuthGateway(rate_limits=policy(1)),
                rate_limiter=FixedWindowRateLimiter(clock=FakeClock(0.0)),
            )
        )
        assert _post(client).status_code == 200
        oversized = make_image("JPEG") + b"\x00" * 5000
        response = _post(client, content=oversized)
        assert response.status_code == 429

    def test_public_routes_unaffected_under_deny_all(self):
        client = TestClient(
            make_app(
                settings=make_settings(API_KEY_RATE_LIMIT_ENFORCEMENT="true"),
                auth_gateway=FakeAuthGateway(rate_limits=policy(0)),
                rate_limiter=FixedWindowRateLimiter(clock=FakeClock(0.0)),
            )
        )
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code in (200, 503)
        assert client.get("/policy").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200

    def test_non_assess_routes_and_wrong_method_not_limited(self):
        client = TestClient(
            make_app(
                settings=make_settings(API_KEY_RATE_LIMIT_ENFORCEMENT="true"),
                auth_gateway=FakeAuthGateway(rate_limits=policy(0)),
                rate_limiter=FixedWindowRateLimiter(clock=FakeClock(0.0)),
            )
        )
        response = client.get("/assess", headers={"x-api-key": VALID_KEY})
        assert response.status_code == 405

    def test_root_path_shares_bucket_with_direct_path(self):
        clock = FakeClock(0.0)
        client = TestClient(
            make_app(
                settings=make_settings(API_KEY_RATE_LIMIT_ENFORCEMENT="true"),
                auth_gateway=FakeAuthGateway(rate_limits=policy(1)),
                rate_limiter=FixedWindowRateLimiter(clock=clock),
            ),
            root_path="/api",
        )
        assert _post(client, path="/api/assess").status_code == 200
        response = _post(client, path="/assess")
        assert response.status_code == 429
        assert response.headers["retry-after"] == "60"

    def test_concurrency_never_exceeds_limit(self):
        app = make_app(
            settings=make_settings(API_KEY_RATE_LIMIT_ENFORCEMENT="true"),
            auth_gateway=FakeAuthGateway(rate_limits=policy(2)),
            rate_limiter=FixedWindowRateLimiter(clock=FakeClock(0.0)),
        )

        async def scenario():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                files = {"image": ("c.jpg", io.BytesIO(make_image("JPEG")), "image/jpeg")}

                async def one():
                    return await client.post("/assess", files=files, headers={"x-api-key": VALID_KEY})

                responses = await asyncio.gather(*[one() for _ in range(6)])
                codes = sorted(response.status_code for response in responses)
                assert codes.count(200) == 2
                assert codes.count(429) == 4

        asyncio.run(scenario())
