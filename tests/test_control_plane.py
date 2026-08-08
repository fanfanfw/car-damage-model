"""Unit tests for the AuthGateway and UsageGateway control-plane clients.

Covers Fase 9 items 5, 6, 7, 8 (client layer), plus TTL/revoke bounds.
No real network: httpx.MockTransport is injected.
"""

import asyncio
import json

import httpx
import pytest

from control_plane import (
    AuthGateway,
    AuthResult,
    ControlPlaneUnavailable,
    IntrospectionCache,
    UsageGateway,
)

from .conftest import make_settings

SERVICE_CRED = "s" * 40
KEY = "managed-key-abcdefghijklmnop"


def settings_for(**overrides):
    base = {
        "ENVIRONMENT": "production",
        "DEMO_FALLBACK_ENABLED": "false",
        "CONTROL_PLANE_BASE_URL": "http://100.124.216.27:18080",
        "SERVICE_CREDENTIAL": SERVICE_CRED,
        "ALLOWED_HOSTS": "testserver",
        "AUTH_CACHE_MAX_TTL_SECONDS": "30",
        "AUTH_CACHE_MAX_ENTRIES": "8",
        "USAGE_MAX_QUEUE_BATCHES": "4",
        "USAGE_MAX_QUEUED_RECORDS": "1000",
    }
    base.update(overrides)
    return make_settings(**base)


def ok_response(**changes):
    body = {
        "active": True,
        "key_id": "11111111-2222-3333-4444-555555555555",
        "owner_user_id": "42",
        "scopes": ["car-damage:assess"],
        "expires_at": None,
        "cache_ttl_seconds": 5,
    }
    body.update(changes)
    return httpx.Response(200, json=body)


def auth_gateway(handler, **settings_overrides):
    transport = httpx.MockTransport(handler)
    return AuthGateway(settings_for(**settings_overrides), transport=transport)


def run(coro):
    return asyncio.run(coro)


class TestAuthGatewaySuccess:
    def test_active_introspection(self):
        async def handler(request):
            assert request.headers["authorization"] == f"Bearer {SERVICE_CRED}"
            payload = json.loads(request.content)
            assert payload["api_key"] == KEY
            assert payload["required_scope"] == "car-damage:assess"
            return ok_response()

        gateway = auth_gateway(handler)
        result = run(gateway.authenticate(KEY, "car-damage:assess"))
        assert result.active
        assert result.key_id == "11111111-2222-3333-4444-555555555555"
        assert result.scopes == ("car-damage:assess",)

    def test_raw_key_is_not_a_cache_key(self):
        calls = {"n": 0}

        async def handler(request):
            calls["n"] += 1
            return ok_response()

        gateway = auth_gateway(handler)
        run(gateway.authenticate(KEY, "car-damage:assess"))
        assert all(KEY not in key for key in gateway.cache._data)
        assert all(SERVICE_CRED not in key for key in gateway.cache._data)

    def test_cache_hit_avoids_second_introspection(self):
        calls = {"n": 0}

        async def handler(request):
            calls["n"] += 1
            return ok_response()

        gateway = auth_gateway(handler)
        run(gateway.authenticate(KEY, "car-damage:assess"))
        run(gateway.authenticate(KEY, "car-damage:assess"))
        assert calls["n"] == 1

    def test_hmac_key_binds_service_scope_and_key(self):
        k1 = IntrospectionCache.key(b"k", "car-damage", "car-damage:assess", KEY)
        k2 = IntrospectionCache.key(b"k", "car-damage", "other:scope", KEY)
        k3 = IntrospectionCache.key(b"k", "other-service", "car-damage:assess", KEY)
        k4 = IntrospectionCache.key(b"k", "car-damage", "car-damage:assess", KEY + "x")
        assert len({k1, k2, k3, k4}) == 4

    def test_effective_ttl_bounded_by_response_and_expiry(self):
        gateway = auth_gateway(lambda request: ok_response())
        result = AuthResult(active=True, key_id="x", scopes=("s",), expires_at=None, cache_ttl=5)
        assert gateway._effective_ttl(result) == 5
        result2 = AuthResult(active=True, key_id="x", scopes=("s",), expires_at=None, cache_ttl=None)
        assert gateway._effective_ttl(result2) == 30
        result3 = AuthResult(
            active=True, key_id="x", scopes=("s",),
            expires_at="2099-01-01T00:00:00Z", cache_ttl=300,
        )
        assert gateway._effective_ttl(result3) <= 30

    def test_cache_eviction_bounds_entries(self):
        gateway = auth_gateway(lambda request: ok_response())
        for i in range(50):
            run(gateway.authenticate(f"key-{i}-abcdefghijklmnop", "car-damage:assess"))
        assert len(gateway.cache._data) <= 8

    def test_introspection_token_bucket_bounds_control_plane_calls(self):
        gateway = auth_gateway(
            lambda request: ok_response(), AUTH_MAX_INTROSPECTIONS_PER_MINUTE="2"
        )
        run(gateway.authenticate(f"{KEY}a", "car-damage:assess"))
        run(gateway.authenticate(f"{KEY}b", "car-damage:assess"))
        with pytest.raises(ControlPlaneUnavailable):
            run(gateway.authenticate(f"{KEY}c", "car-damage:assess"))


class TestAuthGatewayFailures:
    async def handler(self, request):
        raise AssertionError("should not be called")

    def test_control_plane_401_credential_rejected(self):
        gateway = auth_gateway(lambda request: httpx.Response(401))
        with pytest.raises(ControlPlaneUnavailable):
            run(gateway.authenticate(KEY, "car-damage:assess"))

    def test_control_plane_429(self):
        gateway = auth_gateway(lambda request: httpx.Response(429))
        with pytest.raises(ControlPlaneUnavailable):
            run(gateway.authenticate(KEY, "car-damage:assess"))

    def test_control_plane_5xx(self):
        gateway = auth_gateway(lambda request: httpx.Response(500))
        with pytest.raises(ControlPlaneUnavailable):
            run(gateway.authenticate(KEY, "car-damage:assess"))

    def test_network_timeout(self):
        async def handler(request):
            raise httpx.ConnectTimeout("boom")

        gateway = auth_gateway(handler)
        with pytest.raises(ControlPlaneUnavailable):
            run(gateway.authenticate(KEY, "car-damage:assess"))

    def test_malformed_json(self):
        gateway = auth_gateway(lambda request: httpx.Response(200, text="not json"))
        with pytest.raises(ControlPlaneUnavailable):
            run(gateway.authenticate(KEY, "car-damage:assess"))

    def test_malformed_shape(self):
        gateway = auth_gateway(lambda request: httpx.Response(200, json={"nope": True}))
        with pytest.raises(ControlPlaneUnavailable):
            run(gateway.authenticate(KEY, "car-damage:assess"))

    def test_inactive_result_not_cached_and_reported_inactive(self):
        async def handler(request):
            return httpx.Response(200, json={"active": False, "cache_ttl_seconds": 0})

        gateway = auth_gateway(handler)
        result = run(gateway.authenticate(KEY, "car-damage:assess"))
        assert not result.active
        assert len(gateway.cache._data) == 0

    def test_raw_key_validation(self):
        gateway = auth_gateway(self.handler)
        for bad in ("", "short", "ü" * 10, "x" * 600):
            result = run(gateway.authenticate(bad, "car-damage:assess"))
            assert not result.active


class TestIntrospectionContract:
    def test_valid_rate_limits_are_parsed(self):
        gateway = auth_gateway(lambda request: ok_response(rate_limits={"assess": {"requests": 10, "window_seconds": 60}}))
        result = run(gateway.authenticate(KEY, "car-damage:assess"))
        assert result.active
        assert result.rate_limits["assess"].requests == 10
        assert result.rate_limits["assess"].window_seconds == 60

    def test_absent_rate_limits_are_tolerated(self):
        gateway = auth_gateway(lambda request: ok_response())
        result = run(gateway.authenticate(KEY, "car-damage:assess"))
        assert result.active
        assert result.rate_limits is None

    def test_unknown_response_fields_are_tolerated(self):
        gateway = auth_gateway(lambda request: ok_response(future_field={"anything": [1, 2, 3]}))
        result = run(gateway.authenticate(KEY, "car-damage:assess"))
        assert result.active
        assert result.key_id == "11111111-2222-3333-4444-555555555555"

    def test_malformed_rate_limits_fail_closed(self):
        bad_values = [
            "10",
            {},
            {"assess": "10"},
            {"": {"requests": 10, "window_seconds": 60}},
            {"unknown": {"requests": 10, "window_seconds": 60}},
            {
                "assess": {"requests": 10, "window_seconds": 60},
                "submit": {"requests": 30, "window_seconds": 60},
            },
            {"assess": {"requests": 10}},
            {"assess": {"window_seconds": 60}},
            {"assess": {"requests": 10, "window_seconds": 60, "burst": 5}},
            {"assess": {"requests": "10", "window_seconds": 60}},
            {"assess": {"requests": 10, "window_seconds": "60"}},
            {"assess": {"requests": 10, "window_seconds": 5}},
            {"assess": {"requests": -1, "window_seconds": 60}},
            {"assess": {"requests": 10_001, "window_seconds": 60}},
            {"assess": {"requests": True, "window_seconds": 60}},
        ]
        for rate_limits in bad_values:
            gateway = auth_gateway(lambda request, value=rate_limits: ok_response(rate_limits=value))
            with pytest.raises(ControlPlaneUnavailable):
                run(gateway.authenticate(KEY, "car-damage:assess"))

    def test_malformed_cache_ttl_fails_closed(self):
        gateway = auth_gateway(lambda request: ok_response(cache_ttl_seconds="5"))
        with pytest.raises(ControlPlaneUnavailable):
            run(gateway.authenticate(KEY, "car-damage:assess"))


class TestUsageGateway:
    def usage_gateway(self, handler, **overrides):
        transport = httpx.MockTransport(handler)
        return UsageGateway(settings_for(**overrides), transport=transport)

    def test_flush_sends_safe_dimensions_only(self):
        sent = {}

        async def handler(request):
            sent["payload"] = json.loads(request.content)
            return httpx.Response(202, json={"accepted": True})

        gateway = self.usage_gateway(handler)
        run(gateway.start())
        gateway.record("11111111-2222-3333-4444-555555555555", "POST", "/assess", "2xx", 120)
        run(gateway.stop())
        records = sent["payload"]["records"]
        assert records[0]["key_id"] == "11111111-2222-3333-4444-555555555555"
        assert records[0]["endpoint_template"] == "/assess"
        assert records[0]["method"] == "POST"
        assert records[0]["status_class"] == "2xx"
        assert records[0]["request_count"] == 1
        assert records[0]["error_count"] == 0
        assert records[0]["latency_max_ms"] == 120
        payload_text = json.dumps(sent["payload"])
        assert KEY not in payload_text
        assert SERVICE_CRED not in payload_text
        assert "filename" not in payload_text
        assert "image" not in payload_text
        assert "bbox" not in payload_text
        assert "decision" not in payload_text

    def test_5xx_counts_error(self):
        sent = {}

        async def handler(request):
            sent["payload"] = json.loads(request.content)
            return httpx.Response(202)

        gateway = self.usage_gateway(handler)
        run(gateway.start())
        gateway.record("11111111-2222-3333-4444-555555555555", "POST", "/assess", "5xx", 500)
        run(gateway.stop())
        record = sent["payload"]["records"][0]
        assert record["error_count"] == 1
        assert record["request_count"] == 1

    def test_retry_then_success(self):
        calls = {"n": 0}

        async def handler(request):
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(500)
            return httpx.Response(202)

        gateway = self.usage_gateway(handler, USAGE_RETRY_ATTEMPTS="5")
        run(gateway.start())
        gateway.record("11111111-2222-3333-4444-555555555555", "POST", "/assess", "2xx", 10)
        run(gateway.stop())
        assert calls["n"] == 3
        assert gateway.drop_count == 0

    def test_queue_is_bounded_and_drops_oldest(self):
        async def handler(request):
            return httpx.Response(202)

        gateway = self.usage_gateway(handler, USAGE_MAX_QUEUE_BATCHES="2", USAGE_MAX_RECORDS_PER_BATCH="1")
        for i in range(10):
            gateway.record(f"11111111-2222-3333-4444-55555555555{i}", "POST", "/assess", "2xx", 1)
        run(gateway.flush_now())
        assert gateway.records_queued <= 2
        assert gateway.drop_count > 0

    def test_no_control_plane_drops_without_network(self):
        gateway = self.usage_gateway(
            lambda request: httpx.Response(202),
            ENVIRONMENT="development",
            CONTROL_PLANE_BASE_URL="",
        )
        run(gateway.start())
        gateway.record("11111111-2222-3333-4444-555555555555", "POST", "/assess", "2xx", 1)
        run(gateway.stop())
        assert gateway.drop_count == 1


class TestProductionConfigGuards:
    def test_production_http_control_plane_must_be_tailscale_or_loopback(self):
        with pytest.raises(ValueError):
            make_settings(
                ENVIRONMENT="production",
                CONTROL_PLANE_BASE_URL="http://10.0.0.5:18080",
                DEMO_FALLBACK_ENABLED=None,
            )
        # Tailscale peer is allowed.
        settings = make_settings(
            ENVIRONMENT="production",
            CONTROL_PLANE_BASE_URL="http://100.124.216.27:18080",
            DEMO_FALLBACK_ENABLED=None,
        )
        assert settings.control_plane_base_url == "http://100.124.216.27:18080"
        # Loopback (colocated control-plane listener) is allowed, pinned to 18083.
        settings = make_settings(
            ENVIRONMENT="production",
            CONTROL_PLANE_BASE_URL="http://127.0.0.1:18083",
            DEMO_FALLBACK_ENABLED=None,
        )
        assert settings.control_plane_base_url == "http://127.0.0.1:18083"
        # Loopback must NOT point at gunicorn (18080) or another localhost port.
        for bad in ("http://127.0.0.1:18080", "http://127.0.0.1:8080", "http://127.0.0.1:18083/extra"):
            with pytest.raises(ValueError):
                make_settings(
                    ENVIRONMENT="production",
                    CONTROL_PLANE_BASE_URL=bad,
                    DEMO_FALLBACK_ENABLED=None,
                )
        # A public https host is allowed.
        settings = make_settings(
            ENVIRONMENT="production",
            CONTROL_PLANE_BASE_URL="https://control.example",
            SERVICE_CREDENTIAL="s" * 40,
            ALLOWED_HOSTS="testserver",
            DEMO_FALLBACK_ENABLED=None,
        )
        assert settings.control_plane_base_url == "https://control.example"

    def test_production_requires_control_plane(self):
        with pytest.raises(ValueError):
            make_settings(
                ENVIRONMENT="production",
                CONTROL_PLANE_BASE_URL=None,
                DEMO_FALLBACK_ENABLED=None,
            )

    def test_production_requires_service_credential(self):
        with pytest.raises(ValueError):
            make_settings(
                ENVIRONMENT="production",
                CONTROL_PLANE_BASE_URL="http://100.124.216.27:18080",
                SERVICE_CREDENTIAL=None,
                DEMO_FALLBACK_ENABLED=None,
            )

    def test_production_forbids_wildcard_host(self):
        with pytest.raises(ValueError):
            make_settings(
                ENVIRONMENT="production",
                CONTROL_PLANE_BASE_URL="http://100.124.216.27:18080",
                ALLOWED_HOSTS="*",
                DEMO_FALLBACK_ENABLED=None,
            )

    def test_production_forwarded_allow_ips_loopback_only(self):
        with pytest.raises(ValueError):
            make_settings(
                ENVIRONMENT="production",
                CONTROL_PLANE_BASE_URL="http://100.124.216.27:18080",
                FORWARDED_ALLOW_IPS="10.0.0.1",
                DEMO_FALLBACK_ENABLED=None,
            )

    def test_production_forbids_wildcard_cors(self):
        with pytest.raises(ValueError):
            make_settings(
                ENVIRONMENT="production",
                CONTROL_PLANE_BASE_URL="http://100.124.216.27:18080",
                CORS_ALLOWED_ORIGINS="*",
                DEMO_FALLBACK_ENABLED=None,
            )
