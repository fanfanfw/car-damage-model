"""Hardening tests: upload limits, image validation, host/CORS/proxy trust,
production vs development fallback, concurrency, and data leakage.

Covers Fase 9 items 10-16, 19, 20.
"""

import asyncio
import io
import logging

import httpx
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from starlette.requests import Request

import api as api_module

from .conftest import (
    VALID_KEY,
    FakeUsageGateway,
    make_app,
    make_image,
    make_png_header,
    make_settings,
)

AUTH = {"x-api-key": VALID_KEY}


def _post(client, content=None, mime="image/jpeg", filename="car.jpg", key=VALID_KEY, headers=None):
    files = {"image": (filename, io.BytesIO(content or make_image("JPEG")), mime)}
    merged = {}
    if key is not None:
        merged.update(AUTH)
    if headers:
        merged.update(headers)
    return client.post("/assess", files=files, headers=merged)


class TestUploadLimits:
    def _small_limit_settings(self, limit):
        return make_settings(MAX_UPLOAD_BYTES=str(limit))

    def test_upload_within_limit_passes(self):
        limit = 4 * 1024
        content = make_image("JPEG")
        assert len(content) < limit
        client = TestClient(make_app(settings=self._small_limit_settings(limit)))
        response = _post(client, content=content)
        assert response.status_code == 200

    def test_upload_one_byte_over_limit_is_413(self):
        limit = 4 * 1024
        content = make_image("JPEG") + b"\x00" * limit
        client = TestClient(make_app(settings=self._small_limit_settings(limit)))
        response = _post(client, content=content)
        assert response.status_code == 413
        assert response.json() == {"detail": "Image file exceeds the allowed size."}

    def test_declared_content_length_too_large_is_413(self):
        limit = 4 * 1024
        client = TestClient(make_app(settings=self._small_limit_settings(limit)))
        content = make_image("JPEG")
        response = _post(
            client,
            content=content,
            headers={**AUTH, "content-length": str(limit + 999999)},
        )
        assert response.status_code == 413

    def test_upload_boundary_unit(self):
        """Content-Length exactly at the limit passes; limit+1 is rejected."""

        async def run():
            settings = make_settings(MAX_UPLOAD_BYTES="4096")
            payload = b"a" * 4096
            upload = UploadFile(file=io.BytesIO(payload), filename="x.jpg")
            request = Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/assess",
                    "headers": [(b"content-length", b"4096")],
                }
            )
            data = await api_module.read_upload_bounded(request, upload, settings)
            assert data == payload
            upload.file.close()

            upload = UploadFile(file=io.BytesIO(b"a" * 4097), filename="x.jpg")
            request = Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/assess",
                    "headers": [(b"content-length", b"4097")],
                }
            )
            with pytest.raises(api_module.UploadTooLarge):
                await api_module.read_upload_bounded(request, upload, settings)
            upload.file.close()

        asyncio.run(run())

    def test_chunked_read_unit(self):
        async def run():
            settings = make_settings(MAX_UPLOAD_BYTES="2048")
            upload = UploadFile(file=io.BytesIO(b"a" * 4096), filename="x.jpg")
            request = Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/assess",
                    "headers": [],
                }
            )
            with pytest.raises(api_module.UploadTooLarge):
                await api_module.read_upload_bounded(request, upload, settings)
            upload.file.close()

        asyncio.run(run())

    def test_chunked_within_limit_reads_all_bytes(self):
        async def run():
            settings = make_settings(MAX_UPLOAD_BYTES="100000")
            payload = b"a" * 60_000
            upload = UploadFile(file=io.BytesIO(payload), filename="x.jpg")
            request = Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/assess",
                    "headers": [],
                }
            )
            data = await api_module.read_upload_bounded(request, upload, settings)
            assert data == payload
            upload.file.close()

        asyncio.run(run())


class TestImageValidation:
    def test_corrupt_bytes_rejected(self):
        client = TestClient(make_app())
        response = _post(client, content=b"\x00\x01\x02\x03" * 500, mime="image/jpeg")
        assert response.status_code == 400

    def test_truncated_image_rejected(self):
        client = TestClient(make_app())
        data = make_image("JPEG")
        response = _post(client, content=data[: len(data) // 2])
        assert response.status_code == 400

    def test_mime_spoof_rejected(self):
        client = TestClient(make_app())
        response = _post(client, content=b"plain text " * 200, mime="image/png")
        assert response.status_code == 400

    def test_unsupported_mime_rejected(self):
        client = TestClient(make_app())
        response = _post(client, content=make_image("JPEG"), mime="application/pdf")
        assert response.status_code == 400

    def test_decompression_bomb_header_rejected(self):
        client = TestClient(make_app())
        bomb = make_png_header(40000, 40000)
        response = _post(client, content=bomb, mime="image/png")
        assert response.status_code == 400

    def test_pixel_limit_rejected(self):
        settings = make_settings(MAX_IMAGE_PIXELS="100000")
        client = TestClient(make_app(settings=settings))
        response = _post(client, content=make_image("PNG", size=(400, 400)), mime="image/png")
        assert response.status_code == 400


class TestProductionVsDevelopment:
    def _patch_model_unavailable(self):
        old_cv, old_agent = api_module.CV_AVAILABLE, api_module.AGENT_AVAILABLE
        api_module.CV_AVAILABLE = False
        api_module.AGENT_AVAILABLE = False
        return old_cv, old_agent

    def test_production_model_failure_is_503_not_demo(self):
        old = self._patch_model_unavailable()
        try:
            settings = make_settings(
                ENVIRONMENT="production",
                CONTROL_PLANE_BASE_URL="http://100.124.216.27:18080",
                DEMO_FALLBACK_ENABLED=None,
            )
            client = TestClient(make_app(settings=settings))
            response = _post(client)
            assert response.status_code == 503
            assert response.json() == {"detail": "Assessment service unavailable."}
        finally:
            api_module.CV_AVAILABLE, api_module.AGENT_AVAILABLE = old

    def test_production_rejects_demo_flag_at_config_time(self):
        with pytest.raises(ValueError):
            make_settings(
                ENVIRONMENT="production",
                CONTROL_PLANE_BASE_URL="http://100.124.216.27:18080",
                DEMO_FALLBACK_ENABLED="true",
            )

    def test_development_without_flag_is_503(self):
        old = self._patch_model_unavailable()
        try:
            settings = make_settings(DEMO_FALLBACK_ENABLED=None)
            client = TestClient(make_app(settings=settings))
            response = _post(client)
            assert response.status_code == 503
        finally:
            api_module.CV_AVAILABLE, api_module.AGENT_AVAILABLE = old

    def test_development_with_flag_is_200(self):
        old = self._patch_model_unavailable()
        try:
            settings = make_settings(DEMO_FALLBACK_ENABLED="true")
            client = TestClient(make_app(settings=settings))
            response = _post(client)
            assert response.status_code == 200
            assert response.json()["cv_backend"] in ("demo", "demo-fallback")
        finally:
            api_module.CV_AVAILABLE, api_module.AGENT_AVAILABLE = old


class TestHostAndProxy:
    def test_unknown_host_rejected(self):
        settings = make_settings(ALLOWED_HOSTS="testserver")
        client = TestClient(make_app(settings=settings))
        response = _post(client, headers={**AUTH, "host": "evil.example"})
        assert response.status_code == 400
        assert response.json() == {"detail": "Invalid host header"}

    def test_known_host_accepted(self):
        settings = make_settings(ALLOWED_HOSTS="testserver")
        client = TestClient(make_app(settings=settings))
        assert _post(client).status_code == 200

    def test_forwarded_for_spoof_cannot_bypass_auth(self):
        client = TestClient(make_app())
        response = _post(
            client,
            headers={**AUTH, "x-forwarded-for": "6.6.6.6", "x-real-ip": "6.6.6.6"},
        )
        assert response.status_code == 200
        response = _post(
            client, key=None, headers={"x-forwarded-for": "6.6.6.6"}
        )
        assert response.status_code == 401

    def test_forwarded_proto_spoof_does_not_bypass_host(self):
        settings = make_settings(ALLOWED_HOSTS="testserver")
        client = TestClient(make_app(settings=settings))
        response = _post(
            client,
            headers={**AUTH, "x-forwarded-proto": "https"},
        )
        assert response.status_code == 200


class TestCors:
    def _cors_client(self):
        settings = make_settings(CORS_ALLOWED_ORIGINS="https://allowed.example")
        return TestClient(make_app(settings=settings))

    def test_approved_origin_gets_cors_headers(self):
        client = self._cors_client()
        response = client.options(
            "/assess",
            headers={
                "Origin": "https://allowed.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-api-key,content-type",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "https://allowed.example"
        assert "access-control-allow-credentials" not in response.headers

    def test_unapproved_origin_gets_no_cors_headers(self):
        client = self._cors_client()
        response = client.options(
            "/assess",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" not in response.headers

    def test_no_cors_when_not_configured(self):
        client = TestClient(make_app())
        response = client.options(
            "/assess",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert "access-control-allow-origin" not in response.headers


class TestConcurrency:
    def test_concurrency_bound_returns_503_when_busy(self):
        settings = make_settings(MAX_CONCURRENT_ASSESSMENTS="1")
        app = make_app(settings=settings)
        semaphore = app.state._cd.assess_semaphore

        async def scenario():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                files = {"image": ("c.jpg", io.BytesIO(make_image("JPEG")), "image/jpeg")}
                await semaphore.acquire()
                try:
                    response = await client.post("/assess", files=files, headers=AUTH)
                    assert response.status_code == 503
                finally:
                    semaphore.release()
                response = await client.post("/assess", files=files, headers=AUTH)
                assert response.status_code == 200

        asyncio.run(scenario())


class TestUsageRecording:
    def test_usage_2xx(self):
        gateway = FakeUsageGateway()
        client = TestClient(make_app(usage_gateway=gateway))
        assert _post(client).status_code == 200
        assert gateway.records[-1]["status_class"] == "2xx"
        assert gateway.records[-1]["endpoint_template"] == "/assess"
        assert gateway.records[-1]["key_id"] == "11111111-2222-3333-4444-555555555555"

    def test_usage_4xx(self):
        gateway = FakeUsageGateway()
        settings = make_settings(MAX_UPLOAD_BYTES="4096")
        client = TestClient(make_app(settings=settings, usage_gateway=gateway))
        content = make_image("JPEG") + b"\x00" * 5000
        response = _post(client, content=content)
        assert response.status_code == 413
        assert gateway.records[-1]["status_class"] == "4xx"

    def test_usage_5xx(self):
        gateway = FakeUsageGateway()
        old_cv, old_agent = api_module.CV_AVAILABLE, api_module.AGENT_AVAILABLE
        api_module.CV_AVAILABLE = False
        api_module.AGENT_AVAILABLE = False
        try:
            settings = make_settings(
                ENVIRONMENT="production",
                CONTROL_PLANE_BASE_URL="http://100.124.216.27:18080",
                DEMO_FALLBACK_ENABLED=None,
            )
            client = TestClient(make_app(settings=settings, usage_gateway=gateway))
            response = _post(client)
            assert response.status_code == 503
            assert gateway.records[-1]["status_class"] == "5xx"
        finally:
            api_module.CV_AVAILABLE, api_module.AGENT_AVAILABLE = old_cv, old_agent

    def test_unauthenticated_request_not_recorded(self):
        gateway = FakeUsageGateway()
        client = TestClient(make_app(usage_gateway=gateway))
        assert _post(client, key=None).status_code == 401
        assert gateway.records == []

    def test_no_secret_or_content_in_usage(self):
        gateway = FakeUsageGateway()
        client = TestClient(make_app(usage_gateway=gateway))
        _post(client)
        for record in gateway.records:
            assert record["key_id"] == "11111111-2222-3333-4444-555555555555"
            assert VALID_KEY not in str(record)
            assert "jpg" not in str(record)
            assert "assessment" not in str(record)
            assert "image" not in str(record)


class TestLogRedaction:
    def test_redacting_formatter_masks_keys(self):
        formatter = api_module.RedactingFormatter("%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg='x-api-key: "supersecret-abc" bearer tok1234567890 request_id=1',
            args=(),
            exc_info=None,
        )
        out = formatter.format(record)
        assert "supersecret-abc" not in out
        assert "tok1234567890" not in out
        assert "redacted" in out
