"""Shared test fixtures: synthetic images, fake gateways, app factory."""

import io

import pytest

from api import ASSESS_SCOPE
from control_plane import AuthResult
from settings import load_settings

KEY_ID = "11111111-2222-3333-4444-555555555555"
VALID_KEY = "managed-key-abcdefghijklmnop"  # synthetic, >= 8 ascii


def make_settings(**overrides):
    base = {
        "ENVIRONMENT": "development",
        "AUTH_MODE": "control-plane",
        "DEMO_FALLBACK_ENABLED": "true",
        "ALLOWED_HOSTS": "testserver,localhost,127.0.0.1",
        "CONTROL_PLANE_BASE_URL": "http://100.124.216.27:18080",
        "SERVICE_CREDENTIAL": "s" * 40,
        "AUTH_CACHE_MAX_TTL_SECONDS": "30",
    }
    for key, value in overrides.items():
        if value is None:
            base.pop(key, None)
        else:
            base[key] = value
    return load_settings(base)


def make_image(fmt="JPEG", size=(64, 64), color=(120, 120, 120)):
    """Return a valid synthetic image's bytes (JPEG|PNG|WEBP), padded past the
    API's minimum accepted size."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    data = buf.getvalue()
    if len(data) < 1300:
        data += b"\x00" * (1300 - len(data))
    return data


def make_png_header(width, height):
    """Craft a PNG whose header declares width x height but carries no pixels.

    Used to test pixel/dimension limits without allocating huge buffers.
    """
    import struct
    import zlib

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF)
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + ihdr_crc
    raw = b"\x00" + b"\x00\x00\x00"  # one 1x1 pixel row
    compressed = zlib.compress(raw)
    idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF)
    idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + idat_crc
    iend_crc = struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    iend = struct.pack(">I", 0) + b"IEND" + iend_crc
    padding = b"a" * 1200
    pad_crc = struct.pack(">I", zlib.crc32(b"tEXt" + padding) & 0xFFFFFFFF)
    pad_chunk = struct.pack(">I", len(padding)) + b"tEXt" + padding + pad_crc
    return signature + ihdr + idat + pad_chunk + iend


class FakeAuthGateway:
    def __init__(
        self,
        active=True,
        key_id=KEY_ID,
        scopes=(ASSESS_SCOPE,),
        raise_error=None,
        rate_limits=None,
    ):
        self.active = active
        self.key_id = key_id
        self.scopes = scopes
        self.raise_error = raise_error
        self.rate_limits = rate_limits
        self.attempts = []

    async def authenticate(self, api_key, method, endpoint_template):
        self.attempts.append((api_key, method, endpoint_template))
        if self.raise_error is not None:
            raise self.raise_error
        if (
            not api_key
            or not isinstance(api_key, str)
            or not api_key.isascii()
            or not 8 <= len(api_key) <= 512
        ):
            return AuthResult(active=False)
        if self.active:
            return AuthResult(active=True, key_id=self.key_id, scopes=self.scopes, rate_limits=self.rate_limits)
        return AuthResult(active=False)


class FakeUsageGateway:
    def __init__(self):
        self.records = []
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    def record(self, key_id, method, endpoint_template, status_class, latency_ms):
        self.records.append(
            {
                "key_id": key_id,
                "method": method,
                "endpoint_template": endpoint_template,
                "status_class": status_class,
                "latency_ms": latency_ms,
            }
        )


_DEFAULT = object()


def make_app(settings=None, auth_gateway=_DEFAULT, usage_gateway=None, rate_limiter=None):
    from api import create_app

    if auth_gateway is _DEFAULT:
        auth_gateway = FakeAuthGateway()
    return create_app(
        settings=settings or make_settings(),
        auth_gateway=auth_gateway,
        usage_gateway=usage_gateway or FakeUsageGateway(),
        rate_limiter=rate_limiter,
    )


@pytest.fixture
def app():
    return make_app()


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)
