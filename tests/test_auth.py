"""HTTP-level auth tests: 401 / 503 semantics for POST /assess.

Covers Fase 9 items 3, 4, 5 (HTTP layer).
"""

import io

from fastapi.testclient import TestClient

from control_plane import ControlPlaneUnavailable

from .conftest import VALID_KEY, FakeAuthGateway, make_app, make_image


def _post(client, key=VALID_KEY, keys=None, image=None):
    files = {"image": ("car.jpg", io.BytesIO(image or make_image("JPEG")), "image/jpeg")}
    headers = keys if keys is not None else ({"x-api-key": key} if key is not None else {})
    return client.post("/assess", files=files, headers=headers)


class TestMissingAndMalformed:
    def test_missing_key_returns_401(self):
        client = TestClient(make_app())
        response = _post(client, key=None)
        assert response.status_code == 401
        assert response.json() == {"detail": "Missing x-api-key header."}

    def test_invalid_short_key_returns_401(self):
        client = TestClient(make_app())
        assert _post(client, key="short").status_code == 401
        assert _post(client, key="").status_code == 401

    def test_duplicate_x_api_key_returns_401(self):
        client = TestClient(make_app())
        response = _post(
            client,
            keys=[("x-api-key", "aaa-111-bbb"), ("x-api-key", "ccc-222-ddd")],
        )
        assert response.status_code == 401

    def test_get_assess_is_405_not_401(self):
        client = TestClient(make_app())
        response = client.get("/assess", headers={"x-api-key": VALID_KEY})
        assert response.status_code == 405


class TestKeyState:
    def test_route_authoritative_result_does_not_recheck_scope(self):
        gateway = FakeAuthGateway(scopes=("other:scope",))
        client = TestClient(make_app(auth_gateway=gateway))
        response = _post(client)
        assert response.status_code == 200
        assert gateway.attempts == [(VALID_KEY, "POST", "/assess")]

    def test_inactive_key_returns_401(self):
        gateway = FakeAuthGateway(active=False)
        client = TestClient(make_app(auth_gateway=gateway))
        response = _post(client)
        assert response.status_code == 401


class TestControlPlaneFailure:
    def test_control_plane_unavailable_returns_503(self):
        gateway = FakeAuthGateway(raise_error=ControlPlaneUnavailable())
        client = TestClient(make_app(auth_gateway=gateway))
        response = _post(client)
        assert response.status_code == 503

    def test_missing_gateway_returns_503(self):
        client = TestClient(make_app(auth_gateway=None))
        response = _post(client)
        assert response.status_code == 503

    def test_malformed_key_id_returns_503(self):
        gateway = FakeAuthGateway(key_id="not-a-uuid")
        client = TestClient(make_app(auth_gateway=gateway))
        response = _post(client)
        assert response.status_code == 503
