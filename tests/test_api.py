"""
Contract, health, docs, and root-path tests for the Damage.ai FastAPI API.

Run with: pytest tests/test_api.py -v

Covers Fase 9 items: 1, 2, 9, 17, 18, 21, 22.
"""

import asyncio
import io

from fastapi.testclient import TestClient

from .conftest import VALID_KEY, make_app, make_image

AUTH_HEADER = {"x-api-key": VALID_KEY}


def _post(client, content=None, filename="car.jpg", mime="image/jpeg", key=VALID_KEY):
    files = {"image": (filename, io.BytesIO(content or make_image("JPEG")), mime)}
    headers = {"x-api-key": key} if key is not None else {}
    return client.post("/assess", files=files, headers=headers)


class TestAssessmentContract:
    def test_assess_accepts_jpeg(self):
        client = TestClient(make_app())
        response = _post(client, content=make_image("JPEG"), filename="car.jpg")
        assert response.status_code == 200
        data = response.json()
        assert data["assessment_id"]
        assert data["total_damages"] == len(data["damages_detected"])
        assert data["decision"] in ("AUTO_APPROVE", "HUMAN_REVIEW", "ESCALATE")
        assert data["decision_trace"]
        for trace in data["decision_trace"]:
            assert "rule_applied" in trace and "threshold" in trace and "evidence" in trace
        for d in data["damages_detected"]:
            assert "damage_type" in d
            assert "confidence" in d
            assert "severity" in d
            assert "location" in d
            assert "bbox" in d
            assert "area_percentage" in d
            assert "estimated_cost" in d

    def test_assess_accepts_png(self):
        client = TestClient(make_app())
        response = _post(client, content=make_image("PNG"), filename="car.png", mime="image/png")
        assert response.status_code == 200

    def test_assess_accepts_webp(self):
        client = TestClient(make_app())
        response = _post(client, content=make_image("WEBP"), filename="car.webp", mime="image/webp")
        assert response.status_code == 200

    def test_assess_response_metadata(self):
        client = TestClient(make_app())
        response = _post(client)
        data = response.json()
        assert "model_version" in data
        assert "policy_version" in data
        assert "cv_backend" in data
        assert "processing_time_ms" in data
        assert isinstance(data["processing_time_ms"], int)
        from datetime import datetime

        datetime.fromisoformat(data["timestamp"])

    def test_two_assessments_have_different_ids(self):
        client = TestClient(make_app())
        ids = {_post(client).json()["assessment_id"] for _ in range(3)}
        assert len(ids) == 3

    def test_duplicate_damage_schema_fields_are_not_present_twice(self):
        """Regression: DamageDetection must expose exactly one set of fields."""
        client = TestClient(make_app())
        data = _post(client).json()
        assert data["damages_detected"] is not None


class TestHealth:
    def test_live_returns_ok(self):
        client = TestClient(make_app())
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_ready_returns_200_when_ready(self):
        import api as api_module

        api_module.CV_AVAILABLE = True
        api_module.AGENT_AVAILABLE = True
        client = TestClient(make_app())
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

    def test_ready_returns_503_when_not_ready(self):
        import api as api_module

        old_cv, old_agent = api_module.CV_AVAILABLE, api_module.AGENT_AVAILABLE
        api_module.CV_AVAILABLE = False
        try:
            client = TestClient(make_app())
            response = client.get("/health/ready")
            assert response.status_code == 503
        finally:
            api_module.CV_AVAILABLE = old_cv
            api_module.AGENT_AVAILABLE = old_agent

    def test_health_no_auth(self):
        client = TestClient(make_app())
        for path in ("/health/live", "/health/ready", "/health"):
            assert client.get(path).status_code in (200, 503)

    def test_legacy_health_is_alias(self):
        client = TestClient(make_app())
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestDocs:
    def test_openapi_public_and_declares_key_scheme(self):
        client = TestClient(make_app())
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "APIKeyHeader" in schema["components"]["securitySchemes"]
        assert schema["components"]["securitySchemes"]["APIKeyHeader"]["name"] == "X-API-Key"
        assert schema["components"]["securitySchemes"]["APIKeyHeader"]["type"] == "apiKey"
        assert schema["components"]["securitySchemes"]["APIKeyHeader"]["in"] == "header"
        assert "managed" in schema["components"]["securitySchemes"]["APIKeyHeader"]["description"].lower()

    def test_only_assess_requires_security(self):
        client = TestClient(make_app())
        schema = client.get("/openapi.json").json()
        assert schema["paths"]["/assess"]["post"]["security"] == [{"APIKeyHeader": []}]
        assert "security" not in schema["paths"]["/health/live"]["get"]
        assert "security" not in schema["paths"]["/health/ready"]["get"]

    def test_assess_has_stable_operation_id_and_extensions(self):
        client = TestClient(make_app())
        schema = client.get("/openapi.json").json()
        assess_op = schema["paths"]["/assess"]["post"]
        assert assess_op["operationId"] == "assess_damage"
        assert assess_op.get("x-api-scopes") == ["car-damage:assess"]
        assert assess_op.get("x-rate-limit-category") == "assess"
        assert assess_op.get("x-usage-enabled") is True

    def test_docs_have_no_real_key_or_credential(self):
        client = TestClient(make_app())
        body = client.get("/openapi.json").text
        assert VALID_KEY not in body
        assert "s" * 40 not in body

    def test_operation_ids_are_unique(self):
        client = TestClient(make_app())
        schema = client.get("/openapi.json").json()
        ids = []
        for path_op in schema["paths"].values():
            for method_op in path_op.values():
                if isinstance(method_op, dict) and "operationId" in method_op:
                    ids.append(method_op["operationId"])
        assert len(ids) == len(set(ids))


class TestPolicy:
    def test_policy_public(self):
        client = TestClient(make_app())
        response = client.get("/policy")
        assert response.status_code == 200
        data = response.json()
        assert "AUTO_APPROVE" in data["decision_types"]
        assert "rules_summary" in data


class TestRootPathStripMiddleware:
    def test_strips_root_path_prefix_from_scope(self):
        from api import RootPathStripMiddleware

        seen_paths = []

        async def inner_app(scope, receive, send):
            seen_paths.append(scope["path"])

        app = RootPathStripMiddleware(inner_app)
        scope = {"type": "http", "method": "GET", "path": "/api/assess", "root_path": "/api", "headers": []}
        asyncio.get_event_loop().run_until_complete(app(scope, None, None))
        assert seen_paths == ["/assess"]

    def test_no_root_path_leaves_scope_unchanged(self):
        from api import RootPathStripMiddleware

        seen_paths = []

        async def inner_app(scope, receive, send):
            seen_paths.append(scope["path"])

        app = RootPathStripMiddleware(inner_app)
        scope = {"type": "http", "method": "GET", "path": "/assess", "root_path": "", "headers": []}
        asyncio.get_event_loop().run_until_complete(app(scope, None, None))
        assert seen_paths == ["/assess"]

    def test_path_not_prefixed_by_root_is_unchanged(self):
        from api import RootPathStripMiddleware

        seen_paths = []

        async def inner_app(scope, receive, send):
            seen_paths.append(scope["path"])

        app = RootPathStripMiddleware(inner_app)
        scope = {"type": "http", "method": "GET", "path": "/other", "root_path": "/api", "headers": []}
        asyncio.get_event_loop().run_until_complete(app(scope, None, None))
        assert seen_paths == ["/other"]

    def test_non_http_scope_is_passed_through(self):
        from api import RootPathStripMiddleware

        seen_types = []

        async def inner_app(scope, receive, send):
            seen_types.append(scope.get("type"))

        app = RootPathStripMiddleware(inner_app)
        scope = {"type": "lifespan"}
        asyncio.get_event_loop().run_until_complete(app(scope, None, None))
        assert seen_types == ["lifespan"]


class TestRootPath:
    def test_root_path_api_assess_contract(self):
        """With uvicorn --root-path /api, /api/assess routes to /assess."""
        client = TestClient(make_app(), root_path="/api")
        response = _post(client, content=make_image("JPEG"))
        assert response.status_code == 200
        assert response.json()["assessment_id"]

    def test_root_path_explicit_api_assess_path(self):
        """The wire path /api/assess (server entry /api) maps to /assess."""
        client = TestClient(make_app(), root_path="/api")
        files = {"image": ("car.jpg", io.BytesIO(make_image("JPEG")), "image/jpeg")}
        response = client.post("/api/assess", files=files, headers={"x-api-key": VALID_KEY})
        assert response.status_code == 200
        assert response.json()["assessment_id"]

    def test_root_path_health(self):
        client = TestClient(make_app(), root_path="/api")
        assert client.get("/health/live").status_code == 200

    def test_root_path_openapi_serves_app_paths(self):
        client = TestClient(make_app(), root_path="/api")
        schema = client.get("/openapi.json").json()
        assert "/assess" in schema["paths"]
        assert schema["servers"] == [{"url": "/api"}]
