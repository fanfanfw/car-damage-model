"""Smoke the packaged model against production auth and multipart routes."""
import io
import os

from fastapi.testclient import TestClient
from PIL import Image

os.environ.update(ENVIRONMENT="production", AUTH_MODE="kong", ALLOWED_HOSTS="testserver")

from api import app  # noqa: E402

with TestClient(app, root_path="/api") as client:
    expected_ready = os.environ.get("EXPECT_CHECKPOINT", "ready") == "ready"
    assert client.get("/api/health/live").status_code == 200
    assert client.get("/api/health/ready").status_code == (200 if expected_ready else 503)
    for color in ("red", "blue"):
        image = io.BytesIO()
        Image.new("RGB", (256, 256), color).save(image, format="JPEG")
        response = client.post(
            "/api/assess", files={"image": ("synthetic.jpg", image.getvalue(), "image/jpeg")},
            headers={"X-Consumer-ID": "11111111-2222-3333-4444-555555555555"},
        )
        assert response.status_code == (200 if expected_ready else 503), response.status_code
        if expected_ready:
            assert response.json()["cv_backend"] == "model-backed"
    detector = app.state._cd.detector
    if expected_ready and os.environ.get("CAR_DAMAGE_REQUIRE_CUDA") == "true":
        assert detector.model.predictor.device.type == "cuda"
    print(f"readiness and two multipart assessments passed (device={detector.device if detector else 'unavailable'})")
