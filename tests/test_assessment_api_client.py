from unittest.mock import Mock

import pytest
import requests

from assessment_api_client import AssessmentAPIError, assess_image


def test_assess_image_sends_managed_key(monkeypatch):
    monkeypatch.setenv("CAR_DAMAGE_API_URL", "http://api:18000")
    monkeypatch.setenv("CAR_DAMAGE_API_KEY", "managed-key")
    response = Mock(status_code=200)
    response.json.return_value = {"assessment_id": "test"}
    session = Mock()
    session.post.return_value = response

    result = assess_image("car.jpg", "image/jpeg", b"image", session=session)

    assert result == {"assessment_id": "test"}
    session.post.assert_called_once_with(
        "http://api:18000/assess",
        headers={"x-api-key": "managed-key"},
        files={"image": ("car.jpg", b"image", "image/jpeg")},
        timeout=60.0,
    )


def test_assess_image_requires_key(monkeypatch):
    monkeypatch.delenv("CAR_DAMAGE_API_KEY", raising=False)
    with pytest.raises(AssessmentAPIError, match="not configured"):
        assess_image("car.jpg", "image/jpeg", b"image")


def test_assess_image_reports_rate_limit(monkeypatch):
    monkeypatch.setenv("CAR_DAMAGE_API_KEY", "managed-key")
    response = Mock(status_code=429, headers={"Retry-After": "42"})
    session = Mock()
    session.post.return_value = response
    with pytest.raises(AssessmentAPIError, match="42 seconds"):
        assess_image("car.jpg", "image/jpeg", b"image", session=session)


def test_assess_image_reports_network_failure(monkeypatch):
    monkeypatch.setenv("CAR_DAMAGE_API_KEY", "managed-key")
    session = Mock()
    session.post.side_effect = requests.ConnectionError()
    with pytest.raises(AssessmentAPIError, match="unreachable"):
        assess_image("car.jpg", "image/jpeg", b"image", session=session)
