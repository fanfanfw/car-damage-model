"""Assertions on the Car Damage deployment artifacts (preparation, no deploy).

Covers Fase 10 matrix evidence: the nginx example admits only server-111 on
the Tailscale listener, proxies to loopback, and does NOT allow 127.0.0.1
(the local cloudflared tunnel reaches Uvicorn directly on 127.0.0.1:18000).
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class TestNginxArtifact:
    NGINX = REPO / "deploy" / "car-damage-api.nginx.example.conf"

    def test_listen_is_tailscale_ip_only(self):
        text = self.NGINX.read_text()
        assert "listen 100.124.216.27:18000;" in text
        assert "listen 0.0.0.0" not in text
        assert "listen 127.0.0.1" not in text

    def test_allow_only_server_111(self):
        text = self.NGINX.read_text()
        assert "allow 100.76.212.120;" in text
        assert "deny all;" in text
        assert "allow 127.0.0.1;" not in text
        assert "allow 100.124.216.27;" not in text

    def test_proxy_to_loopback(self):
        text = self.NGINX.read_text()
        assert "proxy_pass http://127.0.0.1:18000;" in text

    def test_client_max_body_size_above_upload_limit(self):
        text = self.NGINX.read_text()
        assert "client_max_body_size 21m;" in text


class TestEnvExample:
    ENV = REPO / ".env.example"

    def test_control_plane_base_url_uses_loopback_internal_listener(self):
        text = self.ENV.read_text()
        assert "CONTROL_PLANE_BASE_URL=http://127.0.0.1:18083" in text

    def test_production_defaults(self):
        text = self.ENV.read_text()
        assert "ENVIRONMENT=production" in text
        assert "DEMO_FALLBACK_ENABLED=false" in text
        assert "AUTH_CACHE_MAX_TTL_SECONDS=30" in text
        assert "SERVICE_CREDENTIAL=" in text
