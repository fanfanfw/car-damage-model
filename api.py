"""
Damage.ai — FastAPI Assessment Endpoint
========================================
REST API layer for the Car Damage Assessment system, protected by a managed
``x-api-key`` validated against the Django API Key Management control plane.

Security posture:
- Only ``POST /assess`` requires the managed key (scope ``car-damage:assess``).
- ``/docs``, ``/redoc``, ``/openapi.json``, ``/health*``, ``/policy`` are public.
- Control plane unavailable, credential rejected, or key invalid/wrong-scope
  fail closed: 401 for a bad/inactive key, 503 when the control plane cannot
  authenticate the key. There is no local/legacy key fallback.
- In production, model/decision failures return 503 (no demo result). A demo
  fallback exists only in development and only when ``DEMO_FALLBACK_ENABLED``
  is set explicitly.

Usage:
    uvicorn api:app --host 127.0.0.1 --port 18000
"""

import asyncio
import hashlib
import io
import json
import logging
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.security import APIKeyHeader  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from control_plane import AuthGateway, ControlPlaneUnavailable, FixedWindowRateLimiter, UsageGateway  # noqa: E402
from settings import Settings, load_settings, parse_host  # noqa: E402

# ---------------------------------------------------------------------------
# Logging (redacting formatter as a safety net)
# ---------------------------------------------------------------------------

_SENSITIVE = re.compile(r"(x-api-key|authorization|bearer|service_credential|api[_-]?key)\s*[:=]?\s*[^\s,;]+", re.I)


class RedactingFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        return _SENSITIVE.sub("<redacted>", message)


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("damage_ai_api")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(RedactingFormatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


logger = _setup_logging()

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Optional model/decision modules (availability drives /health/ready)
# ---------------------------------------------------------------------------
try:
    from car_damage_detector import CarDamageDetector

    CV_AVAILABLE = True
    logger.info("CV detection module loaded successfully")
except ImportError:
    CV_AVAILABLE = False
    logger.warning("CV detection module not found")

try:
    from agentic.decision_agent import make_decision
    from agentic.policy_loader import load_policy

    AGENT_AVAILABLE = True
    logger.info("Decision agent loaded successfully")
except ImportError:
    AGENT_AVAILABLE = False
    logger.warning("Decision agent not found")

ASSESS_SCOPE = "car-damage:assess"
ASSESS_TEMPLATE = "/assess"
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
MODEL_CHECKSUM_PATH = MODEL_PATH.with_suffix(".pt.sha256")
CHUNK_SIZE = 64 * 1024
MIN_IMAGE_BYTES = 1000

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}

# ---------------------------------------------------------------------------
# Response models (Pydantic)
# ---------------------------------------------------------------------------


class DamageDetection(BaseModel):
    """Single detected damage instance."""

    damage_type: str = Field(..., description="Type of damage: scratch, dent, crack, shattered_glass, flat_tire, broken_lamp")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    severity: str = Field(..., description="Severity level: minor, moderate, severe")
    location: str = Field("unknown", description="Approximate location on vehicle")
    bbox: list = Field(default_factory=list, description="Bounding box [x1, y1, x2, y2]")
    area_percentage: float = Field(0.0, description="Percentage of image area affected")
    estimated_cost: int = Field(0, description="Estimated repair cost in USD")


class DecisionTrace(BaseModel):
    """Explains why a specific decision was made."""

    rule_applied: str = Field(..., description="Which policy rule triggered the decision")
    threshold: str = Field(..., description="Threshold that was evaluated")
    evidence: str = Field(..., description="SOP or policy evidence supporting the decision")


class AssessmentResult(BaseModel):
    """Full assessment response."""

    assessment_id: str = Field(..., description="Unique assessment identifier")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")

    damages_detected: list[DamageDetection] = Field(default_factory=list)
    total_damages: int = Field(0, description="Number of damages detected")

    decision: str = Field(..., description="AUTO_APPROVE | HUMAN_REVIEW | ESCALATE")
    decision_confidence: float = Field(..., ge=0.0, le=1.0)
    decision_trace: list[DecisionTrace] = Field(default_factory=list)

    model_version: str = Field("demo-v0.1", description="CV model version used")
    policy_version: str = Field("v1.0", description="Policy version applied")
    cv_backend: str = Field("demo", description="CV backend: demo | yolov8 | custom")
    human_review_required: bool = Field(False)


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class UploadTooLarge(Exception):
    pass


class UploadInvalid(Exception):
    pass


class ModelUnavailable(Exception):
    pass


class DecisionUnavailable(Exception):
    pass


# ---------------------------------------------------------------------------
# Demo detection / decision (development-only, explicit flag)
# ---------------------------------------------------------------------------


def demo_detect_damage(image_bytes: bytes) -> list[dict]:
    """Deterministic synthetic detection, used only with DEMO_FALLBACK_ENABLED."""
    import hashlib

    img_hash = hashlib.md5(image_bytes).hexdigest()
    hash_int = int(img_hash[:8], 16)

    damage_types = [
        ("scratch", "minor", "rear_bumper"),
        ("dent", "moderate", "front_door_left"),
        ("crack", "severe", "windshield"),
        ("scratch", "minor", "hood"),
        ("dent", "moderate", "rear_quarter_panel"),
        ("shattered_glass", "severe", "headlight_left"),
    ]

    num_damages = (hash_int % 3) + 1
    results = []
    for i in range(num_damages):
        idx = (hash_int + i * 7) % len(damage_types)
        dtype, severity, location = damage_types[idx]
        confidence = round(0.65 + (((hash_int >> (i * 4)) % 30) / 100), 2)
        confidence = min(confidence, 0.98)
        results.append(
            {
                "damage_type": dtype,
                "confidence": confidence,
                "severity": severity,
                "location": location,
            }
        )
    return results


def demo_make_decision(damages: list[dict]) -> dict:
    """Simple rule-based decision, used only with DEMO_FALLBACK_ENABLED."""
    if not damages:
        return {
            "decision": "AUTO_APPROVE",
            "confidence": 0.95,
            "trace": [{"rule_applied": "no_damage_detected", "threshold": "damages == 0", "evidence": "SOP: No visible damage → auto-approve"}],
            "human_review_required": False,
        }
    severe = [d for d in damages if d.get("severity") == "severe"]
    moderate = [d for d in damages if d.get("severity") == "moderate"]
    if severe:
        return {
            "decision": "ESCALATE",
            "confidence": 0.90,
            "trace": [{"rule_applied": "severe_damage_escalation", "threshold": "severity == severe", "evidence": f"SOP: {len(severe)} severe damage(s) detected → escalate"}],
            "human_review_required": True,
        }
    if len(moderate) >= 2:
        return {
            "decision": "HUMAN_REVIEW",
            "confidence": 0.80,
            "trace": [{"rule_applied": "multiple_moderate_review", "threshold": "moderate_count >= 2", "evidence": f"Policy: {len(moderate)} moderate damages → human review"}],
            "human_review_required": True,
        }
    if moderate:
        return {
            "decision": "HUMAN_REVIEW",
            "confidence": 0.75,
            "trace": [{"rule_applied": "moderate_damage_review", "threshold": "severity == moderate", "evidence": "Policy: Moderate damage detected → human review"}],
            "human_review_required": True,
        }
    return {
        "decision": "AUTO_APPROVE",
        "confidence": 0.85,
        "trace": [{"rule_applied": "minor_only_approve", "threshold": "all(severity == minor)", "evidence": f"Policy: {len(damages)} minor damage(s) only → auto-approve"}],
        "human_review_required": False,
    }


# ---------------------------------------------------------------------------
# Upload validation / hardening
# ---------------------------------------------------------------------------


def _size_bucket(size: int) -> str:
    if size < 64 * 1024:
        return "lt64k"
    if size < 512 * 1024:
        return "lt512k"
    if size < 4 * 1024 * 1024:
        return "lt4m"
    return "ge4m"


async def read_upload_bounded(request: Request, upload: UploadFile, settings: Settings) -> bytes:
    """Read the multipart file with a hard byte cap (never unbounded).

    Content-Length is a cheap early 413; the bounded chunked read still runs
    because Content-Length can be missing or forged.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            declared = 0
        if declared > settings.max_upload_bytes:
            raise UploadTooLarge

    chunks = []
    total = 0
    while True:
        chunk = await upload.read(CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > settings.max_upload_bytes:
            raise UploadTooLarge
        chunks.append(chunk)
    return b"".join(chunks)


def validate_image(data: bytes, settings: Settings) -> None:
    """Validate that the bytes are a real JPEG/PNG/WebP within pixel limits.

    Raises UploadInvalid for corrupt/truncated/spoofed content and for
    decompression-bomb / oversized-pixel images.
    """
    if len(data) < MIN_IMAGE_BYTES:
        raise UploadInvalid("Image file is too small or empty.")
    try:
        from PIL import Image

        original = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
                fmt = image.format
            if fmt not in ALLOWED_PIL_FORMATS:
                raise UploadInvalid("Image file is corrupt or unsupported.")
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
        finally:
            Image.MAX_IMAGE_PIXELS = original
    except UploadInvalid:
        raise
    except Exception:
        raise UploadInvalid("Image file is corrupt or unsupported.") from None
    if width > settings.max_image_dimension or height > settings.max_image_dimension or width * height > settings.max_image_pixels:
        raise UploadInvalid("Image exceeds the allowed pixel dimensions.")


# ---------------------------------------------------------------------------
# Auth / usage middleware
# ---------------------------------------------------------------------------


async def _plain_json(send, status_code: int, body: dict, extra_headers: list | None = None) -> None:
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    headers = [(b"content-type", b"application/json")]
    if extra_headers:
        headers.extend(extra_headers)
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": headers,
        }
    )
    await send({"type": "http.response.body", "body": payload})


def _app_path(scope) -> str:
    path = scope.get("path", "")
    root = scope.get("root_path") or ""
    if root and path.startswith(root):
        path = path[len(root) :]
    return path or "/"


class TrustedHostMiddleware:
    def __init__(self, app, state):
        self.app = app
        self.state = state

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        settings: Settings = self.state.settings
        host_values = [value for name, value in scope["headers"] if name.lower() == b"host"]
        host = None
        if len(host_values) == 1:
            try:
                host = parse_host(host_values[0].decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                host = None
        if host is None or host not in set(settings.allowed_hosts):
            return await _plain_json(send, 400, {"detail": "Invalid host header"})
        return await self.app(scope, receive, send)


class AssessAuthMiddleware:
    def __init__(self, app, state):
        self.app = app
        self.state = state

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = _app_path(scope)
        if scope["method"] != "POST" or path != ASSESS_TEMPLATE:
            return await self.app(scope, receive, send)
        if self.state.settings.auth_mode == "kong":
            values = [value for name, value in scope["headers"] if name.lower() == b"x-consumer-id"]
            anonymous = [value for name, value in scope["headers"] if name.lower() == b"x-anonymous-consumer"]
            if len(values) != 1 or anonymous:
                return await _plain_json(send, 401, {"detail": "Invalid Kong consumer identity."})
            try:
                consumer_id = values[0].decode("ascii")
                parsed_id = uuid.UUID(consumer_id)
            except (UnicodeDecodeError, ValueError, AttributeError, TypeError):
                return await _plain_json(send, 401, {"detail": "Invalid Kong consumer identity."})
            if str(parsed_id) != consumer_id.lower():
                return await _plain_json(send, 401, {"detail": "Invalid Kong consumer identity."})
            scope["api_key_id"] = consumer_id.lower()
            return await self.app(scope, receive, send)

        values = [value for name, value in scope["headers"] if name.lower() == b"x-api-key"]
        if not values:
            return await _plain_json(send, 401, {"detail": "Missing x-api-key header."})
        if len(values) != 1:
            return await _plain_json(send, 401, {"detail": "Invalid API key."})
        try:
            api_key = values[0].decode("ascii")
        except UnicodeDecodeError:
            api_key = ""

        gateway: AuthGateway | None = self.state.auth_gateway
        if gateway is None:
            return await _plain_json(send, 503, {"detail": "Assessment service unavailable."})
        try:
            result = await gateway.authenticate(api_key, "POST", ASSESS_TEMPLATE)
        except ControlPlaneUnavailable:
            return await _plain_json(send, 503, {"detail": "Assessment service unavailable."})
        if not result.active or not result.key_id:
            return await _plain_json(send, 401, {"detail": "Invalid API key."})
        try:
            uuid.UUID(result.key_id)
        except (ValueError, AttributeError, TypeError):
            return await _plain_json(send, 503, {"detail": "Assessment service unavailable."})
        scope["api_key_id"] = result.key_id
        scope["managed_rate_limits"] = result.rate_limits
        return await self.app(scope, receive, send)


class AssessRateLimitMiddleware:
    """Enforces the control-plane per-key ``assess`` rate limit.

    Runs after AssessAuthMiddleware (which sets ``scope["api_key_id"]`` and
    ``scope["managed_rate_limits"]`` from AuthResult.rate_limits) and inside
    UsageRecordingMiddleware so a 429 is recorded as 4xx usage. Only active
    when ``API_KEY_RATE_LIMIT_ENFORCEMENT=true``; policy parsing happens during
    introspection regardless of the kill switch, so a malformed policy still
    fails closed with 503. With enforcement on, an absent or unrelated
    ``assess`` policy also fails closed with 503; with enforcement off the
    request is admitted (migration mode). Health, policy, docs, wrong methods,
    and non-assess routes are never limited.
    """

    def __init__(self, app, state):
        self.app = app
        self.state = state

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = _app_path(scope)
        if scope["method"] != "POST" or path != ASSESS_TEMPLATE:
            return await self.app(scope, receive, send)
        if self.state.settings.auth_mode == "kong" or not self.state.settings.api_key_rate_limit_enforcement:
            return await self.app(scope, receive, send)
        key_id = scope.get("api_key_id")
        rate_limits = scope.get("managed_rate_limits")
        spec = rate_limits.get("assess") if rate_limits else None
        if not key_id:
            return await self.app(scope, receive, send)
        if spec is None:
            logger.warning(
                "rate_limit_event category=policy_missing enforcement=enabled key_id_present=%s",
                bool(key_id),
            )
            return await _plain_json(send, 503, {"detail": "Assessment service unavailable."})
        retry_after = await self.state.rate_limiter.check_and_increment(key_id, spec.requests, spec.window_seconds)
        if retry_after is not None:
            return await _plain_json(
                send,
                429,
                {"detail": "Rate limit exceeded"},
                extra_headers=[(b"retry-after", str(retry_after).encode("ascii"))],
            )
        return await self.app(scope, receive, send)


class UsageRecordingMiddleware:
    """Records /assess usage after the response; never logs sensitive payloads."""

    def __init__(self, app, state):
        self.app = app
        self.state = state

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = _app_path(scope)
        if scope["method"] != "POST" or path != ASSESS_TEMPLATE:
            return await self.app(scope, receive, send)
        started = time.monotonic()
        status_code = None

        async def wrapped_send(message):
            nonlocal status_code
            if message["type"] == "http.response.start" and status_code is None:
                status_code = message["status"]
            await send(message)

        await self.app(scope, receive, wrapped_send)
        key_id = scope.get("api_key_id")
        if status_code is None or not key_id:
            return
        if self.state.settings.auth_mode == "kong":
            return
        gateway: UsageGateway | None = self.state.usage_gateway
        if gateway is None:
            return
        latency_ms = int(round((time.monotonic() - started) * 1000))
        gateway.record(key_id, scope["method"], ASSESS_TEMPLATE, f"{status_code // 100}xx", latency_ms)


class RootPathStripMiddleware:
    """Strip the ASGI root_path prefix from scope['path'] so FastAPI's router
    sees the application-relative path (e.g. /assess not /api/assess).

    When uvicorn is started with ``--root-path /api``, the ASGI scope contains
    root_path="/api" but scope["path"] still holds the full wire path (e.g.
    /api/assess).  The internal middleware already uses _app_path() to compute
    the stripped path, but FastAPI's router reads scope["path"] directly and
    returns404 if it still contains the prefix.  This middleware mutates the
    scope in-place before any downstream middleware or the router sees it.
    """

    def __init__(self, app, **_kwargs):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        root = scope.get("root_path") or ""
        if root and scope.get("path", "").startswith(root):
            scope["path"] = scope["path"][len(root) :] or "/"
        return await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Managed client API key validated against the control plane.",
)


def require_managed_key(request: Request, _: str | None = Security(API_KEY_HEADER)) -> str:
    """OpenAPI-facing dependency; the real check runs in AssessAuthMiddleware."""
    key_id = request.scope.get("api_key_id")
    if not key_id:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return key_id


def _demo_enabled(settings: Settings) -> bool:
    return settings.environment != "production" and settings.demo_fallback_enabled


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = app.state._cd
    if CV_AVAILABLE:
        try:
            from PIL import Image

            production = state.settings.environment == "production"
            if production:
                expected = MODEL_CHECKSUM_PATH.read_text().split()[0]
                with MODEL_PATH.open("rb") as model:
                    if hashlib.file_digest(model, "sha256").hexdigest() != expected:
                        raise ValueError("Checkpoint checksum mismatch")
            detector = CarDamageDetector(str(MODEL_PATH), confidence_threshold=0.25, require_checkpoint=production)
            if state.settings.require_cuda and detector.device != "cuda":
                raise RuntimeError("CUDA device unavailable")
            detector.detect_damage(Image.new("RGB", (64, 64)))
            state.detector = detector
            logger.info("assessment model warmup passed (device=%s)", detector.device)
        except Exception as error:
            logger.error("assessment model startup failed (error=%s)", type(error).__name__)
    if AGENT_AVAILABLE:
        try:
            state.policy_ready = bool(load_policy(PROJECT_ROOT / "policies").rules)
        except Exception as error:
            logger.error("assessment policy startup failed (error=%s)", type(error).__name__)
    if state.settings.auth_mode == "control-plane":
        if state.auth_gateway is None:
            state.auth_gateway = AuthGateway(state.settings)
        if state.usage_gateway is None:
            state.usage_gateway = UsageGateway(state.settings)
        await state.usage_gateway.start()
    logger.info("car-damage api started (environment=%s auth_mode=%s)", state.settings.environment, state.settings.auth_mode)
    try:
        yield
    finally:
        state.detector = None
        state.policy_ready = False
        if state.settings.auth_mode == "control-plane":
            await state.usage_gateway.stop()
            state.auth_gateway = None
            state.usage_gateway = None


async def _run_detection(image_bytes: bytes, state, assessment_id: str, request_id: str) -> tuple[list[dict], str]:
    """Run CV detection; returns (detections, cv_backend)."""
    settings = state.settings
    if state.detector is not None:
        try:
            from PIL import Image as PILImage

            with PILImage.open(io.BytesIO(image_bytes)) as pil_img:
                result = state.detector.detect_damage(pil_img)
            raw = result.get("damages", [])
            detections = []
            for det in raw:
                detections.append(
                    {
                        "damage_type": det.get("type", det.get("damage_type", "unknown")),
                        "confidence": det.get("confidence", 0.5),
                        "severity": det.get("severity", "minor"),
                        "location": det.get("location", "unknown"),
                        "bbox": det.get("bbox", []),
                        "area_percentage": det.get("area_percentage", 0),
                        "estimated_cost": det.get("estimated_cost", 0),
                    }
                )
            return detections, "model-backed"
        except Exception:
            logger.warning(
                "assess_event request_id=%s assessment_id=%s category=detection_failure",
                request_id,
                assessment_id,
            )
            if _demo_enabled(settings):
                return demo_detect_damage(image_bytes), "demo-fallback"
            raise ModelUnavailable from None
    if _demo_enabled(settings):
        return demo_detect_damage(image_bytes), "demo"
    logger.warning(
        "assess_event request_id=%s assessment_id=%s category=model_unavailable",
        request_id,
        assessment_id,
    )
    raise ModelUnavailable


async def _run_decision(damages: list[dict], settings: Settings, assessment_id: str, request_id: str) -> tuple[dict, str]:
    """Run the decision agent; returns (decision_result, policy_version)."""
    if AGENT_AVAILABLE:
        try:
            return make_decision(damages), "v1.0-production"
        except Exception:
            logger.warning(
                "assess_event request_id=%s assessment_id=%s category=decision_failure",
                request_id,
                assessment_id,
            )
            if _demo_enabled(settings):
                return demo_make_decision(damages), "v1.0-demo-fallback"
            raise DecisionUnavailable from None
    if _demo_enabled(settings):
        return demo_make_decision(damages), "v1.0-demo"
    logger.warning(
        "assess_event request_id=%s assessment_id=%s category=decision_unavailable",
        request_id,
        assessment_id,
    )
    raise DecisionUnavailable


def create_app(*, settings: Settings | None = None, auth_gateway=None, usage_gateway=None, rate_limiter=None) -> FastAPI:
    settings = settings if settings is not None else load_settings()

    application = FastAPI(
        title="Damage.ai — Vehicle Damage Assessment API",
        version="0.2.0",
        docs_url="/docs" if settings.api_docs_enabled else None,
        redoc_url="/redoc" if settings.api_docs_enabled else None,
        openapi_url="/openapi.json" if settings.api_docs_enabled else None,
        lifespan=lifespan,
    )

    class _State:
        pass

    state = _State()
    state.settings = settings
    state.auth_gateway = auth_gateway
    state.usage_gateway = usage_gateway
    state.detector = None
    state.policy_ready = False
    state.rate_limiter = rate_limiter or FixedWindowRateLimiter()
    state.assess_semaphore = asyncio.Semaphore(settings.max_concurrent_assessments)
    application.state._cd = state

    application.add_middleware(AssessRateLimitMiddleware, state=state)
    application.add_middleware(UsageRecordingMiddleware, state=state)
    application.add_middleware(AssessAuthMiddleware, state=state)
    application.add_middleware(TrustedHostMiddleware, state=state)
    if settings.cors_allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_allowed_origins),
            allow_credentials=False,
            allow_methods=["POST", "GET", "OPTIONS"],
            allow_headers=["x-api-key", "content-type"],
        )
    application.add_middleware(RootPathStripMiddleware)

    _register_routes(application)
    return application


def _register_routes(application: FastAPI) -> None:
    @application.get("/health/live", tags=["System"])
    async def health_live():
        """Process liveness. Always 200 when the process is up."""
        return {"status": "ok"}

    @application.get("/health/ready", tags=["System"])
    async def health_ready():
        """Readiness: model and decision policy must be available."""
        state = application.state._cd
        if state.detector is not None and state.policy_ready:
            return {"status": "ready"}
        return JSONResponse(status_code=503, content={"status": "not_ready"})

    @application.get("/health", tags=["System"], deprecated=True)
    async def health_legacy():
        """Deprecated alias for /health/live."""
        return {"status": "ok"}

    @application.get("/policy", tags=["Policy"])
    async def get_policy():
        """Returns the current policy configuration (public)."""
        policy_path = PROJECT_ROOT / "policies"
        policy_names = []
        if policy_path.exists():
            for f in list(policy_path.glob("*.yaml")) + list(policy_path.glob("*.yml")):
                policy_names.append(f.name)
        return {
            "policy_version": "v1.0",
            "policy_files": policy_names,
            "decision_types": ["AUTO_APPROVE", "HUMAN_REVIEW", "ESCALATE"],
            "severity_levels": ["minor", "moderate", "severe"],
            "rules_summary": {
                "no_damage": "AUTO_APPROVE",
                "minor_only": "AUTO_APPROVE",
                "any_moderate": "HUMAN_REVIEW",
                "multiple_moderate": "HUMAN_REVIEW",
                "any_severe": "ESCALATE",
            },
        }

    @application.post(
        "/assess",
        response_model=AssessmentResult,
        tags=["Assessment"],
        operation_id="assess_damage",
        openapi_extra={
            "x-api-scopes": ["car-damage:assess"],
            "x-rate-limit-category": "assess",
            "x-usage-enabled": True,
        },
        responses={
            401: {
                "description": ("Missing X-API-Key header, invalid key, inactive key, or valid key missing the required scope."),
            },
            413: {"description": "Upload exceeds the configured byte limit."},
            429: {"description": "Per-key rate limit exceeded for the assess category."},
            503: {"description": "Assessment or control plane unavailable."},
        },
    )
    async def assess_damage(
        request: Request,
        image: UploadFile = File(..., description="Vehicle image (JPEG/PNG/WebP)"),
        _: str = Depends(require_managed_key),
    ):
        """Upload a vehicle image → structured damage assessment.

        Requires a managed `X-API-Key`.
        """
        state = application.state._cd
        settings: Settings = state.settings
        request_id = str(uuid.uuid4())
        assessment_id = str(uuid.uuid4())[:12]
        start_time = time.monotonic()
        size_bucket = None

        try:
            # Concurrency gate BEFORE reading the body so in-flight authenticated
            # requests (and their ~20MB buffers) stay bounded by the semaphore.
            try:
                await asyncio.wait_for(state.assess_semaphore.acquire(), timeout=0.01)
            except TimeoutError:
                logger.warning(
                    "assess_event request_id=%s assessment_id=%s category=concurrency_limit",
                    request_id,
                    assessment_id,
                )
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Assessment service unavailable."},
                    headers={"Retry-After": "5"},
                )
            try:
                if image.content_type not in ALLOWED_MIME:
                    raise UploadInvalid("Unsupported image type. Use JPEG, PNG, or WebP.")

                image_bytes = await read_upload_bounded(request, image, settings)
                validate_image(image_bytes, settings)
                size_bucket = _size_bucket(len(image_bytes))

                damages, cv_backend = await _run_detection(image_bytes, state, assessment_id, request_id)
                decision_result, policy_version = await _run_decision(damages, settings, assessment_id, request_id)
            finally:
                state.assess_semaphore.release()
        except UploadTooLarge:
            logger.warning(
                "assess_event request_id=%s assessment_id=%s category=upload_oversized size_bucket=%s",
                request_id,
                assessment_id,
                size_bucket,
            )
            return JSONResponse(status_code=413, content={"detail": "Image file exceeds the allowed size."})
        except UploadInvalid as error:
            logger.warning(
                "assess_event request_id=%s assessment_id=%s category=upload_invalid",
                request_id,
                assessment_id,
            )
            return JSONResponse(status_code=400, content={"detail": str(error)})
        except ModelUnavailable:
            logger.warning(
                "assess_event request_id=%s assessment_id=%s category=model_unavailable",
                request_id,
                assessment_id,
            )
            return JSONResponse(status_code=503, content={"detail": "Assessment service unavailable."})
        except DecisionUnavailable:
            logger.warning(
                "assess_event request_id=%s assessment_id=%s category=decision_unavailable",
                request_id,
                assessment_id,
            )
            return JSONResponse(status_code=503, content={"detail": "Assessment service unavailable."})
        finally:
            await image.close()

        # Mirror the Streamlit UI display normalization (title case) so the
        # response content matches what the UI renders for the same image.
        for d in damages:
            if d.get("severity"):
                d["severity"] = d["severity"].capitalize()
            if d.get("damage_type"):
                d["damage_type"] = d["damage_type"].replace("_", " ").title()

        normalized = [
            DamageDetection(
                damage_type=d.get("damage_type", "unknown"),
                confidence=d.get("confidence", 0.5),
                severity=d.get("severity", "minor"),
                location=d.get("location", "unknown"),
                bbox=d.get("bbox", []),
                area_percentage=d.get("area_percentage", 0.0),
                estimated_cost=d.get("estimated_cost", 0),
            )
            for d in damages
        ]

        traces = [
            DecisionTrace(
                rule_applied=t.get("rule_applied", "unknown"),
                threshold=t.get("threshold", ""),
                evidence=t.get("evidence", ""),
            )
            for t in decision_result.get("trace", [])
        ]

        processing_time = int((time.monotonic() - start_time) * 1000)
        logger.info(
            "assess_event request_id=%s assessment_id=%s status_class=2xx size_bucket=%s latency_ms=%d damages=%d cv_backend=%s",
            request_id,
            assessment_id,
            size_bucket,
            processing_time,
            len(normalized),
            cv_backend,
        )

        return AssessmentResult(
            assessment_id=assessment_id,
            timestamp=datetime.now(UTC).isoformat(),
            processing_time_ms=processing_time,
            damages_detected=normalized,
            total_damages=len(normalized),
            decision=decision_result["decision"],
            decision_confidence=decision_result.get("confidence", 0.5),
            decision_trace=traces,
            model_version="demo-v0.1" if cv_backend == "demo" else "yolov8-v0.1",
            policy_version=policy_version,
            cv_backend=cv_backend,
            human_review_required=decision_result.get("human_review_required", False),
        )


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=18000, log_level="info")
