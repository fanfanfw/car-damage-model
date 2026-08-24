"""Environment configuration for the Car Damage Assessment API.

All values come from the environment (or an optional .env file loaded by
the FastAPI app). Production validation is strict: explicit control plane,
credential, hosts, and no demo fallback.
"""

from __future__ import annotations

import ipaddress
import os
import re
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_HOST_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")
_CONTROL_PLANE_PEER = ipaddress.ip_network("100.64.0.0/10")

DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 24_000_000
DEFAULT_MAX_IMAGE_DIMENSION = 12_000


def _integer(values: Mapping[str, str], name: str, default: int, minimum: int, maximum: int | None = None) -> int:
    raw = values.get(name)
    if raw is None:
        return default
    text = raw.strip()
    if not re.fullmatch(r"-?[0-9]+", text):
        raise ValueError(f"{name} must be an integer")
    value = int(text)
    if value < minimum or maximum is not None and value > maximum:
        requirement = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        raise ValueError(f"{name} must be {requirement}")
    return value


def _boolean(values: Mapping[str, str], name: str, default: bool) -> bool:
    raw = values.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must use a documented boolean value")


def parse_host(value: str) -> str:
    if not value or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("invalid host")
    if value.startswith("["):
        if value.count("[") != 1 or value.count("]") != 1:
            raise ValueError("invalid host")
        end = value.index("]")
        host, suffix = value[1:end], value[end + 1 :]
        if "%" in host:
            raise ValueError("invalid host")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            raise ValueError("invalid host") from None
        if address.version != 6:
            raise ValueError("invalid host")
        normalized = str(address)
    else:
        if "[" in value or "]" in value or value.count(":") > 1:
            raise ValueError("invalid host")
        host, separator, port = value.partition(":")
        suffix = f":{port}" if separator else ""
        if host.endswith("."):
            raise ValueError("invalid host")
        try:
            normalized = str(ipaddress.ip_address(host))
        except ValueError:
            if len(host) > 253 or any(not _HOST_LABEL.fullmatch(label) for label in host.split(".")):
                raise ValueError("invalid host") from None
            normalized = host.lower()
    if suffix and (not suffix.startswith(":") or not suffix[1:].isdigit() or not 1 <= int(suffix[1:]) <= 65535):
        raise ValueError("invalid host")
    return normalized


def _allowed_hosts(values: Mapping[str, str], production: bool) -> tuple[str, ...]:
    default = "" if production else "localhost,127.0.0.1,testserver"
    hosts = tuple(host.strip() for host in values.get("ALLOWED_HOSTS", default).split(","))
    if not hosts or not all(hosts):
        raise ValueError("ALLOWED_HOSTS must not be empty or contain empty entries")
    if production and any("*" in host for host in hosts):
        raise ValueError("ALLOWED_HOSTS must be explicit in production")
    for host in hosts:
        parse_host(host)
    return hosts


def _cors_origins(values: Mapping[str, str], production: bool) -> tuple[str, ...]:
    raw = values.get("CORS_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return ()
    if raw == "*":
        if production:
            raise ValueError("CORS_ALLOWED_ORIGINS must be explicit in production")
        return ("*",)
    origins = tuple(origin.strip() for origin in raw.split(","))
    if not all(origins):
        raise ValueError("CORS_ALLOWED_ORIGINS must not contain empty entries")
    for origin in origins:
        parsed = urllib.parse.urlsplit(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("CORS_ALLOWED_ORIGINS entries must be http(s) origins")
    return origins


def _control_plane_url(value: str, production: bool) -> str:
    value = value.strip()
    if not value:
        if production:
            raise ValueError("CONTROL_PLANE_BASE_URL is required in production")
        return ""
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
    ):
        raise ValueError("CONTROL_PLANE_BASE_URL must be an http(s) base URL without a path or userinfo")
    if production and parsed.scheme == "http":
        try:
            host_ip = ipaddress.ip_address(parsed.hostname)
        except ValueError:
            raise ValueError(
                "CONTROL_PLANE_BASE_URL over http in production must be a literal "
                "Tailscale peer (100.64.0.0/10) or loopback 127.0.0.1:18083"
            ) from None
        if host_ip in _CONTROL_PLANE_PEER:
            return value.rstrip("/")
        if host_ip.is_loopback:
            if parsed.hostname != "127.0.0.1" or parsed.port != 18083:
                raise ValueError(
                    "CONTROL_PLANE_BASE_URL over loopback http in production must be "
                    "exactly http://127.0.0.1:18083 (the internal-only listener); "
                    "it must not point at gunicorn 18080 or any other localhost port"
                )
            return value.rstrip("/")
        raise ValueError(
            "CONTROL_PLANE_BASE_URL over http in production must be a Tailscale "
            "peer (100.64.0.0/10) or loopback 127.0.0.1:18083; use https for any "
            "other host"
        )
    return value.rstrip("/")


def _forwarded_allow_ips(values: Mapping[str, str], production: bool) -> str:
    raw = values.get("FORWARDED_ALLOW_IPS", "127.0.0.1,::1").strip()
    if not raw:
        raise ValueError("FORWARDED_ALLOW_IPS must not be empty")
    entries = tuple(entry.strip() for entry in raw.split(","))
    if not entries or production and "*" in entries:
        raise ValueError("FORWARDED_ALLOW_IPS must contain explicit peers in production")
    try:
        for entry in entries:
            if entry != "*":
                network = ipaddress.ip_network(entry, strict=False)
                if production and (not network.is_loopback or network.num_addresses > 1):
                    raise ValueError("FORWARDED_ALLOW_IPS must be a single loopback peer in production")
    except ValueError:
        raise ValueError("FORWARDED_ALLOW_IPS contains an invalid peer network") from None
    return ",".join(entries)


@dataclass(frozen=True)
class Settings:
    environment: str
    service_id: str
    control_plane_base_url: str
    service_credential: str
    introspection_timeout_seconds: int
    auth_max_introspections_per_minute: int
    auth_cache_max_entries: int
    auth_cache_max_ttl_seconds: int
    usage_flush_interval_seconds: int
    usage_retry_attempts: int
    usage_max_queue_batches: int
    usage_max_records_per_batch: int
    usage_max_queued_records: int
    allowed_hosts: tuple[str, ...]
    cors_allowed_origins: tuple[str, ...]
    api_docs_enabled: bool
    forwarded_allow_ips: str
    max_upload_bytes: int
    max_image_pixels: int
    max_image_dimension: int
    max_concurrent_assessments: int
    demo_fallback_enabled: bool
    api_key_rate_limit_enforcement: bool


def load_settings(values: Mapping[str, str] | None = None) -> Settings:
    source = values if values is not None else os.environ
    environment = source.get("ENVIRONMENT", "development").strip()
    if environment not in {"development", "production"}:
        raise ValueError("ENVIRONMENT must be development or production")
    production = environment == "production"

    service_credential = source.get("SERVICE_CREDENTIAL", "").strip()
    if production and not (
        service_credential
        and service_credential.isascii()
        and 32 <= len(service_credential) <= 512
    ):
        raise ValueError("SERVICE_CREDENTIAL is required in production")

    control_plane_base_url = _control_plane_url(
        source.get("CONTROL_PLANE_BASE_URL", ""), production
    )
    demo_fallback_enabled = _boolean(source, "DEMO_FALLBACK_ENABLED", False)
    if production and demo_fallback_enabled:
        raise ValueError("DEMO_FALLBACK_ENABLED must be false in production")

    return Settings(
        environment=environment,
        service_id=source.get("SERVICE_ID", "car-damage").strip() or "car-damage",
        control_plane_base_url=control_plane_base_url,
        service_credential=service_credential,
        introspection_timeout_seconds=_integer(source, "CONTROL_PLANE_INTROSPECT_TIMEOUT_SECONDS", 2, 1),
        auth_max_introspections_per_minute=_integer(source, "AUTH_MAX_INTROSPECTIONS_PER_MINUTE", 60, 1, 3600),
        auth_cache_max_entries=_integer(source, "AUTH_CACHE_MAX_ENTRIES", 4096, 1),
        auth_cache_max_ttl_seconds=_integer(source, "AUTH_CACHE_MAX_TTL_SECONDS", 30, 1, 30),
        usage_flush_interval_seconds=_integer(source, "USAGE_FLUSH_INTERVAL_SECONDS", 10, 1),
        usage_retry_attempts=_integer(source, "USAGE_RETRY_ATTEMPTS", 3, 1),
        usage_max_queue_batches=_integer(source, "USAGE_MAX_QUEUE_BATCHES", 1000, 1),
        usage_max_records_per_batch=_integer(source, "USAGE_MAX_RECORDS_PER_BATCH", 500, 1, 500),
        usage_max_queued_records=_integer(source, "USAGE_MAX_QUEUED_RECORDS", 100_000, 1),
        allowed_hosts=_allowed_hosts(source, production),
        cors_allowed_origins=_cors_origins(source, production),
        api_docs_enabled=_boolean(source, "API_DOCS_ENABLED", True),
        forwarded_allow_ips=_forwarded_allow_ips(source, production),
        max_upload_bytes=_integer(source, "MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES, 1024),
        max_image_pixels=_integer(source, "MAX_IMAGE_PIXELS", DEFAULT_MAX_IMAGE_PIXELS, 10_000),
        max_image_dimension=_integer(source, "MAX_IMAGE_DIMENSION", DEFAULT_MAX_IMAGE_DIMENSION, 64),
        max_concurrent_assessments=_integer(source, "MAX_CONCURRENT_ASSESSMENTS", 4, 1),
        demo_fallback_enabled=demo_fallback_enabled,
        api_key_rate_limit_enforcement=_boolean(source, "API_KEY_RATE_LIMIT_ENFORCEMENT", False),
    )
