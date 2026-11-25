#!/usr/bin/env python3
"""Deploy one immutable Car Damage image to the existing GPU Compose on .13."""
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

DOKPLOY_URL = "http://100.64.0.13:3005/api"
COMPOSE = "2Ga3dc2OpCu568hsGLvkS"
SERVER = None
BACKEND = "https://car-damage.ifcontech.com/api"
TITLE_PREFIX = "car-damage-ci-"
IMAGE_RE = r"ghcr\.io/fanfanfw/car-damage-api"
PIN_RE = IMAGE_RE + r"@sha256:[0-9a-f]{64}"
PIN_KEY = "CAR_DAMAGE_IMAGE"
BOOTSTRAP_IMAGE = "car-damage-api:bootstrap-09ae5c41b40a"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def opener():
    # No proxy and no redirects: the key must only ever reach the approved host.
    return build_opener(ProxyHandler({}), NoRedirect)


class Api:
    def __init__(self):
        self.base = os.environ.get("DOKPLOY_URL", DOKPLOY_URL).rstrip("/")
        if self.base != DOKPLOY_URL:
            raise ValueError("Unapproved Dokploy management URL")
        self.key = os.environ["DOKPLOY_API_KEY"]
        self.opener = opener()

    def call(self, endpoint, payload=None, **query):
        url = self.base + "/" + endpoint
        if query:
            url += "?" + urlencode(query)
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(url, data=data, headers={"x-api-key": self.key,
                                                  "Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=30) as response:
                body = response.read()
                return json.loads(body) if body else None
        except HTTPError as error:
            raise RuntimeError(f"Dokploy {endpoint}: HTTP {error.code}") from None
        except (URLError, TimeoutError):
            raise RuntimeError(f"Dokploy {endpoint}: connection failed; outcome may be unknown") from None


def wait_deployment(api, before, title, seconds=600):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rows = api.call("deployment.allByCompose", composeId=COMPOSE)
        if not isinstance(rows, list):
            raise RuntimeError("Unexpected deployment list response")
        matches = [row for row in rows if row.get("deploymentId") not in before
                   and row.get("title") == title]
        if len(matches) > 1:
            raise RuntimeError("Ambiguous deployment identity; manual inspection required")
        if matches and matches[0].get("status") in ("done", "error"):
            return matches[0]
        time.sleep(5)
    raise TimeoutError("Deployment outcome unknown; do not queue competing rollback")


def release_from_artifacts(sha, image):
    if Path("build/release-commit.txt").read_text().strip() != sha:
        raise ValueError("Build artifact commit does not match deployment commit")
    metadata = json.loads(Path("build/image-metadata.json").read_text())
    digest = metadata["containerimage.digest"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Invalid build digest")
    return image + "@" + digest


def verify_backend(attempts=18):
    client = opener()
    for _ in range(attempts):
        try:
            for path, status in (("/health/live", "ok"), ("/health/ready", "ready")):
                with client.open(BACKEND + path, timeout=5) as response:
                    if response.status != 200 or json.load(response).get("status") != status:
                        break
            else:
                return
        except (URLError, TimeoutError, ValueError):
            pass
        time.sleep(5)
    raise RuntimeError("Backend liveness or model readiness check did not pass")


def production_pin(api):
    current = api.call("compose.one", composeId=COMPOSE)
    expected = {"composeId": COMPOSE, "serverId": SERVER, "sourceType": "raw",
                "appName": "compose-back-up-mobile-system-6osf3y", "composeType": "docker-compose",
                "environmentId": "ZnSiTBEAf9TiaMVg9T98z",
                "createEnvFile": True, "autoDeploy": False}
    if any(current.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Production target, source or trigger does not match")
    if current.get("composeStatus") in ("running", "queued"):
        raise RuntimeError("Production target is already deploying")
    if current.get("composeFile") != Path("deploy/compose.production.yaml").read_text():
        raise RuntimeError("Production Compose drifted from the reviewed release manifest")
    env = current.get("env")
    match = re.fullmatch(PIN_KEY + r"=([^\r\n]+)\n?", env or "")
    if not match or (match[1] != BOOTSTRAP_IMAGE and not re.fullmatch(PIN_RE, match[1])):
        raise RuntimeError("Production environment must contain exactly one approved image pin")
    return match[1]


def update_and_deploy(api, image, title, description, before):
    api.call("compose.saveEnvironment", {"composeId": COMPOSE, "env": PIN_KEY + "=" + image})
    if production_pin(api) != image:
        raise RuntimeError("Image pin read-back mismatch; inspect before deploying")
    api.call("compose.deploy", {"composeId": COMPOSE, "title": title, "description": description})
    return wait_deployment(api, before, title)


def deployment_ids(api, pipeline):
    rows = api.call("deployment.allByCompose", composeId=COMPOSE)
    if not isinstance(rows, list):
        raise RuntimeError("Unexpected deployment list response")
    for row in rows:
        if row.get("status") in ("running", "queued"):
            raise RuntimeError("Target already deploying; refusing concurrent update")
        match = re.fullmatch(TITLE_PREFIX + r"(\d+)-\d+", row.get("title") or "")
        if match and int(match[1]) > pipeline and row.get("status") == "done":
            raise RuntimeError("A newer pipeline was deployed; refusing obsolete release")
    return {row["deploymentId"] for row in rows}


def main():
    if (os.getenv("CI_PIPELINE_SOURCE") not in ("push", "web", "api")
            or (os.getenv("CI_COMMIT_BRANCH"), os.getenv("CI_COMMIT_REF_PROTECTED"),
                os.getenv("CAR_DAMAGE_DEPLOY_ENABLED")) != ("main", "true", "true")):
        raise RuntimeError("Deploy requires the enabled manual job on protected main")
    sha = os.environ["CI_COMMIT_SHA"]
    image = os.environ["GHCR_IMAGE"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or not re.fullmatch(IMAGE_RE, image):
        raise ValueError("Invalid release commit or image repository")
    release = release_from_artifacts(sha, image)
    api = Api()
    current = production_pin(api)
    pipeline = int(os.environ["CI_PIPELINE_ID"])
    before = deployment_ids(api, pipeline)
    title = f"{TITLE_PREFIX}{pipeline}-{int(os.environ['CI_JOB_ID'])}"
    receipt = {"composeId": COMPOSE, "commit": sha, "image": release, "previousImage": current}
    receipt_file = Path("build/deploy-receipt.json")
    receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")
    result = update_and_deploy(api, release, title, "GitLab commit " + sha, before)
    try:
        if result["status"] != "done":
            raise RuntimeError("Deployment failed")
        verify_backend()
        if production_pin(api) != release:
            raise RuntimeError("Release changed during deployment")
    except RuntimeError:
        if production_pin(api) != release:
            raise RuntimeError("Concurrent Compose change; rollback requires an operator") from None
        rolled = update_and_deploy(api, current, title + "-rollback", "Restore previous release",
                                   deployment_ids(api, pipeline))
        if rolled["status"] != "done":
            raise RuntimeError("Deployment and rollback failed; operator required") from None
        verify_backend()
        raise RuntimeError("Release failed; previous release redeployed") from None

    receipt.update(deploymentId=result["deploymentId"], readiness="ok")
    receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")
    print("Deployment complete; backend liveness and model readiness passed")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        # Only explicitly sanitized RuntimeError messages may be shown.
        print("Deploy failed:", str(error) if type(error) is RuntimeError else type(error).__name__)
        raise SystemExit(1)
