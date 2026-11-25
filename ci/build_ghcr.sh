#!/bin/sh
# Protected-main CI only: build Dockerfile.api with rootless BuildKit and
# push ghcr.io/<owner>/car-damage-api:<commit>. Never print credentials or
# enable shell tracing. Writes build/image-metadata.json + build/release-commit.txt.
set -eu
case ${CI_PIPELINE_SOURCE:-} in
  push|web|api) ;;
  *) echo "Refusing registry push outside the supported pipeline sources" >&2; exit 1 ;;
esac
[ "${CI_COMMIT_BRANCH:-}" = main ] &&
[ "${CI_COMMIT_REF_PROTECTED:-}" = true ] || {
  echo 'Refusing registry push outside protected main pipeline' >&2; exit 1;
}
: "${GHCR_USERNAME:?Missing GHCR_USERNAME}" "${GHCR_PUSH_TOKEN:?Missing GHCR_PUSH_TOKEN}"
: "${GHCR_IMAGE:?Set GHCR_IMAGE to ghcr.io/<owner>/car-damage-api}"
printf '%s\n' "$GHCR_IMAGE" | grep -Eq '^ghcr\.io/[a-z0-9]+([._-][a-z0-9]+)*/car-damage-api$' || {
  echo 'Invalid GHCR_IMAGE: expected lowercase ghcr.io/<owner>/car-damage-api' >&2; exit 1;
}
printf '%s\n' "${CI_COMMIT_SHA:-}" | grep -Eq '^[0-9a-f]{40}$' || {
  echo 'Invalid commit SHA' >&2; exit 1;
}
[ -f Dockerfile.api ] && [ -f .dockerignore ] || {
  echo 'Missing Dockerfile.api or .dockerignore' >&2; exit 1;
}
export DOCKER_CONFIG="$(mktemp -d)"
trap 'unlink "$DOCKER_CONFIG/config.json"' EXIT
umask 077
auth="$(printf '%s:%s' "$GHCR_USERNAME" "$GHCR_PUSH_TOKEN" | base64 | tr -d '\n')"
printf '{"auths":{"ghcr.io":{"auth":"%s"}}}\n' "$auth" > "$DOCKER_CONFIG/config.json"
unset auth GHCR_PUSH_TOKEN
mkdir -p build
buildctl-daemonless.sh build \
  --frontend dockerfile.v0 \
  --local context=. \
  --local dockerfile=. \
  --opt filename=Dockerfile.api \
  --opt "label:org.opencontainers.image.revision=$CI_COMMIT_SHA" \
  --opt "label:org.opencontainers.image.source=$CI_PROJECT_URL" \
  --metadata-file build/image-metadata.json \
  --output "type=image,name=$GHCR_IMAGE:$CI_COMMIT_SHA,push=true"
printf '%s\n' "$CI_COMMIT_SHA" > build/release-commit.txt
echo "Image pushed: $GHCR_IMAGE:$CI_COMMIT_SHA (production not deployed by this job)"
