# Car Damage API release

GitLab project `330` uses project runners `car-damage-ci` and
`car-damage-build` on `100.64.0.13`. Merge requests run the API tests. Protected
`main` pipelines also build `Dockerfile.api` with rootless BuildKit. The build
runs the full suite and model-backed CPU smoke against the production image
layers before pushing `ghcr.io/fanfanfw/car-damage-api:<commit>` and recording
the digest. The runner jobs have no Docker socket or privileged mode.

The API image pins Python dependencies in `requirements-api-image.lock` and
verifies `models/best.pt.sha256` at build and startup. The CUDA worker warms up
the checkpoint before it reports ready. To smoke an already pulled release on
the GPU host, substitute its digest in the command below.

```sh
docker run --rm --gpus device=0 --network none --read-only \
  --tmpfs /tmp:size=256m,mode=1777 -e CAR_DAMAGE_REQUIRE_CUDA=true -e PYTHONPATH=/app \
  -v "$PWD/ci/smoke_api_image.py:/smoke-api.py:ro" \
  ghcr.io/fanfanfw/car-damage-api@sha256:<digest> python /smoke-api.py
```

Dokploy Compose `2Ga3dc2OpCu568hsGLvkS` on `.13` uses the reviewed raw
manifest in `deploy/compose.production.yaml` with native auto-deploy disabled.
Its environment contains one nonsecret setting, `CAR_DAMAGE_IMAGE`. Kong
`.13` reaches the host loopback port `18000`. Kong `.14` uses a Tailnet bridge
on `.13` that admits only `.14`. Kong owns the key and consumer identity. The
API has no runtime credentials or persistent application data.

GitLab holds the protected GHCR push credential from Infisical `ci/cd` prod
and a protected `production`-scoped Dokploy key with access only to this
Compose. The host uses Dokploy's existing GHCR read-only registry credentials.
Play `deploy-production` manually on the approved protected commit after the
digest passes the GPU smoke. The job is available when the protected
`CAR_DAMAGE_DEPLOY_ENABLED` variable equals `true`.

The job records the old pin and waits for its own deployment. A terminal
failure restores the old pin and checks readiness. An unknown deployment
outcome requires checking the live Compose and image before queuing another
release. The first promotion can roll back to the retained local bootstrap
image `car-damage-api:bootstrap-09ae5c41b40a`. Gateway rollback on server-207
returns the Car Damage vhost to `.13:8080` independently of the image pin.
