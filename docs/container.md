# Inference container and CI

The container serves `lightgbm-low-margin-dev-v1`, MLflow model `payment-risk-low-margin` version 1, under the 0.3% legitimate-margin / 100% fraud-loss scenario. The current application release is `lightgbm-low-margin-dev-v1-explorer-v2`, adding aggregate monitoring to the transaction explorer and inference API. The original inference manifest is preserved separately. The artifact is a development candidate with uncalibrated scores and selection evidence. Container verification does not establish independent model quality or promote the candidate.

## Run locally

From the project root, with Docker Desktop running:

```bash
docker build --build-arg REVISION=local --tag payment-risk:explorer-v2 .
docker run --detach --name payment-risk-explorer --read-only --tmpfs /tmp \
  --cap-drop ALL --security-opt no-new-privileges \
  --publish 127.0.0.1:8002:8000 payment-risk:explorer-v2
docker exec payment-risk-explorer python scripts/check_release.py \
  --manifest deployment/explorer-v2.json --url http://127.0.0.1:8000
```

Open `http://127.0.0.1:8002/` for the explorer, `/docs` for the API, or call `/health` and `/predict`. The host binding is local. Port 8002 lets this run alongside the original inference container on port 8001. Stop and remove it with `docker rm -f payment-risk-explorer`.

The pinned Python base is a multi-platform image index: Docker selects native ARM64 on Apple Silicon and AMD64 on the GitHub-hosted runner. No Intel emulation is requested on the Mac. The service runs as UID 10001 and loads its checksum-verified model once at startup. Its health check calls `/health`; smoke tests also verify predictions and rejected inputs. The model is embedded in the image; serving does not require MLflow, a registry connection, or training data.

## Model snapshot and release evidence

Only `models/candidates/lightgbm-low-margin-dev-v1.joblib` (about 357 KiB) and its JSON manifest are allowed into Git and the build context from the models directory. Other model snapshots, local MLflow storage, raw/processed data, notebooks, environment files, and credentials are excluded from the container. The UI assets and curated selection sample are included explicitly. This allows a clean checkout to build the exact existing model without retraining or access to the local registry.

`deployment/release.json` preserves the original inference manifest. `deployment/explorer-v1.json` preserves the previous explorer snapshot. `deployment/explorer-v2.json` records the current release ID, exact registered model version, artifact SHA-256, scenario, threshold, and checksums of the original inference files plus the UI, sample, and explorer code. Build verification fails on a checksum mismatch or an unloadable model. `scripts/check_release.py --manifest deployment/explorer-v2.json --url ...` checks those files, three saved predictions (probabilities and modeled profit within absolute tolerance 1e-12), development metadata, OpenAPI fields, and nine invalid requests returning JSON 422. It also scores all 200 demo payments, checks the page/assets and a reconstructed explanation, and rejects labels as explanation inputs. The v2 manifest additionally binds monitoring code and its operational checker; CI verifies live counter changes using saved smoke traffic.

The Python base digest and direct Python dependency versions are pinned. Transitive Python dependencies and Debian package repositories are still resolved at build time, so builds are not claimed to be byte-identical. Record each built/deployed image identity; a tag alone is mutable. Original inference evidence is in `reports/container-verification.json`; the explorer's checks are in `reports/explorer-verification.json`. Render currently builds its own image from the source commit; sharing one published image between CI and hosting is subsequent work.

## GitHub Actions

`.github/workflows/inference-ci.yml` runs on pushes to main, pull requests, and manual dispatch. With read-only repository permissions and actions pinned to commit SHAs, it:

1. Builds an image labeled with the Git commit, release ID, and model checksum.
2. Runs ten Python inference/explorer/monitoring tests inside the image, including ties, zero amounts, validation, artifact integrity, saved predictions, explanation reconstruction, and sample provenance; also runs two browser-policy calculation tests with Node.
3. Starts the service with a read-only filesystem and runs the real-model HTTP smoke checks.
4. Archives the smoke report, release manifest, and inspected image metadata under the commit identifier, then removes the CI container.

The workflow passed on GitHub for the initial application commit `386be79`; its [run](https://github.com/tferrerd/01-payment-risk-workbench/actions/runs/39186793706) archives verification evidence. The workflow itself does not publish an image or create a public service. The [Render blueprint](cloud-deployment.md) has separately deployed the Free service, with successful public health and inference checks recorded. The [operations runbook](operations.md) documents aggregate monitoring and the executed local application rollback, with [rollback evidence](../reports/rollback-verification.json). Image publishing, persistent alert collection, delayed-label quality monitoring, retraining, and a Render rollback exercise remain subsequent delivery work. The historical evaluation freeze and analytics lockfile are preserved.

Implementation references: [Dockerfile instructions](https://docs.docker.com/reference/dockerfile/) and [GitHub Actions artifacts](https://github.com/actions/upload-artifact).
