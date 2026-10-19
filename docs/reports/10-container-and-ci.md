# Container and CI delivery

Recorded 2026-10-18 (America/Argentina/Cordoba).

The existing low-margin LightGBM development candidate now runs in a local Docker container. Release `lightgbm-low-margin-dev-v1-inference-v1` embeds the exact version 1 snapshot, its checksum manifest, inference code, and three source-derived examples. No classifier was refitted, no new final evaluation was performed, and no production alias was assigned.

The Python base image is pinned by digest and uses native Linux ARM64 on this Mac. The service runs as a non-root user with a read-only root filesystem and local-only port binding. Training data, notebooks, environment files, and the local MLflow database are outside the build context. Only the compact deployment model and manifest are explicit exceptions to the model-artifact Git exclusion.

Verification passed: **12 local unit tests**, **five inference-contract tests inside the image**, three saved HTTP predictions matching probabilities and modeled profit within 1e-12, nine malformed requests returning JSON 422, OpenAPI field checks, and a healthy Docker health check. The host endpoint is `http://127.0.0.1:8001/docs`. Historical evaluation lineage and frozen model snapshots remain unchanged. Exact image identity, runtime versions, and verification details are saved in `reports/container-verification.json`.

The GitHub Actions workflow builds the container, runs inference tests and HTTP smoke checks, and archives the release manifest, smoke report, and image metadata under the Git commit identifier. It is prepared locally and has not yet been pushed or run on GitHub. It does not publish an image or deploy publicly.

**Next:** commit/push to exercise the remote checks, build the transaction explorer around the API, and select public hosting. Image publishing, monitoring, retraining, and rollback remain delivery work. Probability reliability near the low-margin cutoff and fresh labeled confirmation remain open before production promotion. See [container usage and CI scope](../container.md).
