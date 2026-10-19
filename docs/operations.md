# Monitoring and rollback

Release `lightgbm-low-margin-dev-v1-explorer-v2` adds operational telemetry to the existing development candidate. The model, feature schema, economics, and saved predictions remain unchanged. It adds no dependency, database, or monitoring service.

## Inspect the service

`GET /health` reports readiness, model/scenario identity, application release, and source revision. `GET /metrics` returns a JSON snapshot containing:

- Request counts and HTTP status counts for fixed route names, with unknown paths grouped as `/other`.
- Mean and nearest-rank p95 server latency from the last 1,000 requests per route.
- Accepted payment and cancellation counts, plus fixed histograms for raw fraud scores, Amount, and elapsed Time.
- Worker start time and uptime, so a restart can be distinguished from a decrease in counters.

Only successful `/predict` results enter the payment histograms. Invalid requests affect request/error counters, not score or input distributions. `/explain` does not double-count payments. Health probes and metrics scrapes are excluded from request statistics. Static assets share the bounded `/other` bucket. Telemetry retains no transaction bodies, labels, feature vectors, identifiers, query strings, or individual scores. Structured `http_request` logs contain route, status, elapsed milliseconds, and release ID.

Source identity uses Render's runtime `RENDER_GIT_COMMIT`, falling back to the Docker build's `APP_REVISION` locally. See [Render's documented environment variables](https://render.com/docs/environment-variables#render_git_commit).

The configured service has **one worker**. Counters live in memory and reset when it restarts or sleeps. Latency covers application handling, including inference and serialization, rather than network transit or a sleeping instance's wake-up. The snapshot is public aggregate demo telemetry. It is not a persistent availability history, a cross-worker metrics backend, or an audit ledger.

Run a read-only inspection from the repository root:

```bash
python3 scripts/check_operations.py --url https://payment-risk-workbench.onrender.com
```

Add `--exercise` to send the three saved payment examples and one deliberately invalid empty batch, then verify that accepted-payment and validation-error counters increased. This creates explicitly identified smoke traffic; it is not organic usage. The script prints a dated JSON report and exits unsuccessfully on failed checks or alert conditions.

## Alert criteria and response

These are provisional demo thresholds, not measured service-level objectives. After at least 100 requests to an inference route (`/predict` or `/explain`), the checker flags either:

| Signal | Trigger | Response |
| --- | --- | --- |
| Server errors | More than 1% HTTP 5xx since worker start | Inspect Render logs and release identity; reproduce with saved examples. Roll back a faulty release. |
| Server latency | p95 above 1,000 ms in the retained route samples | Check batch sizes, resource usage, recent release, and warm-instance behavior. Do not treat cold starts as model latency. |
| Readiness or prediction parity | Health fails, identity differs, or saved predictions disagree | Stop a rollout; restore the known-good build and rerun release checks. |

Validation errors remain visible separately as 422 responses. A 422 from the deliberately invalid smoke request is expected and is not a model failure. The automated tests inject a server failure inside an isolated ASGI application and verify both error recording and alert thresholds; no public fault endpoint is exposed.

Alerts are printed by the checker and produce a nonzero exit status. No email, paging destination, scheduler, or continuous uptime poller is configured. Run the checker when investigating or after deployment; add scheduled collection and a notification destination when the demo needs unattended operations.

Score, Amount, and Time histograms support distribution comparisons only when the population, window, release, and denominators are comparable. The curated 200-payment explorer sample is deliberately fraud-enriched. Neither it nor repeated smoke traffic provides a production baseline. No drift or fraud-quality alarm is inferred from these counters alone. Delayed-label precision/loss monitoring and fresh model validation remain open.

## Rehearse an application rollback locally

The drill uses two **exact Docker image IDs**, starts one isolated container at a time, and publishes no host port. It checks the previous release, the candidate, the restored previous release, and the candidate again. Each phase verifies release-file checksums, saved prediction parity, nine invalid input cases, all 200 explorer payments, and explanation reconstruction. It only removes the uniquely named container it creates; the existing local demo and Render service are unaffected.

```bash
docker build --build-arg REVISION=local --tag payment-risk:explorer-v2 .
python3 scripts/drill_rollback.py \
  --previous-image payment-risk:explorer-v1 \
  --candidate-image payment-risk:explorer-v2 \
  > reports/rollback-verification.json
```

The previous image must already exist; tags are resolved to immutable local IDs before the drill. Rebuild an unavailable prior image from the earlier source checkout, not by applying the old manifest to changed current files. The original `deployment/release.json` and `deployment/explorer-v1.json` remain historical snapshots; current builds use `deployment/explorer-v2.json`.

Both application releases embed the same LightGBM artifact and policy. This demonstrates application-release recovery and post-rollback inference, **not a rollback to a different trained model**. A model-version rollback requires another compatible packaged model, matched preprocessing/policy, and its own evidence.

## Restore a Render build

1. Open the service's [Deploys page](https://dashboard.render.com/web/srv-dbanldcs728c73eqoglg). Record the live deploy ID, commit, and health response before changing anything.
2. Select a retained, successful known-good build and choose **Rollback**, then **Rollback to this deploy**. Restore its existing build artifact, rather than rebuilding a moving branch or mutable tag.
3. Wait for the restored service to become healthy. From the matching source checkout, run `scripts/check_release.py --manifest deployment/explorer-v1.json --url https://payment-risk-workbench.onrender.com` for v1, or the v2 manifest for v2. Preserve the resulting parity report and the actual Render deploy ID.
4. Diagnose/fix the regression before returning to the newer release. Render dashboard rollbacks disable automatic deploys; after recovery, restore the existing **After CI Checks Pass** behavior in service settings. Verify the desired release and source revision after deployment.

Render can only restore builds it still retains. Its rollback restores a build and selected service settings; it does not reverse external state. This service has no database or disk migration. See [Render's rollback documentation](https://render.com/docs/rollbacks). A local drill report does not establish that a Render rollback has been executed.
