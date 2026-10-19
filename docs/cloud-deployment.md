# Render free-tier demo deployment

**Deployed and publicly verified:** [Payment Risk Workbench](https://payment-risk-workbench.onrender.com), service `srv-dbanldcs728c73eqoglg`, on the Free plan in Virginia. The initial deploy `dep-dbanldss728c73eqoi40` used commit `386be79612efb05087e95773c3662cc3334c6fbb`. Public health, saved-prediction parity, all 200 demo payments, input rejection, page/assets, and explanation reconstruction passed. [Deployment evidence](../reports/cloud-deployment-verification.json) records the exact checks and identifiers. The model remains a development candidate.

The selected demo host is Render's free Docker web service. `render.yaml` declares one service, `payment-risk-workbench`, with `plan: free`, the existing Dockerfile, `/health`, and deploys triggered when linked CI checks pass. It declares no database, persistent disk, or paid compute. The schema has been validated against [Render's published blueprint schema](https://render.com/schema/render.yaml.json).

The free service has 512 MB RAM and 0.1 CPU. It sleeps after 15 minutes without inbound traffic and takes about a minute to wake. The workspace includes 750 free instance hours per calendar month; bandwidth and build minutes have separate shared limits. A free compute plan is not an unlimited usage allowance: workspaces with a payment method can incur supplemental usage charges. Keep usage within the workspace's included limits and use the applicable spend controls. These are provider terms checked on 2026-10-18, not a guarantee about an existing account's bill. See [free-tier details](https://render.com/docs/free) and [service plans](https://render.com/docs/blueprint-spec#plan).

## Deploy the prepared service

1. Commit and push the application, exact deployment snapshot, UI assets, release manifests, tests, and workflow. Check that the `Inference container` GitHub Actions run succeeds for that commit.
2. Connect Render to the chosen GitHub repository. Create a Blueprint from `https://github.com/tferrerd/01-payment-risk-workbench`, branch `main`, using the root `render.yaml`. Confirm the resulting service is **Free**; no upgrade or add-on is part of this configuration.
3. Record the actual Render service ID, deployed Git commit, image/deploy identity, and returned `https://…onrender.com` URL. Never infer the hostname from the display name. Check the service reports healthy before handoff.
4. Run the smoke check against that returned URL from a checkout of the same release:

```bash
poetry run python scripts/check_release.py \
  --manifest deployment/explorer-v2.json \
  --url https://ACTUAL-SERVICE-URL.onrender.com
```

The service listens on `0.0.0.0:8000`; Render's router detects the container's single HTTP port and provides TLS. The UI and API share an origin, so no separate frontend host or CORS setup is needed. The embedded model loads without a network call to MLflow. The local SQLite tracking store stays local. See [Render Docker support](https://render.com/docs/docker), [port binding](https://render.com/docs/web-services#port-binding), and [CI-gated deploy behavior](https://render.com/docs/blueprint-spec#autodeploytrigger).

## Verification and release scope

Local Docker verification is implementation evidence. The initial Render deployment is complete: the actual service URL, deploy identity, health response, and public inference smoke result are recorded in `reports/cloud-deployment-verification.json`. `reports/explorer-verification.json` retains the earlier local checks. No independent model-quality test or production promotion is implied by deployment.

The initial deployment builds on Render from the pinned source/model and base-image digest. It is not claimed to be the identical image built by GitHub Actions: transitive Python dependencies and Debian packages still resolve at build time. Record each deployed image/deploy identity. A later image-publishing step can make CI and deployment consume one immutable registry image.

Monitoring release `lightgbm-low-margin-dev-v1-explorer-v2` is publicly verified at the same service URL. `/health` reported source revision `dc3fbe424d55af85ad85857aa6f86000c6ca262f` after its [CI run](https://github.com/tferrerd/01-payment-risk-workbench/actions/runs/39190409081) passed. Saved predictions, validation, explorer explanations, and live aggregate counter changes passed; see [release smoke evidence](../reports/monitoring-cloud-smoke.json), [monitoring evidence](../reports/monitoring-cloud-verification.json), and the [operating record](operating-record.md). The local rollback report does not establish a Render rollback, and this update does not record a new provider deploy ID or immutable image digest.

This remains a public portfolio demonstration of a development candidate. The monitoring release and local application rollback are covered by the [operations runbook](operations.md). Fresh low-margin model validation, reliable probability calibration, persistent monitoring, retraining, and a Render rollback exercise remain subsequent work. No production model promotion is performed by deployment.
