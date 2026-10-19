# Render free-tier demo deployment

The selected demo host is Render's free Docker web service. `render.yaml` declares one service, `payment-risk-workbench`, with `plan: free`, the existing Dockerfile, `/health`, and deploys triggered when linked CI checks pass. It declares no database, persistent disk, or paid compute. The schema has been validated against [Render's published blueprint schema](https://render.com/schema/render.yaml.json).

The free service has 512 MB RAM and 0.1 CPU. It sleeps after 15 minutes without inbound traffic and takes about a minute to wake. The workspace includes 750 free instance hours per calendar month; bandwidth and build minutes have separate shared limits. A free compute plan is not an unlimited usage allowance: workspaces with a payment method can incur supplemental usage charges. Keep usage within the workspace's included limits and use the applicable spend controls. These are provider terms checked on 2026-10-18, not a guarantee about an existing account's bill. See [free-tier details](https://render.com/docs/free) and [service plans](https://render.com/docs/blueprint-spec#plan).

## Deploy the prepared service

1. Commit and push the application, exact deployment snapshot, UI assets, release manifests, tests, and workflow. Check that the `Inference container` GitHub Actions run succeeds for that commit.
2. Connect Render to the chosen GitHub repository. Create a Blueprint from `https://github.com/tferrerd/01-payment-risk-workbench`, branch `main`, using the root `render.yaml`. Confirm the resulting service is **Free**; no upgrade or add-on is part of this configuration.
3. Record the actual Render service ID, deployed Git commit, image/deploy identity, and returned `https://…onrender.com` URL. Never infer the hostname from the display name. Check the service reports healthy before handoff.
4. Run the smoke check against that returned URL from a checkout of the same release:

```bash
poetry run python scripts/check_release.py \
  --manifest deployment/explorer-v1.json \
  --url https://ACTUAL-SERVICE-URL.onrender.com
```

The service listens on `0.0.0.0:8000`; Render's router detects the container's single HTTP port and provides TLS. The UI and API share an origin, so no separate frontend host or CORS setup is needed. The embedded model loads without a network call to MLflow. The local SQLite tracking store stays local. See [Render Docker support](https://render.com/docs/docker), [port binding](https://render.com/docs/web-services#port-binding), and [CI-gated deploy behavior](https://render.com/docs/blueprint-spec#autodeploytrigger).

## Verification and release scope

Local Docker verification is implementation evidence. Render deployment is complete only once an actual service URL, deploy identity, health response, and public inference smoke result have been recorded. Connecting an account or validating a blueprint alone does not deploy a service. `reports/explorer-verification.json` records the local checks; a cloud deployment record should be added only after the remote checks pass.

The initial deployment builds on Render from the pinned source/model and base-image digest. It is not claimed to be the identical image built by GitHub Actions: transitive Python dependencies and Debian packages still resolve at build time. Record each deployed image/deploy identity. A later image-publishing step can make CI and deployment consume one immutable registry image.

This remains a public portfolio demonstration of a development candidate. Fresh low-margin model validation, reliable probability calibration, monitoring, retraining and rollback evidence remain subsequent work. No production model promotion is performed by deployment.
