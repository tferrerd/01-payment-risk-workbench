# Transaction explorer and cloud preparation

Recorded 2026-10-18 (America/Argentina/Cordoba).

The registered LightGBM development candidate now powers a transaction explorer served by the existing Python API. It is one service with a plain HTML/CSS/JavaScript page, no frontend framework, and no database. The model snapshot is about 357 KiB; the final container is about 150 MiB including Python and its numerical libraries.

The UI scores 200 source-derived selection payments through the real API, supports search/filter/sort/pagination and CSV export, explains selected scores with native LightGBM TreeSHAP, and simulates alternative legitimate-margin/fraud-loss assumptions. All explanation contributions reconstruct the model's raw log-odds and match its probability. Browser simulation controls leave the packaged API's 0.3% / 100% policy unchanged.

The sample contains 190 legitimate and 10 fraudulent payments, deliberately enriched for inspection. It is labeled throughout as curated selection evidence. No evaluation rows, model refits, new final evaluation, or production promotion are part of this delivery. The original inference manifest, historical freeze, and frozen model snapshots are preserved; a separate explorer service manifest binds the additional code and assets.

Verification passed: 14 local Python tests, two JavaScript policy tests, seven Python contract tests in the container before the runtime thread-limit adjustment, and final live HTTP checks for the page/assets, all 200 scored rows, three saved prediction examples, nine invalid prediction requests, and explanation reconstruction/label rejection. Desktop UI checks covered search, selection, live explanations, economic updates, and reset. The final local demo has a 512 MB memory cap and no artificial CPU quota. An earlier strict 0.1-CPU readiness test timed out; this is not a cloud performance benchmark. Detailed evidence is in `reports/explorer-verification.json`.

Render's free tier was selected for the demo. The single-service `render.yaml` passes Render's published schema validation and includes the Free plan, `/health`, and CI-gated deployment updates. No database, disk, paid compute, or additional cloud service is configured. Actual hosting remains pending an installed/connected Render integration and successful remote health/inference checks. No public service URL is claimed before that work succeeds.

See [explorer usage](../explorer.md), [container checks](../container.md), and [cloud deployment](../cloud-deployment.md).
