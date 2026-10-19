# Payment Risk Workbench — implementation framework

Recorded 2026-10-17; status updated 2026-10-18. Stages 01–08 are executed, including a frozen historical-scenario evaluation and a separate low-margin economic hypothesis. Reusable LightGBM inference, batch prediction, API, development registry entry, and a publicly verified Render demo are implemented. Stronger probability-calibration evidence and the remaining operational lifecycle work are pending.

## Implementation approach

Progress from business economics through analysis, fitted inference, and operational delivery. Use numbered notebooks with corresponding executive reports, shared Python code, repeatable scripts, MLflow lineage, and a versioned deployment. Base project findings and implementation decisions on the ULB dataset and reproducible experiments.

## Project charter

- **User:** A payment risk analyst managing a limited transaction review queue.
- **Decision:** Initially approve or cancel each transaction using its fraud probability and amount. Challenge/review extensions require separately specified economics.
- **Prediction time:** Immediately before authorization. Assume published Time, V1–V28 and Amount are available then; their upstream construction cannot be verified from this dataset. Time is elapsed time, not a calendar timestamp or a known local hour.
- **Baseline:** Approve all; also compare a simple amount rule, logistic regression, and a boosted tree on identical evaluation rows.
- **Primary KPI:** Simulated profit per 1,000 transactions, with total profit, percentage of the perfect-label maximum, transaction count, split, and comparison with approve-all.
- **Guardrails:** Legitimate approval rate, legitimate cancellation rate, fraud canceled recall, and total cancellation rate. The current binary policy has no review queue or capacity constraint; those are deferred extensions.
- **Success:** A reproducible learned policy improves primary-scenario profit over approve-all and the amount rule on the reserved random evaluation set. Disclose prior EDA exposure. Report paired uncertainty; if the interval includes no improvement, label the result inconclusive. Select the deployable candidate using development data only.

## Economics and policy

The [historical profit scenario](cost-scenario.md) and [versioned configuration](../config/cost-scenario.json) use 25% legitimate profit, 100% fraud loss, and zero cancellation payoff for stages 02–07. The development inference candidate uses stage 08’s [0.3% margin scenario](../config/cost-scenario-low-margin.json). Initial baseline fee/effectiveness assumptions have been superseded.

For meaningful fraud probabilities, cancellation is preferred above margin/(margin + fraud loss): 0.20 historically, approximately 0.002991 for the development candidate. Exact ties favor approval. Compare economic value and ranking metrics separately. Calibrate and version probabilities and policy together before final release. Review/challenge policies need their own assumptions if introduced later.

## Analysis sequence

Each stage produces an executed notebook and a short report under `docs/reports/` describing the question, evidence, limitation, and decision. Add stages when their inputs exist; do not create empty notebooks.

| Stage | Question and required evidence |
|---|---|
| 01 — Exploration and charter | Audit schema, missingness, labels, amounts, duplicates and temporal counts; freeze economics and the evaluation protocol. |
| 02 — Baselines | Compare approve-all, an amount rule, and fitted logistic regression; record ranking, calibration, and decision costs. |
| 03 — Features | Test only transformations supported by these numeric features; fit all learned transformations on training data. |
| 04 — Experiments | Compare a boosted tree with logistic regression; track data, code, splits, parameters and artifacts in MLflow. Tune only when baseline evidence justifies it. |
| 05 — Explanations and failures | Explain representative expensive mistakes and useful predictions; PCA components cannot support invented customer or merchant narratives. |
| 06 — Calibration and policy | Compare mappings by grouped calibration-fold Brier, fit the selected mapping on calibration, and assess the fixed approve/cancel policy on selection. |
| 07 — Uncertainty and recommendation | Run the frozen final evaluation, paired cost comparisons, and sensitivity analysis; recommend deployment or document why evidence is insufficient. |
| 08 — Low-margin payments | Reprice existing scores at a user-proposed 0.3% retained margin / 100% fraud loss; compare algebraic policies and label consumed evaluation reuse as regression. |

## Evaluation protocol

The active protocol is `grouped-random-v1`: seeded, label-stratified random allocation of exact V1–V28/Amount groups, targeting 60% training, 15% selection, 10% calibration, and 15% evaluation. Time remains an input. Retain every payment and keep repeated patterns within one split. See the [protocol](evaluation-protocol.md), [configuration](../config/evaluation.json), and [audit](../reports/duplicate-split-audit.json).

Fit preprocessing on training only, compare candidates on selection, choose any calibration mapping using calibration-only folds, and assess the resulting policy on the reused selection set. Quantify paired development uncertainty, freeze one candidate, then score evaluation once. Report average precision, recall, false-positive rate, Brier score, reliability, and simulated profit with transaction and fraud counts. For paired uncertainty, resample pattern groups together. Full-dataset EDA and earlier experiments preceded this protocol; evaluation is not analytically untouched. The random split estimates held-out performance within this two-day population; future-period and production claims require additional data.

## Delivery after analytical evidence

1. Package preprocessing, classifier, calibrator, and policy; provide train, evaluate, and batch-predict commands.
2. Add validated API inference and the transaction explorer, risk explanations, policy controls, review queue, and cost tradeoff chart.
3. Register candidates in MLflow. Bind evaluation to code, data, schema, feature, split, and policy versions. Evaluate candidate and incumbent on the same evaluation payments before promotion.
4. Build a container with a pinned artifact; add CI, health and inference checks, and a traceable release. Choose hosting when deployment requirements are known.
5. Implement score/input monitoring and delayed-label quality reports, a retraining replay, and an executed rollback with a post-rollback prediction. Clearly label replayed monitoring and simulated labels/outcomes.
6. Publish the stakeholder memo, ownership and operating record, public demo, case study, and original editable presentation with reviewed PDF.

Stages 01–07 provide exploration, a split audit, fitted baselines, feature and model comparisons, error analysis, calibration checks, paired uncertainty, and a frozen final evaluation. The selected raw Random Forest supports modeled improvement over approve-all under the primary scenario; superiority over logistic regression is inconclusive. Next is a versioned offline/demo prototype and artifact registration. Production promotion and claims of reliable probabilities require further evidence.

The additional stage 08 preserves stages 02–07 and their freeze. The proposed low-margin economics change the development model ranking; they require their own calibration and fresh confirmation before a new final policy claim.

Inference delivery after the notebooks is recorded in [report 09](reports/09-inference-candidate.md). The local API and CSV path share validated inference code; the packaged artifact reproduces saved selection scores and is registered with a development-only `candidate` alias. The [container and CI workflow](container.md) packages that exact snapshot with release checksums, health checks, and saved-prediction smoke tests. The [transaction explorer](explorer.md) adds live explanations, search/filter/export, and explicit browser-only economic simulations. The [Render Free deployment](cloud-deployment.md) is publicly verified, with identity and smoke evidence in `reports/cloud-deployment-verification.json`. Aggregate score/input/request monitoring and a verified local application rollback are recorded in the [operations runbook](operations.md) and [operating record](operating-record.md). Image publishing, persistent alert collection, delayed-label quality monitoring, retraining, a Render rollback exercise, and stronger validation before production promotion remain pending.
