# Payment Risk Workbench — implementation framework

Recorded 2026-10-17. Status: plan agreed in direction; analysis and implementation pending.

## Reference and approach

Follow [JuanFeDS/fraud_shipping](https://github.com/JuanFeDS/fraud_shipping/tree/734e4306725c3aa306845e7040a0525d6730da70), reviewed at commit `734e4306725c3aa306845e7040a0525d6730da70`.

Adopt its progression from business economics through analysis, fitted inference, and operational delivery. Use numbered notebooks with corresponding executive reports, shared Python code, repeatable scripts, MLflow lineage, and a versioned deployment. This is an original implementation on the ULB dataset. Reference results are not project results. No reference code has been copied.

## Project charter

- **User:** A payment risk analyst managing a limited transaction review queue.
- **Decision:** Approve, challenge, or review each transaction using its calibrated fraud probability, amount, and an explicit policy.
- **Prediction time:** Immediately before authorization. Assume published V1–V28 and Amount are available then; their upstream construction cannot be verified from this dataset. Time is elapsed time, not a calendar timestamp or a known local hour.
- **Baseline:** Approve all; also compare a simple amount rule, logistic regression, and a boosted tree on identical evaluation rows.
- **Primary KPI:** Simulated total policy cost per 1,000 transactions, with the absolute cost, transaction count, time window, and difference versus approve-all reported alongside it.
- **Guardrails:** Legitimate approval rate, false-positive rate, fraud recall, review utilization, and challenge rate. Proposed primary review cap: 1% of transactions in each one-hour replay window; sensitivity at 0.5%, 2%, and 5%. These are scenario settings, not observed operational capacity.
- **Success:** A reproducible learned policy improves primary-scenario cost over approve-all and the amount rule on untouched temporal evaluation data while respecting the review cap. Report paired uncertainty; if the interval includes no improvement, label the result inconclusive. Select the deployable candidate using development data only.

## Economics and policy

Define costs before model selection. Count approved fraud as a configurable fraction of Amount, legitimate friction as a configurable cost, and review as a per-transaction cost. Specify challenge abandonment, fraud prevention, review accuracy, and review delay explicitly. None of these outcomes or costs is observed in the source data.

Choose and freeze a primary scenario in the first analysis stage, with alternative scenarios for sensitivity. Show any profit-margin scenario separately. A threshold of 0.20 or a 25% margin is not established for this dataset.

Rank review candidates by expected cost reduction from review, subject to capacity. Replay arrivals in order: a live decision cannot use transactions that have not arrived. Label any full-window ranking as a retrospective batch scenario. Keep calibration and policy parameters with the fitted model so the application serves the evaluated behavior.

## Analysis sequence

Each stage produces an executed notebook and a short report under `docs/reports/` describing the question, evidence, limitation, and decision. Add stages when their inputs exist; do not create empty notebooks.

| Stage | Question and required evidence |
|---|---|
| 01 — Exploration and charter | Audit schema, missingness, labels, amounts, duplicates and temporal counts; freeze economics and the evaluation protocol. |
| 02 — Baselines | Compare approve-all, an amount rule, and fitted logistic regression; record ranking, calibration, and decision costs. |
| 03 — Features | Test only transformations supported by these numeric features; fit all learned transformations on training data. |
| 04 — Experiments | Compare a boosted tree with logistic regression; track data, code, splits, parameters and artifacts in MLflow. Tune only when baseline evidence justifies it. |
| 05 — Explanations and failures | Explain representative expensive mistakes and useful predictions; PCA components cannot support invented customer or merchant narratives. |
| 06 — Temporal calibration and policy | Fit calibration on later development observations; assess cost and capacity policies using earlier data only. |
| 07 — Uncertainty and recommendation | Run the frozen final evaluation, paired cost comparisons, and sensitivity analysis; recommend deployment or document why evidence is insufficient. |

## Evaluation protocol

Proposed time boundaries, to validate against label counts during stage 01: training `[0, 86400)`, model selection `[86400, 115200)`, calibration/policy development `[115200, 144000)`, and final evaluation `[144000, end]`. Freeze boundaries before model experiments. Keep tied timestamps together and persist split assignments with source row IDs. If class counts cannot support a stage, revise the protocol before fitting and record the reason.

Audit exact duplicates and identical-feature conflicting labels across boundaries. Report overlap and a sensitivity evaluation excluding repeated evaluation feature vectors; do not silently discard records or invent customer groups. There are no customer identifiers for grouped validation. Upstream PCA fitting provenance limits any claim of fully leakage-free preprocessing.

Use average precision as the stated PR ranking metric, recall at a development-selected threshold targeting 0.1% false positives, realized false-positive rate, Brier score, reliability plots, and simulated policy cost. Report denominators and fraud counts for every window. The final period is opened once after candidate, calibrator, costs, and policy are frozen. Reusing it later makes it a regression benchmark, not a new independent test.

Use paired time-block resampling for cost differences, disclose block length and the short-window limitation, and check sensitivity to block size. Two days cannot establish long-term drift robustness, seasonality, or production impact.

## Delivery after analytical evidence

1. Package preprocessing, classifier, calibrator, and policy; provide train, evaluate, and batch-predict commands.
2. Add validated API inference and the transaction explorer, risk explanations, policy controls, review queue, and cost tradeoff chart.
3. Register candidates in MLflow. Bind evaluation to code, data, schema, feature, split, and policy versions. Evaluate candidate and incumbent on the same window before promotion.
4. Build a container with a pinned artifact; add CI, health and inference checks, and a traceable release. Choose hosting when deployment requirements are known.
5. Implement score/input monitoring and delayed-label quality reports, a retraining replay, and an executed rollback with a post-rollback prediction. Clearly label replayed monitoring and simulated labels/outcomes.
6. Publish the stakeholder memo, ownership and operating record, public demo, case study, and original editable presentation with reviewed PDF.

The first implementation milestone is stage 01. The dataset acquisition profile alone does not complete the audit. No model, operational service, or business improvement is claimed yet.
