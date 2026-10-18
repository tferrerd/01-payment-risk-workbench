# 01 — Payment Risk Workbench

- **Application:** Fraud detection
- **Portfolio project:** 01 of 15
- **Status:** Published source data acquired and verified; exploratory analysis is in progress. ML, application and MLOps implementation is planned.

Part of [Tomas Ferrer's DS/ML portfolio](https://tomasferrer.com/).

## Implementation framework

Follow the business-to-production workflow of [JuanFeDS/fraud_shipping](https://github.com/JuanFeDS/fraud_shipping), adapted to this dataset and the portfolio lifecycle requirements. The [implementation plan](docs/implementation-plan.md) defines the project charter, seven analysis stages, proposed temporal evaluation protocol, and delivery sequence. Start with exploration and explicit transaction economics; carry the evaluated preprocessing, model, calibration, and policy into the application.

## Committed CSV dataset

The **complete dataset is committed** under [`data/raw/`](data/raw/), split into five CSV files with identical headers. All 284,807 transactions and 492 fraud labels are included; no download step is needed to use the CSVs. See [data files and attribution](data/DATASET.md) and [CSV checksums](data/csv-manifest.json). The original ARFF is retained locally and remains available through the acquisition script.

## Project prompt

Build an end-to-end fraud detection application that turns a published online dataset into a useful business decision. Implement the analysis, a trained machine learning component, a deployed application, its MLOps lifecycle, and a presentation. The learned model must influence the application's output. Begin with the dataset and adapt the scope to the evidence it actually contains.

### Business question and application

**Decision:** Which transactions should be approved, challenged, or sent for review under a fixed review budget?

**Application to build:** A transaction explorer, risk explanations, threshold controls, review-capacity controls, and a chart showing the tradeoff between estimated fraud loss and legitimate approvals.

Write a short project charter defining the user, current baseline, primary business KPI, guardrails, prediction time, and measurable success criteria before selecting a model. Explain which decision the model changes and the costs of incorrect decisions.

### Published dataset and raw data

[ULB credit card fraud via OpenML 1597 v1](https://www.openml.org/search?type=data&id=1597). The original ARFF has been acquired under `data/raw/creditcard.arff`; the downloader verified the publisher checksum and the expected 284,807 rows/492 fraud labels and saved provenance/profile artifacts. Use source-derived sample inputs where display is permitted. Transaction economics and challenge effectiveness are explicit assumptions, not observed columns.

**Additional source/access notes:** Downloaded and verified: 284,807 rows, 492 frauds, 31 columns; publisher MD5 matched. Two days of data; no identity fields or observed review outcomes

Acquire a pinned source release into `data/raw/` and preserve the original bytes. Add a repeatable acquisition script and a manifest containing source URL, release/version, retrieval time, citation, terms, file sizes and SHA-256 checksums. Keep cleaned data in `data/interim/` and model-ready features/split assignments in `data/processed/`. Raw data stay out of Git unless redistribution is explicitly appropriate.

Audit labels, decision-time feature availability, duplicates, missingness, class balance, and temporal/group structure. If the candidate cannot support the task or intended use, find another suitable published dataset or narrow the question. Generated fixtures and economic scenarios are supplementary, clearly labeled artifacts.

### Machine learning component

**Required fitted component:** Calibrated fraud classifier.

Compare a simple rule and logistic regression with a boosted tree model. Include calibration and a cost-based decision policy.

Produce a reproducible training command and save the fitted preprocessing, model and decision policy together where appropriate. Compare the learned approach with a simple baseline. Pretrained models may support the solution, but a pretrained API call or prompt alone does not fulfill the trained-component requirement.

### Evaluation and business impact

PR-AUC, recall at a fixed false-positive rate, calibration, and expected loss under alternative cost assumptions. Use the time information available, document the short observation period, and avoid claims about long-term drift robustness.

Separate development/model selection from final evaluation. Fit transformations on training data only and compare candidate and incumbent on the same evaluation protocol. Save metrics, uncertainty/robustness checks, and representative failure cases with code/data/model identifiers.

Connect model metrics to the business KPI, with explicit denominators, comparator, population and period. Distinguish measured service performance, offline evaluation, simulated policy value and outcomes that require a real rollout. Never invent company impact or actual stakeholder buy-in.

### MLOps requirements

**Project-specific operational evidence:** Version features, calibration and approval policy together; monitor score distributions and delayed-label loss/precision; replay a drift scenario.

Implement and demonstrate:

1. Data/schema validation and versioned acquisition, features, splits and configuration.
2. Reproducible training with MLflow experiment tracking and saved metrics/artifacts.
3. Model registration with code/data lineage, exact versions, candidate/incumbent comparison and promotion gates.
4. Production inference with validated inputs and meaningful tests, including consistency with the evaluated model.
5. Containerized deployment and GitHub Actions CI/CD, with a pinned model/image, health check, inference smoke test and release identifier.
6. Service telemetry and task-specific input/prediction/quality monitoring, alert criteria and a response runbook. Identify replayed quality reports separately when real outcome labels are unavailable.
7. A runnable data-refresh/retraining workflow that evaluates the replacement before promotion.
8. A demonstrated rollback to the previous model/application release and a successful post-rollback inference check.

Use the smallest architecture that completes this lifecycle. Select real-time or batch inference according to the decision; shared managed services are acceptable, while the repository remains independently reproducible.

### Communication and portfolio deliverables

- A readable README with setup, actual results, architecture, limitations, and links to the released demo/case study/slides.
- Purpose-specific notebooks and short executive summaries documenting important experiments and decisions.
- A stakeholder decision memo covering the recommendation, evidence, tradeoffs and proposed rollout.
- A personal ownership statement and dated operating record identifying what was implemented, actual service usage, model changes, and incident/rollback exercises.
- A working public demo with source-derived examples where permitted, plus screenshots or a short recorded fallback.
- An original 10–12 slide presentation: editable `presentation/slides.pptx`, reviewed `presentation/slides.pdf`, speaker notes and underlying figures. Include model evaluation, business implications, MLOps architecture and operational evidence.
- A case study prepared for `tomasferrer.com`, linking the repository, demo and presentation.

**Presentation story:** Why the policy with the best business tradeoff may differ from the model with the best ranking metric.

**Distinctive evidence to include:** Temporal evaluation, error-cost sensitivity, a realistic review queue, an explicit deployment decision, and a dated operating record showing model update, monitoring and rollback. Apply the KPI contract above.

## Completion checklist

- [x] Published source acquired, checksum verified and provenance/profile recorded.
- [ ] Data audit, project charter, success criteria and evaluation protocol complete.
- [ ] Baseline and trained ML component evaluated; artifacts and failure analysis saved.
- [ ] Working application deployed with the exact evaluated model version.
- [ ] Tracking, registry, CI/CD, monitoring, retraining and rollback demonstrated.
- [ ] Stakeholder memo, ownership/operating record, case study and presentation complete.

## Current implementation

The [exploration notebook](notebooks/01-exploration.ipynb) covers schema and missingness, duplicates, class imbalance, transaction time and amount distributions, fraud incidence, PCA-feature distributions, and predictor correlations. Executed Seaborn grids and the Spearman heatmap are embedded in the notebook; standalone figures are in `reports/figures/`. These are descriptive findings, not trained-model evaluation or demonstrated business impact.

### Exploration environment

Use Python 3.12 and Poetry 2.5.1 (the setup version used here). From the repository root:

```bash
poetry env use python3.12
poetry install
poetry run python -m ipykernel install --sys-prefix --name payment-risk-workbench --display-name "Python 3.12 (Payment Risk Workbench)"
```

The environment lives in `.venv/`. In your notebook editor, use **Select Kernel → Python Environments** and select this repository's `.venv/bin/python`. The kernel is also installed inside the environment under the display name above; no global kernel registration is required.

`pyproject.toml` declares NumPy, pandas, Plotly, and the development dependencies ipykernel and nbformat (for Plotly notebook rendering). `poetry.lock` pins their resolved dependencies; commit both files, and use `poetry install` to reproduce the environment. Add statistical, modeling, and chart-export dependencies when those stages need them. Poetry currently manages dependencies in non-package mode; enable packaging when reusable source code is introduced.

### Data acquisition

The source dataset contains 284,807 transactions, 492 fraud labels and 31 columns. Acquisition, checksum verification and initial schema/population/range profiling are implemented; no model has been trained yet.

```bash
git clone https://github.com/tferrerd/01-payment-risk-workbench.git
cd 01-payment-risk-workbench
python3 scripts/download_data.py
python3 scripts/download_data.py --verify-only
```

Requires Python 3.9+ and `curl`; acquisition and initial checks use no third-party Python packages. See [data provenance](data/README.md), [manifest](data/manifest.json), and [dataset profile](reports/dataset-profile.json).
