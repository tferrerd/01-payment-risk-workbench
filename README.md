# 01 — Payment Risk Workbench

Fraud-risk scoring, explainable predictions, and payment decisions driven by explicit economics.

[![Inference CI](https://img.shields.io/github/actions/workflow/status/tferrerd/01-payment-risk-workbench/inference-ci.yml?branch=main&style=flat-square&logo=githubactions&logoColor=white&label=Inference%20CI)](https://github.com/tferrerd/01-payment-risk-workbench/actions/workflows/inference-ci.yml)
[![Live demo on Render](https://img.shields.io/badge/Render-Live_demo-46E3B7?style=flat-square&logo=render&logoColor=white)](https://payment-risk-workbench.onrender.com)
[![Project case study](https://img.shields.io/badge/Portfolio-Case_study-172A2A?style=flat-square)](https://tomasferrer.com/projects/payment-risk-workbench/)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Poetry](https://img.shields.io/badge/Poetry-Dependency_management-60A5FA?style=flat-square&logo=poetry&logoColor=white)](poetry.lock)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-298729?style=flat-square)](requirements-inference.txt)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](requirements-inference.txt)
[![MLflow](https://img.shields.io/badge/MLflow-3.17.1-0194E2?style=flat-square&logo=mlflow&logoColor=white)](docs/mlflow-tracking.md)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.146.0-009688?style=flat-square&logo=fastapi&logoColor=white)](https://payment-risk-workbench.onrender.com/docs)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-0.54.0-4051B5?style=flat-square)](requirements-inference.txt)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.14.1-E92063?style=flat-square&logo=pydantic&logoColor=white)](requirements-inference.txt)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker&logoColor=white)](docs/container.md)
[![aiohappyeyeballs](https://img.shields.io/badge/aiohappyeyeballs-2.7.1-7C3AED?style=flat-square)](poetry.lock)

- **Application:** Fraud detection
- **Portfolio project:** 01 of 15
- **Status:** Public demo deployed on Render's Free plan; GitHub container checks and public inference smoke checks passed. A registered LightGBM development candidate powers CSV/API inference, live TreeSHAP explanations, and economic controls. Notebooks 01–08 preserve the historical-scenario evaluation and separate low-margin analysis. Fresh low-margin validation, probability calibration, and remaining operational evidence are still pending.

**[Open the live demo](https://payment-risk-workbench.onrender.com)** · [API reference](https://payment-risk-workbench.onrender.com/docs) · [Deployment evidence](reports/cloud-deployment-verification.json). Free instances can take about a minute to wake after inactivity.

Part of [Tomas Ferrer's DS/ML portfolio](https://tomasferrer.com/).

## Technology stack

| Area | Tools | Role in this project |
| --- | --- | --- |
| Environment | Python 3.12, Poetry | Reproducible analysis environment and locked dependencies. |
| Data and exploration | NumPy, pandas, Jupyter/ipykernel, Plotly, Seaborn, Matplotlib | Transaction analysis, notebooks, and visualizations. |
| Modeling and explanations | scikit-learn, LightGBM, joblib, native LightGBM TreeSHAP contributions | Baselines, fraud scoring, saved models, and individual prediction explanations. |
| Experiment tracking | MLflow, SQLite | Experiment metrics, artifacts, model registry, and lineage. |
| API and validation | FastAPI, Uvicorn, Pydantic | Validated prediction and explanation endpoints, ASGI serving, and OpenAPI documentation. |
| Interface | HTML, CSS, JavaScript | Transaction explorer, policy controls, profit charts, and CSV export. |
| Delivery | Docker, GitHub Actions, Render | Container packaging, automated checks, and the public demo. |
| Supporting networking dependencies | aiohttp, **aiohappyeyeballs 2.7.1** | Transitive dependencies in the MLflow development environment: `mlflow → aiohttp → aiohappyeyeballs`. |

FastAPI is used directly by the inference service and pinned in [`requirements-inference.txt`](requirements-inference.txt), alongside Uvicorn and Pydantic. **aiohappyeyeballs** supports asynchronous connection establishment through aiohttp; Poetry installs it transitively for the development stack. It is not included in the lightweight inference runtime. Analysis dependencies are declared in [`pyproject.toml`](pyproject.toml) and resolved in [`poetry.lock`](poetry.lock).

## Implementation framework

The project follows a business-to-production workflow adapted to this dataset and the portfolio lifecycle requirements. The [implementation plan](docs/implementation-plan.md) defines the project charter, seven analysis stages, random grouped evaluation protocol, and delivery sequence. Start with exploration and explicit transaction economics; carry the evaluated preprocessing, model, calibration, and policy into the application.

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
- [x] Data audit, project charter, success criteria and evaluation protocol complete.
- [x] Baseline and trained ML component evaluated; artifacts and failure analysis saved.
- [x] Public demo deployed with the exact registered development-candidate snapshot and prediction parity verified; fresh low-margin model-quality confirmation remains open.
- [ ] Tracking, registry, CI/CD, monitoring, retraining and rollback demonstrated.
- [ ] Stakeholder memo, ownership/operating record, case study and presentation complete.

## Run the development candidate

The reusable [inference package and instructions](docs/inference.md) provide CSV prediction and a local API. It uses the existing LightGBM model with the stage 08 0.3% margin scenario, and is registered as `payment-risk-low-margin` version 1 with the `candidate` alias. This is implementation validation with selection evidence; it is not a promoted production model.

```bash
poetry run python -m scripts.package_candidate --register
poetry run python -m payment_risk --input examples/payment-inputs.csv --output data/processed/example-predictions.csv
poetry run uvicorn payment_risk.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://127.0.0.1:8000/docs`. Use `examples/predict-request.json` for a source-derived request. See the [delivery report](docs/reports/09-inference-candidate.md) for parity and validation results.

The [Docker and CI instructions](docs/container.md) provide an isolated explorer/API container, a pinned candidate snapshot, release checksums, and a GitHub Actions build/test workflow. The explorer serves locally on port 8002 and [publicly on Render](https://payment-risk-workbench.onrender.com) without the local MLflow database; the original API-only container used port 8001.

### Transaction explorer

```bash
poetry run uvicorn payment_risk.explorer:app --host 127.0.0.1 --port 8002 --workers 1
```

Open `http://127.0.0.1:8002/` to search and inspect payments, view live model contributions, change cost assumptions, compare simulated sample profit, and export decisions. The 200 source-derived selection examples are deliberately enriched with fraud, so sample totals are illustrative. [Explorer usage](docs/explorer.md) explains the distinction between the fixed API policy and browser simulations. [Render deployment](docs/cloud-deployment.md) documents the free-tier configuration and the checks required before claiming a public deployment.

## Current implementation

The [exploration notebook](notebooks/01-exploration.ipynb) covers schema and missingness, duplicates, class imbalance, transaction time and amount distributions, fraud incidence, PCA-feature distributions, and predictor correlations. Executed Seaborn grids and the Spearman heatmap are embedded in the notebook; standalone figures are in `reports/figures/`. These are descriptive findings, not trained-model evaluation or demonstrated business impact.

The [random evaluation protocol](docs/evaluation-protocol.md) uses seed 42, targets 60%/15%/10%/15% for training/selection/calibration/evaluation, and keeps identical V1–V28/Amount patterns together. Time remains a predictor. Reproduce the audit and local row assignments with `poetry run python scripts/audit_splits.py`. The protocol documents prior EDA and experiment exposure; evaluation is not analytically untouched.

The executed [baseline notebook](notebooks/02-baseline.ipynb) compares approve-all, a training-p99 amount rule, and logistic regression including Time. See the [baseline report](docs/reports/02-baseline.md) and [approve/cancel profit assumptions](docs/cost-scenario.md). Approved legitimate payments earn 25% of Amount, approved fraud loses 100%, and canceled payments return zero. On selection, logistic regression achieves AP 0.7418 and about 98.99% of the perfect-label maximum profit versus 98.67% for approve-all. These are development scenario results, not realized business impact. The classifier has not been post-hoc calibrated or deployed.

The executed [feature engineering notebook](notebooks/03-feature-engineering.ipynb) audits the 30 available inputs and tests a 24-hour phase transform for Time. The [feature report](docs/reports/03-feature-engineering.md) records the decision to retain the existing inputs after a nearly tied selection comparison.

The executed [experiment notebook](notebooks/04-experiments.ipynb) compares fixed Random Forest and LightGBM models on the same selection payments. See the [comparison report](docs/reports/04-experiments.md). LightGBM has the best average precision (0.8421); Random Forest has the highest provisional profit at the fixed 0.20 threshold. These are development comparisons; later notebooks record calibration checks and frozen evaluation.

The executed [error-analysis notebook](notebooks/05-explanations-and-failures.ipynb) and [report](docs/reports/05-explanations-and-failures.md) trace expensive selection mistakes, fitted tree feature usage, and LightGBM contributions for representative payments. A few high-amount payments dominate simulated losses; stages 06–07 examine calibration and paired uncertainty.

The executed [calibration notebook](notebooks/06-calibration-and-policy.ipynb) and [report](docs/reports/06-calibration-and-policy.md) compare score mappings on grouped calibration folds. Isotonic improves Random Forest Brier within calibration folds but reduces its selection profit; LightGBM retains its raw scores. The 0.20 policy is fixed. Stage 07 records paired uncertainty, a candidate freeze, and final evaluation.

The executed [final notebook](notebooks/07-uncertainty-and-recommendation.ipynb) and [recommendation](docs/reports/07-uncertainty-and-recommendation.md) evaluate the [frozen candidate](config/candidate-freeze.json) on 42,694 payments (71 frauds). Raw Random Forest gains 112.50 simulated profit per 1,000 over approve-all (paired 95% interval: 42.07–201.26), with AP 0.7941. Its profit advantage over logistic regression is inconclusive. This supports an offline/demo prototype under the stated scenario, not a production or future-period claim. Evaluation is now a consumed benchmark; further model changes need fresh confirmation.

The additional [low-margin notebook](notebooks/08-low-margin-payments.ipynb) tests a user-proposed +0.3% legitimate margin / −100% fraud-loss scenario ([report](docs/reports/08-low-margin-payments.md)). The theoretical threshold falls to 0.2991% fraud probability, and LightGBM leads selection under these assumptions. This is a new development hypothesis; its reused evaluation comparison is regression evidence and does not replace stage 07’s frozen result.

```bash
poetry run python scripts/audit_splits.py
poetry run python scripts/run_baselines.py
poetry run python scripts/run_feature_engineering.py
poetry run python scripts/run_experiments.py
poetry run python scripts/analyze_failures.py
poetry run python scripts/run_calibration.py
poetry run python scripts/run_final_evaluation.py  # checks freeze and reuses cached evaluation
poetry run python scripts/run_low_margin_scenario.py
poetry run python -m unittest discover -s tests
poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5050 --workers 1
```

The scripts verify source and split checksums, fit on training rows, score selection rows, and record MLflow runs. Open `http://127.0.0.1:5050` after starting the local server. [Tracking setup](docs/mlflow-tracking.md) explains the existing SQLite store and how `MLFLOW_TRACKING_URI` directs future runs to a local or hosted server. Training model and tracking artifacts are excluded from Git; the compact inference candidate and its manifest are explicit deployment exceptions. Final evaluation requires the exact frozen snapshots; rerunning development scripts creates new lineage and must not overwrite the consumed evaluation.

### Exploration environment

Use Python 3.12 and Poetry 2.5.1 (the setup version used here). From the repository root:

```bash
poetry env use python3.12
poetry install
poetry run python -m ipykernel install --sys-prefix --name payment-risk-workbench --display-name "Python 3.12 (Payment Risk Workbench)"
```

The environment lives in `.venv/`. In your notebook editor, use **Select Kernel → Python Environments** and select this repository's `.venv/bin/python`. The kernel is also installed inside the environment under the display name above; no global kernel registration is required.

`pyproject.toml` declares the numerical/modeling dependencies and a development group for notebooks, Seaborn/Matplotlib, notebook execution, and MLflow. `poetry.lock` pins the resolved dependencies; commit both files, and use `poetry install` to reproduce the environment. Poetry retains the historical analysis environment in non-package mode. The importable `payment_risk` package runs from the project root; `requirements-inference.txt` records separate runtime pins without changing the stage 07 lockfile freeze.

### Data acquisition

The source dataset contains 284,807 transactions, 492 fraud labels and 31 columns. Acquisition, checksum verification and initial schema/population/range profiling can be reproduced separately from model training.

```bash
git clone https://github.com/tferrerd/01-payment-risk-workbench.git
cd 01-payment-risk-workbench
python3 scripts/download_data.py
python3 scripts/download_data.py --verify-only
```

Requires Python 3.9+ and `curl`; acquisition and initial checks use no third-party Python packages. See [data provenance](data/README.md), [manifest](data/manifest.json), and [dataset profile](reports/dataset-profile.json).
