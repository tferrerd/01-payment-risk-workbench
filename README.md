# Payment Risk Workbench

Project 01 of Tomas Ferrer's DS/ML portfolio. **Current status: source data acquired and verified; model/application/MLOps implementation is next.**

The project will turn a trained fraud model into transaction decisions, with explicit approval/loss tradeoffs, evaluation, deployment, monitoring, retraining and rollback.

## Start from the published data

Use the [ULB credit card fraud dataset, OpenML 1597 version 1](https://www.openml.org/search?type=data&id=1597). The untouched downloaded ARFF belongs in `data/raw/creditcard.arff`. Cleaned data and features will be written separately.

```bash
git clone https://github.com/tferrerd/payment-risk-workbench.git
cd payment-risk-workbench
python3 scripts/download_data.py
python3 scripts/download_data.py --verify-only
```

Requires Python 3.9+ and `curl`; no third-party Python dependencies are needed for acquisition or the initial checks. The downloader verifies the publisher MD5, records SHA-256 and source metadata, validates the schema/population, and preserves an existing raw file. It performs no model training.

The dataset contains 284,807 transactions, 492 labeled frauds, 28 anonymized PCA features, elapsed time and amount. It covers only two days. It does not contain identity attributes, customer review outcomes, challenge outcomes or measured business margins. Any business-policy assumptions will be explicit.

See [data provenance](data/README.md), the [acquisition manifest](data/manifest.json), and the [initial data profile](reports/dataset-profile.json). Raw data are excluded from Git; the metadata, manifest and download instructions provide reproducibility.

## Planned development

1. Audit the data and define business metrics, assumptions and temporal validation.
2. Compare baseline and trained fraud models; evaluate calibration and transaction policies.
3. Package inference and build the interactive transaction-review application.
4. Add experiment tracking, model registration, gated deployment, monitoring, retraining and rollback.
5. Publish the case study and presentation with measured results and limitations.

Author: [Tomas Ferrer](https://tomasferrer.com/).
