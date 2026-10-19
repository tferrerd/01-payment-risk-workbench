# Development-candidate inference

The reusable `payment_risk` package wraps the existing LightGBM model with the stage 08 economics: approved legitimate payments earn 0.3% of Amount, approved fraud loses 100%, and cancellation returns zero. The score threshold is `0.003 / 1.003`, about 0.2991% fraud probability. Ties and zero amounts approve. The model is not refitted and its scores have not been post-hoc calibrated. It is a development candidate with selection evidence, not a newly final-tested or promoted production model.

## Package and register

Run commands from the project root using its Python 3.12 Poetry environment:

```bash
poetry run python -m scripts.package_candidate --register
```

This verifies every saved LightGBM selection score and stage 08 decision, saves the model, economics, schema, and metadata under `models/candidates/`, and checks the serialized artifact after reload. The sidecar JSON binds its SHA-256 checksum. Repeating the command reuses matching artifacts and model versions; different lineage requires a new candidate version instead of overwriting v1. The root analytics lockfile is preserved because stage 07's freeze binds it. `requirements-inference.txt` records the runtime dependency pins separately; the current Poetry environment already contains them.

The MLflow model is `payment-risk-low-margin`, version 1, alias `candidate`. A pyfunc model includes the code, candidate artifact, manifest, input/output signature, example, runtime requirements, and validation record. Its prediction output was checked after registry reload. No `champion` alias is assigned. The registry uses `MLFLOW_TRACKING_URI` when configured and otherwise the project's existing local database.

## CSV batch prediction

```bash
poetry run python -m payment_risk \
  --input examples/payment-inputs.csv \
  --output data/processed/example-predictions.csv
```

The CSV needs `Time`, `Amount`, and `V1` through `V28`. Field order does not matter. Values must be complete finite numbers; Time and Amount must be nonnegative. Other input columns, such as Class or a local row identifier, are retained in the output but never passed to the model. Duplicate column names, conflicting output fields, and overwriting the input or an existing output are rejected.

The output adds fraud probability, approve/cancel decision, modeled approval/policy profit, model ID, scenario, development status, and the explicit `probability_calibrated=false` field. Modeled profit uses the model score and assumed economics; it is not observed profit or a guarantee of realized value.

## Local API

```bash
poetry run uvicorn payment_risk.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://127.0.0.1:8000/docs`. The artifact is verified and loaded once at startup. `PAYMENT_RISK_MODEL_PATH` can select a separately versioned candidate artifact. The module does not import MLflow or connect to the registry during serving.

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/predict-request.json
```

`POST /predict` accepts `{"transactions": [...]}` with 1–1,000 records. Each record requires exactly the 30 numeric features. Unknown fields, strings, booleans, missing values, nonfinite values, and negative Time/Amount return JSON 422. Validation errors do not echo transaction values. `/health` reports the model ID, scenario, threshold, and development status.

## Verification and scope

```bash
poetry run python -m unittest discover -s tests
# With the local API running:
poetry run python -m scripts.verify_inference
```

The smoke check compares package, CSV, HTTP, and registered-model predictions on examples taken only from selection, exercises invalid HTTP requests, checks the OpenAPI schema, and verifies that historical frozen inputs remain unchanged. Reports are saved in `reports/candidate-package.json` and `reports/inference-verification.json`. Training artifacts, prediction outputs, and the local registry are excluded from Git. The deployment snapshot and its manifest are explicit exceptions, so a clean checkout can build the candidate container without retraining.

The [container and CI instructions](container.md) describe the pinned snapshot, local container, release checksums, and GitHub Actions workflow. Probability reliability near the low-margin cutoff and fresh labeled validation under these economics remain open before production promotion. The original stage 07 Random Forest and 25%-margin evaluation are retained as historical evidence.
