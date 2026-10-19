# MLflow tracking

Stages 02–04 and 06–08 log parameters, selection metrics, fitted artifacts, and data/configuration/code checksums to MLflow. The current store is the project-local SQLite database `mlflow.db`, with artifacts under `mlruns/`. Both paths are excluded from Git. The database already contains the executed runs; a server is only needed to browse them or to receive runs over HTTP.

From the project root, start the local server and open `http://127.0.0.1:5050`:

```bash
poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5050 --workers 1
```

The training scripts use this same local database by default. To send future runs through the server, set `MLFLOW_TRACKING_URI=http://127.0.0.1:5050` in the terminal that launches the notebooks or scripts. To use a hosted MLflow server later, set `MLFLOW_TRACKING_URI` to its HTTPS address and configure its credentials using MLflow's standard environment variables. The scripts no longer override that setting.

Development runs and the frozen stage 07 evaluation have been logged. The historical stage 02–08 `.joblib` files remain run artifacts. The later inference delivery registers `payment-risk-low-margin` version 1 with alias `candidate`, including code, signature, artifact manifest, and inference-parity checks. No production `champion` is promoted. A hosted service also needs a durable metadata database, artifact storage, authentication, and hosting choices; none is configured for this project yet.

Stage 08 logs economic analysis of existing predictions under `08-low-margin-economics`, with explicit `selection_development` and `evaluation_regression` scope labels. It fits no new model and preserves the stage 07 evaluation.

Inference packaging is logged under `09-inference-candidate`. Repeat registration reuses a version with the same candidate-artifact digest and verifies its reload before assigning the development alias. The API serves the pinned local artifact independently of MLflow availability. See [inference instructions](inference.md).

The executed [Optuna experiment](reports/10-optuna-tuning.md) is logged under `10-optuna-low-margin`: one parent run and 40 child trials. It optimizes training-only grouped three-fold profit, then compares the fixed CV winner with the deployed artifact on selection. The parent stores the tuned artifact and lineage/results; child runs store parameters and fold/pooled metrics. The separate Optuna study database is under `data/processed/optuna/`. It does not replace the registered candidate or change the historical analysis lockfile.
