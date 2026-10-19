# MLflow tracking

Stages 02–04 and 06–08 log parameters, selection metrics, fitted artifacts, and data/configuration/code checksums to MLflow. A private DagsHub project provides hosted tracking and artifact storage. The original project-local SQLite database `mlflow.db` and artifacts under `mlruns/` remain available as a backup and local default. Both paths are excluded from Git.

From the project root, start the local server and open `http://127.0.0.1:5050`:

```bash
poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5050 --workers 1
```

The training scripts use this same local database by default. To send future runs through the local server, set `MLFLOW_TRACKING_URI=http://127.0.0.1:5050` in the terminal that launches the notebooks or scripts. For hosted tracking, load the per-command configuration below. The scripts respect MLflow's standard URI and credential environment variables.

Development runs and the frozen stage 07 evaluation have been logged. The historical stage 02–08 `.joblib` files remain run artifacts. The later inference delivery registers `payment-risk-low-margin` version 1 with alias `candidate`, including code, signature, artifact manifest, and inference-parity checks. No production `champion` is promoted. DagsHub supplies the hosted metadata store, artifact storage, and authentication.

Stage 08 logs economic analysis of existing predictions under `08-low-margin-economics`, with explicit `selection_development` and `evaluation_regression` scope labels. It fits no new model and preserves the stage 07 evaluation.

Inference packaging is logged under `09-inference-candidate`. Repeat registration reuses a version with the same candidate-artifact digest and verifies its reload before assigning the development alias. The API serves the pinned local artifact independently of MLflow availability. See [inference instructions](inference.md).

The executed [Optuna experiment](reports/10-optuna-tuning.md) is logged under `10-optuna-low-margin`: one parent run and 40 child trials. It optimizes training-only grouped three-fold profit, then compares the fixed CV winner with the deployed artifact on selection. The parent stores the tuned artifact and lineage/results; child runs store parameters and fold/pooled metrics. The separate Optuna study database is under `data/processed/optuna/`. It does not replace the registered candidate or change the historical analysis lockfile.

## Hosted tracking setup

The private DagsHub project is [tferrerd/01-payment-risk-workbench-mlflow](https://dagshub.com/tferrerd/01-payment-risk-workbench-mlflow). Sign in with an authorized account to view its experiments and models; it is separate from the public GitHub source repository.
Its tracking URI is `https://dagshub.com/tferrerd/01-payment-risk-workbench-mlflow.mlflow`.
All 93 historical runs are uploaded. The [hosted verification report](../reports/mlflow-hosting-verification.json) confirms parameters, 2,021 latest metric records (values, steps, timestamps), all 40 Optuna parent links, and `payment-risk-low-margin` version 1 with alias `candidate`. The project is private and anonymous access is denied.

The exhaustive remote history/artifact audit was stopped to meet the requested execution-time limit. Full remote historical artifact checksums and remote candidate reload parity remain unchecked. The complete local migration rehearsal passed those checks. This hosting migration does not change the model's validation status.

From the project root, save a DagsHub access token with the hidden prompt:

```bash
.venv/bin/python scripts/configure_hosted_tracking.py
```

This writes `.env.mlflow`, excluded from Git and readable only by its owner. Load it for each training command with `python-dotenv`, already in the development environment:

```bash
.venv/bin/dotenv -f .env.mlflow run -- .venv/bin/python scripts/run_optuna.py --help
```

Use the same prefix for a training script or Jupyter launch when hosted logging is desired. Commands without that prefix retain the local SQLite default. Credentials belong on development machines; the Render inference app serves its pinned artifact without a tracking-server dependency.

## Migration verification

Before remote migration, the SQLite database and `mlruns/` were backed up under ignored `data/interim/mlflow-migration-backup-20261019/`. The official [MLflow Export Import](https://github.com/mlflow/mlflow-export-import) utility is pinned in `requirements-migration.txt` as a separate development overlay. Install it into the existing development environment with `pip install --no-deps -r requirements-migration.txt`; its Databricks/cloud extras are unnecessary for this migration. The historical Poetry lockfile remains unchanged.

The local export is under `data/interim/mlflow-export-20261019/`. A disposable local-server import verified 93 finished runs across eight nonempty experiments, 2,021 complete metric histories (values, steps, timestamps), 146 run artifact files, and all 40 nested Optuna links. It also verified the registered candidate's artifact bytes and predictions on the checked-in example payments.

The hosted transfer uses a resumable helper that creates experiment names sequentially, writes source IDs when creating runs, batches parameters and metrics, and reuses completed runs. It requires a private DagsHub repository and preserves the export format's source tags. An interrupted upstream run with missing source tags is recovered only when its parameters, metrics, and experiment match exactly one unused source run. Automatic HTTP retries are disabled during import because a retried creation request can create a second run; explicit resume uses the source-ID tags instead. Verification rejects duplicate source IDs.

```bash
.venv/bin/dotenv -f .env.mlflow run -- .venv/bin/python -m scripts.import_tracking_history \
  --export data/interim/mlflow-export-20261019 --workers 4
```

The upstream bulk importer can omit the MLflow 3 registered version. The verification helper can create that missing version from the imported logged model and restore its `candidate` alias:

```bash
.venv/bin/dotenv -f .env.mlflow run -- .venv/bin/python -m scripts.verify_tracking_migration \
  --export data/interim/mlflow-export-20261019 \
  --output data/interim/mlflow-hosted-verification.json --restore-registry
```

Run and model IDs change during import. Original run IDs and start/end times remain in `mlflow_exim.run_info.*` tags; the destination's native run timestamps represent migration time. Parent links point to destination runs. Model payload files must match byte for byte; `MLmodel` metadata legitimately changes to reference the imported IDs. Never rerun a full import against a partially populated destination without checking for duplicate source IDs first.
