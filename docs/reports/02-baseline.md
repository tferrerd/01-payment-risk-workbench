# Initial baseline comparison — random selection set

Protocol: `grouped-random-v1`, seed 42. Fit on 171,029 training payments (297 frauds); compare on 42,632 selection payments (76 frauds). Time is available and included in logistic regression. Split allocation is random and label-stratified at the exact V1–V28/Amount pattern-group level, targeting 60%/15%/10%/15%. Calibration and evaluation are not scored.

## Assumptions and comparators

Approve legitimate earns 25% of Amount; approve fraud loses 100%; cancel returns zero. Logistic regression cancels above fraud probability 0.20. It uses standardized V1–V28 and Time, plus standardized log(1 + Amount), with preprocessing fitted on training only. The amount rule cancels amounts at or above training p99 (1,006.908 source units). Every record is retained.

| Comparator | Average precision | Total profit | Profit / 1,000 | Maximum profit captured | Fraud recall |
|---|---:|---:|---:|---:|---:|
| approve_all | 0.0018 | 926,860.78 | 21,740.96 | 98.6710% | 0.00% |
| amount_rule | 0.0031 | 737,502.73 | 17,299.28 | 78.5124% | 2.63% |
| logistic_regression | 0.7418 | 929,883.38 | 21,811.86 | 98.9927% | 65.79% |

Profit is in source Amount units under the assumed scenario. Logistic regression improves selection profit by 3,022.59 over approve-all and cancels 50 of 76 frauds and 13 legitimate payments. The amount rule sacrifices substantial legitimate profit. Retain logistic regression as the learned reference and compare a boosted tree next.

Probability calibration, paired uncertainty, economic sensitivity, and final evaluation remain pending. The 0.1% false-positive diagnostic chooses its threshold on selection itself. Full-dataset EDA and earlier experiments preceded this revised protocol; evaluation is not analytically untouched. These are development results within the same two-day population, not measured business impact.

Reproduce with `poetry run python scripts/run_baselines.py`, or execute notebook 02. Each run creates three local MLflow runs, saved model/policy artifacts, selection predictions, and results with source/config/code/lockfile lineage. Notebook 02 executed without cell errors. The split isolation and policy arithmetic tests passed.
