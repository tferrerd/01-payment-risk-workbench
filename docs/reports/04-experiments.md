# Fixed tree-model comparison

The same grouped random protocol, training rows (171,029; 297 frauds), and selection rows (42,632; 76 frauds) used in notebook 02 were used here. All fitted models include Time, V1–V28, and Amount. The exploratory 24-hour phase transform from [stage 03](03-feature-engineering.md) was not adopted. The Random Forest and LightGBM settings were fixed before this comparison; neither model was tuned or calibrated. Evaluation was not scored.

| Model | Average precision | Brier score | Profit / 1,000 | Fraud canceled | Legitimate canceled |
|---|---:|---:|---:|---:|---:|
| Approve all | 0.0018 | 0.001780 | 21,740.96 | 0.00% | 0.0000% |
| Amount rule | 0.0031 | — | 17,299.28 | 2.63% | 0.9517% |
| Logistic regression | 0.7418 | 0.000810 | 21,811.86 | 65.79% | 0.0305% |
| Random Forest | 0.8365 | 0.000466 | **21,841.44** | 78.95% | 0.0188% |
| LightGBM | **0.8421** | **0.000432** | 21,829.36 | 78.95% | **0.0141%** |

Profit uses the illustrative approve/cancel economics and the same probability threshold of 0.20 for all three learned models. Random Forest leads by 29.58 source units per 1,000 payments over logistic regression; LightGBM leads the ranking and Brier metrics. These small profit differences could change with calibration or sampling variation. Carry both tree models into calibration and paired uncertainty analysis before selecting a policy. No final evaluation or production claim follows from this selection comparison.

Reproduce via `poetry run python scripts/run_experiments.py` or [notebook 04](../../notebooks/04-experiments.ipynb). The report, predictions, fitted artifacts, and local MLflow runs record settings and source, split, code, and lockfile lineage. The test population has prior analytical exposure through full-dataset EDA and earlier experiments.
