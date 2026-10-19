# Selection-set error analysis

[Notebook 05](../../notebooks/05-explanations-and-failures.ipynb) audits the existing 42,632 selection payments (76 frauds) without refitting or reading calibration/evaluation labels. Opportunity loss is the approved fraud's full Amount or 25% of a canceled legitimate payment's Amount. Scores have not been calibrated.

| Model | Approved frauds | Fraud loss | Canceled legitimate | Forgone legitimate profit | Total opportunity loss |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 26 | 2,745.35 | 13 | 6,716.40 | 9,461.75 |
| Random Forest | 16 | 1,561.73 | 8 | 6,638.97 | **8,200.70** |
| LightGBM | 16 | 2,242.21 | 6 | 6,473.53 | 8,715.74 |

The trees both catch 59 frauds and both miss 15. They disagree on only two frauds: Random Forest catches a 720.38 payment that LightGBM approves; LightGBM catches a 39.90 payment that Random Forest approves. Random Forest therefore avoids 680.48 more fraud loss, while canceling legitimate payments worth 165.44 more in forgone profit. That produces its 515.04 total-profit lead on selection. This narrow difference is fragile with only 76 frauds.

One legitimate payment (source row 274771, Amount 25,691.16) is canceled by both trees. Its forgone profit is 6,422.79, accounting for **78.32%** of Random Forest's total opportunity loss and **73.69%** of LightGBM's. The five most expensive mistakes account for 96.18% and 96.86% of their respective loss totals. Keep this record in the primary result; use it to understand how the assumed economics interact with outliers, not as a reason to discard it.

The models use anonymized inputs differently. Random Forest's largest impurity importance is V17 (19.15% within that model); LightGBM's largest split-gain importance is V6 (44.48%). LightGBM also uses Time (4.61% of its gain). These are different internal measures, so their percentages are not directly comparable and do not explain individual predictions or identify business causes. The dataset provides no meanings for V1–V28, customer identities, or known local clock hours.

For representative payments, the notebook also shows LightGBM's native per-feature contributions to the raw log-odds score. All 30 contributions plus the base value were checked against each raw prediction. For the expensive legitimate cancellation at row 274771, V10 and V11 strongly raise the fitted score while V6 strongly lowers it; other features also contribute. The displayed top five alone do not sum to the score. These values describe the model's computation, not a causal reason for the payment or its true label. See the [LightGBM prediction API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html) for `pred_contrib`.

**Decision:** carry both trees into the separate calibration stage. Compare calibrated policies on the development protocol and add paired uncertainty before choosing a model. Evaluation remains reserved. The selection rows had prior analytical exposure through full-dataset EDA and earlier experiments.
