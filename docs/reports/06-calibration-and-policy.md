# Calibration and approve/cancel policy

[Notebook 06](../../notebooks/06-calibration-and-policy.ipynb) keeps the trained Random Forest and LightGBM models fixed. Their scores were checked on 28,452 calibration payments (48 frauds). Identity, sigmoid, temperature, and isotonic mappings were compared using three label-stratified calibration folds, with exact V1–V28/Amount groups kept together. Each model's mapping was chosen by **out-of-fold Brier score**, then a selected non-identity mapping was fitted on the full calibration split. Scikit-learn's `FrozenEstimator` prevents the tree from being refitted. The 0.20 approve/cancel threshold follows the fixed 25% margin / 100% fraud-loss assumptions; it was not tuned. [Calibrator API](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)

| Model | Identity Brier | Sigmoid | Temperature | Isotonic | Mapping selected |
|---|---:|---:|---:|---:|---|
| Random Forest | 0.000387 | 0.000416 | 0.000421 | **0.000334** | Isotonic |
| LightGBM | **0.000577** | 0.000837 | 0.000578 | 0.000659 | Identity |

Only 48 frauds support this comparison; fold rankings are uncertain. Isotonic improves Random Forest's calibration-fold Brier, but its 3-fold log loss is worse than sigmoid's. LightGBM's selected identity mapping means **no post-hoc calibration was applied** to that model. Full-calibration fit diagnostics are in-sample and should not be read as independent validation.

The method-selected policies were then compared on the already-used selection split (42,632 payments, 76 frauds):

| Model / mapping | AP | Brier | Profit / 1,000 | Fraud canceled | Legitimate canceled |
|---|---:|---:|---:|---:|---:|
| Random Forest, raw | 0.8365 | 0.000466 | **21,841.44** | 60 | 8 |
| Random Forest, isotonic | 0.8071 | 0.000497 | 21,803.08 | 54 | 4 |
| LightGBM, raw / identity | **0.8421** | **0.000432** | 21,829.36 | 60 | 6 |

Isotonic Random Forest loses 38.36 source units per 1,000 selection payments compared with raw Random Forest. Its AP also falls because a stepwise mapping creates score ties. Among the mappings selected by calibration-fold Brier, LightGBM has higher selection profit. Across all observed development variants, raw Random Forest leads by 12.08 per 1,000 over raw LightGBM. The selection set has already guided model and feature comparisons; these differences are development evidence, not a final ranking.

**Decision:** do not freeze or register a champion yet. Stage 07 should quantify paired uncertainty on selection, then freeze a single model, mapping, and 0.20 policy before scoring evaluation once. The evaluation split has not been used for fitting, calibrator choice, policy tuning, or scoring in this stage. Earlier full-dataset label-aware EDA still limits any claim that evaluation is analytically untouched.
