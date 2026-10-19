# Random evaluation protocol — grouped-random-v1

Recorded 2026-10-18. Predict fraud for individual payments using Time, V1–V28, and Amount. Time is an available predictor; it does not determine split assignment.

## Allocation

Use seed 42 to randomly allocate exact V1–V28/Amount pattern groups, stratifying by their fraud label. Target proportions are 60% training, 15% model selection, 10% calibration, and 15% evaluation. Whole groups are indivisible, so payment proportions and fraud rates are approximate. Conflicting labels within a group are rejected.

The prediction unit is a payment. The allocation unit groups identical observed patterns to prevent their repetition across sets; these groups do not identify actual customers or cards. Keep all rows. Each split spans the dataset's elapsed-time range.

| Split | Payments | Frauds | Fraud rate |
|---|---:|---:|---:|
| train | 171,029 | 297 | 0.1737% |
| selection | 42,632 | 76 | 0.1783% |
| calibration | 28,452 | 48 | 0.1687% |
| evaluation | 42,694 | 71 | 0.1663% |

See the [configuration](../config/evaluation.json) and [audit](../reports/duplicate-split-audit.json). Reproduce with `poetry run python scripts/audit_splits.py`. Assignments retain source row IDs, file lineage, Time, and pattern group IDs.

## Use of the sets

Fit preprocessing and classifiers on training only. Compare candidates on selection. For the remaining trees, compare identity, sigmoid, temperature, and isotonic score mappings by three grouped calibration folds; fit a selected non-identity mapping on the full calibration set. Compare the resulting fixed-0.20 policies on the reused selection set and quantify paired uncertainty there. Then freeze one model, mapping, and policy before evaluation. Identity means no post-hoc calibration. Any future classifier refit would require fresh calibration scores from that exact refitted artifact. Do not tune using evaluation labels.

Report average precision, ROC AUC, Brier score, reliability plots, fraud recall, legitimate cancellation rate, and simulated profit under the [approve/cancel scenario](cost-scenario.md). The probability policy cancels above 0.20. A threshold chosen on selection to target 0.1% false positives is a development diagnostic, not an independently evaluated operating point.

## Duplicate audit and limitations

There are 1,081 excess exact repeats and 9,144 excess V1–V28/Amount repeats when Time is ignored. There are no conflicting labels or cross-split pattern groups. Every evaluation pattern is absent from training, selection, and calibration; a separate unseen-pattern subset is unnecessary.

Full-dataset label-aware EDA and earlier chronological baseline experiments preceded this revised protocol. Evaluation is reserved from subsequent fitting and tuning but is not an untouched analytical holdout. This random comparison estimates performance on held-out payments from the same two-day population. Future-period performance, unseen-customer performance, and production impact require separate evidence. Upstream PCA fitting provenance and fraud-label delays are unavailable.

For paired uncertainty estimates, resample pattern groups together and compare policies on the same sampled payments. Record the resampling seed and interval method. The short dataset cannot establish long-term drift robustness.

## Evaluation status — candidate-v1

Stage 07 froze raw Random Forest with identity mapping and the fixed 0.20 economic threshold before scoring evaluation. The [manifest](../config/candidate-freeze.json) binds the candidate, comparator snapshots, lineage, bootstrap plan, and economic sensitivities. The [final report](reports/07-uncertainty-and-recommendation.md) records all 42,694 evaluation payments and the primary paired profit interval. Evaluation is now consumed; result-informed changes must treat it as a regression benchmark and obtain new labeled data for fresh confirmation. The cached evaluator refuses changed frozen inputs.

Stage 08 adds a later user-proposed 0.3% margin scenario. It changes neither the split nor the historical freeze. Existing evaluation scores are only repriced as a consumed regression benchmark; they do not provide a new independent final test for this hypothesis.
