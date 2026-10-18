# Temporal evaluation protocol — temporal-v1

Recorded 2026-10-18 before model fitting. The versioned configuration is [evaluation.json](../config/evaluation.json). Run `poetry run python scripts/audit_splits.py` to regenerate the [audit report](../reports/duplicate-split-audit.json) and local split assignments in `data/processed/split-assignments.csv`.

## Boundaries and roles

Time is seconds since the first transaction. Boundaries use elapsed hours, not local clock time. Every observation at a boundary enters the later window, so tied timestamps stay together. Source row IDs are zero-based across the five CSV parts; source data-row numbers are one-based within a part, excluding its header.

| Split | Elapsed hours | Transactions | Frauds | Role |
|---|---|---:|---:|---|
| train | [0, 24) | 144,786 | 281 | Fit initial transformations and candidate classifiers. |
| selection | [24, 32) | 17,739 | 75 | Compare candidates and select hyperparameters using earlier training data. |
| calibration | [32, 40) | 62,340 | 59 | Fit a preselected calibration method and develop the decision policy for the fixed candidate. |
| evaluation | [40, end] | 59,942 | 77 | Evaluate the frozen model, calibrator, and policy; do not tune from these results. |

These are the boundaries proposed before model experiments; the population audit did not change them. All windows contain both classes. The calibration window has only 59 frauds: calibration and threshold estimates will be uncertain. Use this as a fixed development protocol, not evidence of a workable eight-hour retraining cadence. Actual fraud-label maturation times are unavailable.

The initial model is fitted on train only. If a classifier is refitted on train plus selection, do that before calibrator fitting; generate fresh calibration scores from that exact refitted artifact. Once fitted, freeze the classifier and calibrator together. Calibration-method or policy searches on the calibration window are development decisions, and must be disclosed. Do not refit any component on evaluation labels before scoring the frozen release.

## Repeated records and patterns

The audit compares exact parsed numeric values; it does not perform approximate record linkage.

- **All columns:** 1,081 repeated rows beyond first occurrences; none span split boundaries.
- **All predictors, including Time:** the same 1,081 excess repeats; no conflicting labels and no cross-split repeats.
- **V1–V28 plus Amount, excluding Time:** 9,144 excess repeats, 1,128 groups spanning splits, and no conflicting-label groups. These are repeated feature patterns, not proven duplicates of the same real transaction.
- **Earlier-pattern overlap:** 353 calibration rows and 1,744 evaluation rows repeat a non-Time predictor vector from an earlier split. All 1,744 evaluation overlaps are legitimate in this snapshot.

Retain all records for the primary evaluation. Report a companion evaluation excluding evaluation rows whose V1–V28/Amount vector appeared in any earlier split: **58,198 rows and 77 frauds**. Report its changed population and denominator explicitly. Keep both candidate and incumbent on the same rows. This sensitivity does not remove duplicates internal to evaluation or establish independence between customers; no customer IDs exist.

Persist row-level overlap flags and source lineage so later code uses the same masks. Do not group repeated patterns by moving future rows into training. If the training duplicate treatment changes, document it as an experiment; do not silently alter the frozen evaluation population.

## Evaluation contract and limitations

Report average precision (the explicitly named PR summary), recall at a development-selected threshold targeting 0.1% false positives, the realized false-positive rate, Brier score, reliability plots, and simulated policy cost. Include record counts, fraud counts, time windows, comparator, costs, and capacity in each report. Choose economics before policy selection; business assumptions remain to be specified.

**Exposure disclosure:** full-dataset label-aware plots were produced before this protocol was frozen. The evaluation window is therefore reserved from subsequent fitting and tuning, but cannot be described as an untouched analytical holdout. A newly acquired later labeled dataset is needed for genuinely fresh confirmation. Do not erase this limitation after a favorable result.

Upstream PCA fitting provenance and decision-time availability cannot be independently verified. The dataset has two days of observations, few frauds, and unknown label delays. Metrics will not establish long-term robustness or actual business impact. Once evaluation results influence a change, that window becomes a regression benchmark for the revised model, rather than a fresh final test.
