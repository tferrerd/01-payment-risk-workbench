# Frozen evaluation and recommendation

[Notebook 07](../../notebooks/07-uncertainty-and-recommendation.ipynb) completes the analysis with a paired profit comparison and a single frozen evaluation. Raw Random Forest was selected by the highest primary-scenario selection profit. The [candidate-v1 manifest](../../config/candidate-freeze.json) was written before evaluation scoring and binds the model snapshot, identity mapping, 30 inputs including Time, 0.20 policy, comparator artifacts, dataset/configuration hashes, and sensitivity plan. Identity means that no post-hoc calibrator was adopted. This is an offline candidate, not a production champion.

## Selection uncertainty before the freeze

Resample whole V1–V28/Amount pattern groups with replacement, with the same sampled groups for every policy. The bootstrap uses 2,000 replicates, seed 42, and percentile 95% intervals. Each replicate divides by its sampled payment count, preserving repeated payments within groups. Selection has 42,632 payments in 41,349 groups. Its Random Forest gain over approve-all was +100.48 per 1,000, with an interval of **−299.61 to +365.59**: inconclusive. Its gain over LightGBM was also inconclusive (+12.08; −12.57 to +50.57). The candidate was frozen as the observed selection profit leader without a claim of established superiority.

## Evaluation on all reserved payments

The fixed candidate and pre-existing comparators were scored on all 42,694 evaluation payments, including 71 frauds. No alternative tree or calibration search was run on evaluation. All records and repeated patterns were retained.

| Policy | AP | Brier | Profit / 1,000 | Fraud canceled | Legitimate canceled |
|---|---:|---:|---:|---:|---:|
| Random Forest, raw | 0.7941 | 0.000522 | **22,573.90** | 54 / 71 | 10 |
| Approve all | 0.0017 | 0.001660 | 22,461.40 | 0 / 71 | 0 |
| Training-p99 amount rule | 0.0020 | — | 17,420.22 | 2 / 71 | 477 |
| Logistic regression | 0.6632 | 0.000816 | 22,544.91 | 43 / 71 | 13 |

Random Forest earns 963,769.96 simulated total profit, or 99.55% of the perfect-label maximum under the primary 25% margin / 100% fraud-loss scenario. Its fraud cancellation recall is 76.06%; the legitimate cancellation rate is 0.0235%.

| Candidate minus comparator | Profit gain / 1,000 | Paired 95% interval | Conclusion |
|---|---:|---:|---|
| Approve all (primary) | +112.50 | +42.07 to +201.26 | Positive under this scenario |
| Amount rule (secondary) | +5,153.67 | +4,566.82 to +5,752.08 | Positive |
| Logistic regression (secondary) | +28.99 | −4.38 to +86.46 | Inconclusive |

Evaluation contains 41,351 pattern groups. The intervals condition on the fixed fitted artifacts and observed source population. They exclude model-training uncertainty, earlier model selection, unseen entities, and future drift; pattern groups are not real customer identities. The five most expensive errors account for 91.79% of opportunity loss, so monetary tails still matter.

The separate 0.1% false-positive diagnostic uses the threshold fixed on selection (0.3054167). Evaluation only measures it: recall 77.46%, false-positive rate 0.0188%. This diagnostic flags by score alone, including zero-amount frauds. The economic policy approves zero amounts because both actions have zero modeled payoff; its recall is therefore a different quantity. No evaluation-derived threshold is used.

## Economic sensitivity and decision

All nine configured margin/fraud-loss combinations were recorded before evaluation. Candidate profit minus approve-all remains positive at the point-estimate level both with frozen primary actions and with algebraic scenario thresholds on fixed scores. These are descriptive sensitivities, not nine independently confirmed effects, and the raw scores make probability-based scenario policies provisional. The primary assumptions remain 25% margin / 100% fraud loss.

**Recommendation:** proceed to a versioned offline/demo prototype with the frozen artifact. The primary evaluation supports a modeled gain over approve-all, but not a demonstrated profit advantage over logistic regression. Do not claim a production effect or validated probability calibration. Full-dataset label-aware EDA and earlier experiments preceded the protocol; this is not an analytically untouched holdout. The two-day dataset cannot establish long-term performance.

The final evaluation is now consumed. Subsequent changes influenced by these results must treat it as a regression benchmark; fresh confirmation requires new labeled data. `scripts/run_final_evaluation.py` checks the freeze and reuses its cached result rather than scoring repeatedly. The evaluation is logged in MLflow experiment `07-frozen-evaluation`; local model snapshots and row predictions remain excluded from Git. Restore the exact snapshots for reproduction, or record newly trained artifacts as a new experiment rather than overwriting this result.
