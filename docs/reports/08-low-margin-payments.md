# Low-margin payment scenario

[Notebook 08](../../notebooks/08-low-margin-payments.ipynb) adds the user's proposed economics in [low-margin-payments-v1](../../config/cost-scenario-low-margin.json): approve legitimate earns **0.3% of Amount**, approve fraud loses **100%**, and cancel earns zero. The 0.3% is an assumed retained contribution margin, not a verified industry fee or observed dataset outcome. An Amount of 100 earns 0.30 if legitimate but loses 100 if fraudulent; one approved fraud offsets about 333 equal-sized legitimate approvals.

Expected approval profit is `Amount × [0.003 × (1 − p) − p]`. The break-even fraud probability is `0.003 / 1.003 = 0.0029910269`, or **0.2991%**. This is a probability threshold, not 0.2991% of Amount. Ties and zero amounts approve. The fraud-loss fraction stays at 100%; the smaller legitimate upside makes fraud much more expensive relative to earned margin.

## Selection development results

Only existing scores are used; no models or calibrators are refitted and no score threshold is optimized using labels. The original 20% policy is repriced under the new payoffs so the difference between changing economics and changing actions is visible.

| Policy | Profit / 1,000 | Fraud canceled / 76 | Legitimate canceled |
|---|---:|---:|---:|
| Approve all | −28.43 | 0 | 0 |
| Cancel all positive amounts | 0.00 | 72 | 42,297 |
| Training-p99 amount rule | −18.15 | 2 | 405 |
| RF, original 20% threshold | 225.90 | 60 | 8 |
| RF, new 0.2991% threshold | 208.33 | 65 | 1,377 |
| Logistic, new threshold | 227.64 | 64 | 582 |
| LightGBM, new threshold | **238.67** | 62 | 90 |
| Isotonic RF, new threshold | 223.29 | 60 | 34 |

LightGBM leads selection under this new economic hypothesis. Lowering raw RF's threshold catches five additional frauds but increases legitimate cancellations from 8 to 1,377. Approved fraud loss falls from 1,561.73 to 941.44, while forgone legitimate margin rises from 79.67 to 1,449.17. Its profit therefore falls by 17.57 per 1,000, with a paired interval of −35.30 to +11.99: inconclusive. The RF-new-threshold minus LightGBM comparison is −30.34 per 1,000 (−35.42 to −25.67), conditional on these fixed development scores and observed pattern groups. Repeated development/model comparison is not independent confirmation.

Zero-amount payments remain approved by the economic policies because both actions have zero payoff; this explains why cancel-all-positive does not cancel every fraud label.

## Consumed evaluation benchmark: regression only

The evaluation population was already scored in stage 07 before this scenario was proposed. Repricing its cached RF and logistic scores is a reused regression/sensitivity comparison, **not a new final evaluation**. LightGBM and isotonic RF are shown only on selection because they were not scored in stage 07's frozen evaluation. The historical frozen candidate is not replaced.

| Policy | Profit / 1,000 | Fraud canceled / 71 | Legitimate canceled |
|---|---:|---:|---:|
| Approve all | 56.47 | 0 | 0 |
| Cancel all positive amounts | 0.00 | 68 | 42,357 |
| Amount rule | 73.38 | 2 | 477 |
| RF, original threshold | 176.78 | 54 | 10 |
| RF, new threshold | **221.63** | 61 | 1,481 |
| Logistic, new threshold | 164.06 | 57 | 558 |

Here the lower RF threshold gains 44.85 per 1,000 over its old policy, with an interval of −34.70 to +165.08: inconclusive. Its gain over approve-all is 165.16 (48.93 to 313.12). These intervals use the same 2,000-replicate paired pattern-group bootstrap and remain conditional descriptions of a consumed benchmark, not fresh confirmation of the new scenario.

**Decision:** the economic hypothesis materially changes profit totals, the acceptable-risk threshold, and model ranking. Treat LightGBM as a development lead for further low-margin policy work, without promoting a new candidate from these reused data. Calibrate and validate risk near 0.3%, and obtain new labeled evidence before a new final claim. The original 25% scenario, candidate freeze, and final report are preserved. Results are logged under MLflow experiment `08-low-margin-economics`, with scope labels distinguishing selection development from evaluation regression.
