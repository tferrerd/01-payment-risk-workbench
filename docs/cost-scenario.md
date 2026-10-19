# Historical approve/cancel profit scenario — stages 02–07

Scenario: `approve-cancel-profit-v1`, specified by the user on 2026-10-18 and recorded in [cost-scenario.json](../config/cost-scenario.json). This replaces the earlier exploratory challenge-fee/effectiveness assumptions. Baseline model settings are fixed; the active allocation is grouped-random-v1. The frozen final evaluation is recorded in stage 07.

## Profit per transaction

Let `A` be transaction amount, in the source Amount units.

| Action | Legitimate transaction | Fraudulent transaction |
|---|---:|---:|
| Approve | `+0.25 × A` | `−1.00 × A` |
| Cancel | `0` | `0` |

There are no cancellation fees, challenge fees, partial fraud prevention, or abandonment effects in this model. Review/challenge actions require separate economics if introduced later. These are assumed economics applied to historical labels, not measured service outcomes.

## Decision threshold

For fraud probability `p`, expected approval profit is:

`A × [0.25 × (1 − p) − p] = A × (0.25 − 1.25p)`.

Cancellation earns zero. For positive amounts, approve below `p = 0.20` and cancel above it. Both actions break even at exactly 0.20; ties favor approval. Zero-amount transactions have zero modeled profit for either action, and the policy approves them. The threshold assumes the model outputs meaningful probabilities; miscalibration can reduce policy profit.

## Reporting

Primary economic metrics are **simulated total profit**, **profit per 1,000 transactions**, and **profit as a percentage of the perfect-label maximum**. The maximum is `0.25 × sum(legitimate Amount)`: approve all legitimate payments and cancel all frauds. It is an oracle comparator, not an achievable production claim.

Keep a secondary opportunity-loss measure to explain errors: approving a fraud loses `A`, while canceling a legitimate payment forfeits `0.25A`. Cancellation itself has zero direct cash loss; forfeited profit is an opportunity cost. Profit equals maximum possible profit minus this opportunity loss. This distinguishes cancellation's zero payoff from its cost relative to an ideal decision.

Always report transaction count, fraud count, split, cancellation rate, legitimate approval rate, and comparator on the same population. Results in notebook 02 use selection only, not calibration or final evaluation.

Sensitivity settings for alternative margins and fraud losses remain explicit in the configuration. The 25%/100% scenario is the historical primary assumption for stages 02–07; do not retrospectively relabel those results. The baseline amount rule remains a fixed training-p99 cancellation rule, distinct from the probability-based policy.

## Additional user-proposed scenario — stage 08

The user subsequently proposed a 0.3% legitimate margin with the same 100% fraud loss. It is recorded separately in [cost-scenario-low-margin.json](../config/cost-scenario-low-margin.json), with an executed [notebook](../notebooks/08-low-margin-payments.ipynb) and [report](reports/08-low-margin-payments.md). Its algebraic break-even probability is about 0.2991%. The original configuration and frozen evaluation remain historical artifacts; reused evaluation comparisons for the new scenario are regression/sensitivity evidence only.
