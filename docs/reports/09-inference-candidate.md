# Reusable low-margin inference candidate

The existing LightGBM model is now wrapped in the `payment_risk` inference package with the stage 08 0.3% margin / 100% fraud-loss economics. This is code delivery after the notebooks; no additional model fit or final evaluation was performed. The candidate threshold is 0.0029910269, with ties and zero amounts approved.

Packaging verified all **42,632 selection scores** against the saved experiment predictions, with a maximum absolute probability difference of **1.11 × 10⁻¹⁶**. Every decision matches the stage 08 LightGBM policy. Serialization/reload preserves the predictions. The artifact contains the model, ordered 30-feature schema, economics, status, runtime versions, and source/code/report lineage, with a checksum manifest.

MLflow now contains `payment-risk-low-margin` **version 1**, alias **candidate**. The registered pyfunc packages code and artifacts with an input/output signature. Its reloaded outputs match the local package. No production/champion alias was created.

CSV batch prediction and a local FastAPI service use the same inference code. Three selection-derived examples match across package, CSV, HTTP, and registry reload. Nine malformed HTTP cases return JSON 422. Unit checks cover percent conversion, ties, zero amounts, feature order, input validation, artifact integrity, CSV preservation, and the earlier analysis contracts. All 11 tests pass. The historical stage 07 inputs and freeze checks remain intact.

**Status:** reusable development candidate and local inference are ready. This verifies implementation consistency, not independent model quality under the new economics. LightGBM's low-margin lead is selection evidence; probability calibration around 0.3% risk and fresh confirmation remain open. Next is a versioned container/release workflow for an offline/demo prototype, followed by the remaining operational evidence before any production promotion. See [inference usage](../inference.md).
