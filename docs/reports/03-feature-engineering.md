# Feature engineering decision

The 30 published predictors are Time, Amount, and V1–V28. Class is excluded from inputs. The components have no interpretable source meaning, and neither customer identifiers nor clock-time metadata are available.

The notebook 02 logistic pipeline already applies `log1p(Amount)` and scales the numeric inputs, fitting every learned statistic on the 171,029 training payments. Notebook 04's tree models use raw values; monotonic amount transformations and scaling are unnecessary for their threshold splits. All candidate features are available from the transaction record.

One additional hypothesis was tested on the same 42,632 selection payments: add sine and cosine of Time using an 86,400-second period to the existing logistic inputs. The period is measured relative to the first transaction, not a known local clock hour. The transformed features are calculated per row; the existing scalers are fitted on training only.

| Logistic feature set | Average precision | Brier score | Profit / 1,000 | Fraud canceled |
|---|---:|---:|---:|---:|
| Existing 30 inputs | **0.741775** | **0.000810** | 21,811.86 | 65.79% |
| Add 24-hour sine/cosine | 0.741458 | 0.000817 | **21,814.81** | 67.11% |

The profit change is 2.95 source units per 1,000 payments while ranking and Brier score slightly worsen. Two days do not establish a stable daily pattern. **Decision:** do not adopt the extra phase features; notebook 04 uses the existing 30 inputs. This is a selection-set development comparison with prior analytical exposure, not a final evaluation. The phase model is saved as an exploratory artifact with predictions and a local MLflow run.
