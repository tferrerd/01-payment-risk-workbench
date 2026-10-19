# Transaction explorer

The explorer serves a browser UI and the existing inference API from one Python process. Start it from the project root:

```bash
poetry run uvicorn payment_risk.explorer:app --host 127.0.0.1 --port 8002 --workers 1
```

Open `http://127.0.0.1:8002/`. The page fetches 200 curated selection examples from `/demo` and sends their 30 predictors to `/predict` for live inference. Labels are used only in the browser's illustrative payoff calculations and are never model inputs. There are 190 legitimate and 10 fraudulent payments, intentionally enriched for exploration; sample totals do not estimate population performance. Amounts retain the source dataset's monetary units.

Search, filter, sort, and paginate payments, then select one to inspect its score and decision. `/explain` computes native LightGBM TreeSHAP contributions in raw log-odds. The base plus all 30 contributions reconstructs the raw score, whose sigmoid matches the predicted probability. The chart shows six large contributions and the sum of the other 24. These are model explanations, not causal findings; V1–V28 are anonymized PCA components.

The policy lab changes the browser's assumed legitimate margin and fraud loss, recalculates margin/(margin + fraud loss), and applies the strict cancellation rule with approval for ties and zero amounts. Sample profit uses published sample labels and the chosen assumptions. The cutoff curve is descriptive and must not be used as an independent threshold-selection result. The packaged API policy remains +0.3% / −100% throughout. CSV exports distinguish API decisions from simulated decisions and include both economic fractions and the model ID.

The model, its registered version, and historical evaluation inputs are unchanged. `payment_risk/explorer.py` adds routes around the original `payment_risk.api` application. The separate service release `lightgbm-low-margin-dev-v1-explorer-v1` binds the new UI, sample, explanations code, and original model files in `deployment/explorer-v1.json`; it does not replace the original inference manifest or model registration. The candidate's original lineage remains a creation-time record. Changes to application code require a new service release, not overwriting the version 1 model artifact or rerunning its packaging command against changed source.

Prepare the same sample again only from verified sources and saved selection assignments:

```bash
poetry run python -m scripts.prepare_explorer_sample
poetry run python -m unittest discover -s tests
node --test tests/test_explorer.mjs
```

`reports/explorer-sample.json` records source row IDs, source/split checksums, and the preparation code checksum. Historical evaluation rows are excluded. Container smoke checks score all 200 examples, confirm the policy, check an explanation and reject label-bearing explanation inputs. The UI uses plain HTML, CSS, and JavaScript with no frontend build step. Google Fonts is optional; system fonts are the fallback. The app does not persist payment requests or add a database.

For Docker and hosting, see [Render deployment](cloud-deployment.md).
