"""Transaction explorer layered on the unchanged version 1 inference API."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import Transaction, app
from .pipeline import FEATURES

ROOT = Path(__file__).resolve().parents[1]


@app.get("/demo", include_in_schema=False)
def demo(request: Request):
    sample = json.loads((ROOT / "examples/explorer-sample.json").read_text())
    pipeline = request.app.state.pipeline
    return {**sample, "model_id": pipeline.metadata["model_id"],
            "economics": pipeline.metadata["economics"], "threshold": pipeline.threshold}


@app.post("/explain")
def explain(body: Transaction, request: Request):
    pipeline = request.app.state.pipeline
    frame = pipeline.validate_inputs(pd.DataFrame([body.model_dump()]))
    contributions = np.asarray(pipeline.model.predict(frame, pred_contrib=True))[0]
    raw = float(pipeline.model.predict(frame, raw_score=True)[0])
    probability = float(pipeline.predict(frame).fraud_probability.iloc[0])
    if contributions.shape != (len(FEATURES) + 1,) or not np.isfinite(contributions).all():
        raise ValueError("Unexpected model contribution output")
    if not np.isclose(contributions.sum(), raw, rtol=0, atol=1e-8):
        raise ValueError("Contributions do not reconstruct the model score")
    if not np.isclose(np.exp(-np.logaddexp(0, -raw)), probability, rtol=0, atol=1e-10):
        raise ValueError("Explanation and prediction disagree")
    return {"model_id": pipeline.metadata["model_id"], "fraud_probability": probability,
            "base_log_odds": float(contributions[-1]), "raw_log_odds": raw,
            "contributions": [{"feature": name, "value": float(frame[name].iloc[0]),
                               "log_odds": float(value)} for name, value in zip(FEATURES, contributions[:-1])],
            "units": "raw log-odds", "method": "LightGBM native TreeSHAP",
            "interpretation": "Model contributions, not causes. V1–V28 are anonymized PCA components."}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "web/index.html")


app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="explorer-assets")
