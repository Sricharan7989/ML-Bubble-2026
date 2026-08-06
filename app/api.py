"""
app/api.py — FastAPI credit-risk scoring microservice.

Serves the trained winner model behind a REST endpoint. For every applicant it
returns: a calibrated default probability, an approve/decline decision at the
cost-optimal threshold, and the top SHAP reason codes (adverse-action reasons
required for a compliant lending decision).

Run:  uvicorn app.api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.data_prep import build_features
from src.explain import top_reason_codes
from src.persistence import load_model, load_feature_names

ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((ROOT / "outputs" / "results.json").read_text(encoding="utf-8"))
WINNER = RESULTS["winner"]
THRESHOLD = RESULTS["cost_threshold"]["optimal_threshold"]
MODEL = load_model(WINNER, ROOT / "models")
FEATURES = load_feature_names(ROOT / "models")   # canonical training column order

app = FastAPI(title="Credit-Risk Scoring API", version="1.0",
              description="Explainable, fairness-audited credit default scoring.")


class Applicant(BaseModel):
    LIMIT_BAL: float = Field(..., example=90000)
    SEX: int = Field(..., example=2, description="1=male, 2=female")
    EDUCATION: int = Field(..., example=2)
    MARRIAGE: int = Field(..., example=1)
    AGE: int = Field(..., example=34)
    PAY_1: int = 0; PAY_2: int = 0; PAY_3: int = 0
    PAY_4: int = 0; PAY_5: int = 0; PAY_6: int = 0
    BILL_AMT1: float = 0; BILL_AMT2: float = 0; BILL_AMT3: float = 0
    BILL_AMT4: float = 0; BILL_AMT5: float = 0; BILL_AMT6: float = 0
    PAY_AMT1: float = 0; PAY_AMT2: float = 0; PAY_AMT3: float = 0
    PAY_AMT4: float = 0; PAY_AMT5: float = 0; PAY_AMT6: float = 0


def _row_to_features(a: Applicant) -> pd.DataFrame:
    # build_features reindexes to the persisted training column order, so the
    # field order of this model can never affect the score.
    return build_features(a.model_dump(), FEATURES)


@app.get("/health")
def health():
    return {"status": "ok", "model": WINNER, "threshold": THRESHOLD}


@app.post("/score")
def score(applicant: Applicant, explain: bool = True):
    X = _row_to_features(applicant)
    p = float(MODEL.predict_proba(X)[:, 1][0])
    decision = "DECLINE" if p >= THRESHOLD else "APPROVE"
    out = {"default_probability": round(p, 4),
           "threshold": THRESHOLD,
           "decision": decision,
           "model": WINNER}
    if explain:
        try:
            codes = top_reason_codes(MODEL, X.to_numpy()[0], list(X.columns), k=5)
            out["reason_codes"] = [
                {"feature": f, "impact": round(s, 4), "value": round(v, 4)}
                for f, s, v in codes]
        except Exception as e:
            out["reason_codes_error"] = str(e)
    return out
