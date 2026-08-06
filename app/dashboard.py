"""
app/dashboard.py — Streamlit demo dashboard.

Three tabs for the live demo:
  1. Score an applicant  -> probability, decision, SHAP reason codes
  2. Model comparison    -> the comparative-analysis results
  3. Fairness            -> the bias audit before/after mitigation

Run:  streamlit run app/dashboard.py
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_prep import build_features
from src.explain import top_reason_codes
from src.persistence import load_model, load_feature_names

ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((ROOT / "outputs" / "results.json").read_text(encoding="utf-8"))
WINNER = RESULTS["winner"]
THRESHOLD = RESULTS["cost_threshold"]["optimal_threshold"]
MODEL = load_model(WINNER, ROOT / "models")
FEATURES = load_feature_names(ROOT / "models")   # canonical training column order

st.set_page_config(page_title="Credit-Risk Scorer", layout="wide")
st.title("💳 Explainable, Fairness-Audited Credit-Risk Scorer")
st.caption(f"Winner model: **{WINNER}**  ·  cost-optimal threshold "
           f"**{THRESHOLD}**  ·  ML Bubble 2026")

tab1, tab2, tab3 = st.tabs(["🎯 Score applicant", "📊 Model comparison",
                            "⚖️ Fairness audit"])

with tab1:
    c1, c2, c3 = st.columns(3)
    limit = c1.number_input("Credit limit (NT$)", 10000, 1000000, 90000, 10000)
    age = c2.number_input("Age", 18, 90, 34)
    sex = c3.selectbox("Sex", [1, 2],
                       format_func=lambda v: "Male" if v == 1 else "Female")
    edu = c1.selectbox("Education", [1, 2, 3, 4],
                       format_func=lambda v: {1: "Grad school", 2: "University",
                                              3: "High school", 4: "Other"}[v])
    marr = c2.selectbox("Marriage", [1, 2, 3],
                        format_func=lambda v: {1: "Married", 2: "Single",
                                               3: "Other"}[v])
    st.markdown("**Repayment status last 6 months** "
                "(-1 = paid duly, positive = months delayed)")
    pcols = st.columns(6)
    pay = [pcols[i].number_input(f"PAY_{i+1}", -2, 8, 0, key=f"p{i}")
           for i in range(6)]
    st.markdown("**Bill amounts** and **payments** (NT$)")
    bcols = st.columns(6)
    bill = [bcols[i].number_input(f"BILL_{i+1}", 0, 1000000, 40000, 5000,
                                  key=f"b{i}") for i in range(6)]
    acols = st.columns(6)
    paid = [acols[i].number_input(f"PAY_AMT{i+1}", 0, 1000000, 2000, 500,
                                  key=f"a{i}") for i in range(6)]

    if st.button("Score applicant", type="primary"):
        raw = {"LIMIT_BAL": limit, "SEX": sex, "EDUCATION": edu,
               "MARRIAGE": marr, "AGE": age}
        for i in range(6):
            raw[f"PAY_{i+1}"] = pay[i]
            raw[f"BILL_AMT{i+1}"] = bill[i]
            raw[f"PAY_AMT{i+1}"] = paid[i]
        X = build_features(raw, FEATURES)
        p = float(MODEL.predict_proba(X)[:, 1][0])
        decision = "DECLINE" if p >= THRESHOLD else "APPROVE"
        m1, m2 = st.columns(2)
        m1.metric("Default probability", f"{p:.1%}")
        m2.metric("Decision", decision,
                  delta="above threshold" if decision == "DECLINE" else "below",
                  delta_color="inverse")
        st.progress(min(p, 1.0))
        codes = top_reason_codes(MODEL, X.to_numpy()[0], FEATURES, k=5)
        st.subheader("Reason codes (why this decision)")
        st.dataframe(pd.DataFrame(
            [{"Feature": f, "Impact on risk": round(s, 3),
              "Applicant value": round(v, 3)} for f, s, v in codes]),
            use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Comparative analysis (test set)")
    st.dataframe(pd.DataFrame(RESULTS["comparison"]),
                 use_container_width=True, hide_index=True)
    st.image(str(ROOT / "outputs" / "roc_pr_curves.png"))

with tab3:
    st.subheader("Fairness audit across sex")
    f = RESULTS["fairness"]
    cA, cB = st.columns(2)
    cA.markdown("**Before (single threshold)**")
    cA.dataframe(pd.DataFrame(f["before"]), hide_index=True)
    cA.json(f["gaps_before"])
    cB.markdown("**After (equal-opportunity thresholds)**")
    cB.dataframe(pd.DataFrame(f["after"]), hide_index=True)
    cB.json(f["gaps_after"])
    st.image(str(ROOT / "outputs" / "fairness_audit.png"))
