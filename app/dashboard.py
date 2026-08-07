"""
app/dashboard.py — Streamlit demo dashboard.

Three tabs for the live demo:
  1. Score an applicant  -> risk gauge, decision, SHAP waterfall + reason codes
  2. Model comparison    -> the comparative-analysis results
  3. Fairness            -> the bias audit before/after mitigation

The scoring tab puts the *answer* at the top: decision badge, metric cards, gauge
and waterfall all render above the input form, so a judge sees the result without
scrolling. Inputs are pre-fillable from two one-click preset applicants.

This module is presentation only — it loads the artifacts written by `main.py`
and never re-fits, re-thresholds, or otherwise touches the modelling logic.

Run:  py -m streamlit run app/dashboard.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_prep import build_features
from src.explain import top_reason_codes, tree_explainer, _positive_class
from src.persistence import load_model, load_feature_names

ROOT = Path(__file__).resolve().parents[1]

# Brand palette, kept in sync with .streamlit/config.toml.
# DANGER is #DC2626 rather than a rose red: it clears OKLab deutan ΔE 12.8
# against the emerald, so approve/decline survives red-green colour blindness.
EMERALD = "#10B981"
DANGER = "#DC2626"
SLATE = "#94A3B8"
INK = "#F1F5F9"
GRID = "rgba(148,163,184,0.18)"

st.set_page_config(
    page_title="FairCredit — credit-risk scorer",
    page_icon=":material/account_balance:",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    results = json.loads(
        (ROOT / "outputs" / "results.json").read_text(encoding="utf-8"))
    model = load_model(results["winner"], ROOT / "models")
    features = load_feature_names(ROOT / "models")
    return results, model, features


@st.cache_resource
def get_explainer(_model):
    """SHAP explainer for the winner. Underscore skips Streamlit's arg hashing."""
    return tree_explainer(_model, None)


RESULTS, MODEL, FEATURES = load_artifacts()
WINNER = RESULTS["winner"]
THRESHOLD = RESULTS["cost_threshold"]["optimal_threshold"]

# ---------------------------------------------------------------------------
# Input state — every widget is keyed, so presets can fill the form by writing
# to session_state in an on_click callback (callbacks run before the rerun).
# ---------------------------------------------------------------------------
FIELD_DEFAULTS = {
    "limit": 90000, "age": 34, "sex": 2, "edu": 2, "marr": 1,
    **{f"pay_{i}": 0 for i in range(1, 7)},
    **{f"bill_{i}": 40000 for i in range(1, 7)},
    **{f"amt_{i}": 2000 for i in range(1, 7)},
}

# Both presets are real points either side of the cost-optimal threshold:
# they score ~0.89 and ~0.13 against the shipped model.
PRESETS = {
    "high": {
        "limit": 20000, "age": 24, "sex": 2, "edu": 3, "marr": 2,
        **{f"pay_{i}": 2 for i in range(1, 7)},
        **{f"bill_{i}": b for i, b in
           enumerate([19500, 19200, 18800, 18400, 18000, 17600], start=1)},
        **{f"amt_{i}": 0 for i in range(1, 7)},
    },
    "low": {
        "limit": 360000, "age": 46, "sex": 1, "edu": 1, "marr": 1,
        **{f"pay_{i}": -1 for i in range(1, 7)},
        **{f"bill_{i}": b for i, b in
           enumerate([12000, 14000, 11000, 13000, 10000, 12500], start=1)},
        **{f"amt_{i}": a for i, a in
           enumerate([14000, 11000, 13000, 10000, 12500, 9000], start=1)},
    },
}

for _key, _val in FIELD_DEFAULTS.items():
    st.session_state.setdefault(_key, _val)
st.session_state.setdefault("scored", False)


def apply_preset(name: str):
    """Fill every input from a preset and score it in the same click."""
    st.session_state.update(PRESETS[name])
    st.session_state.scored = True


def mark_scored():
    st.session_state.scored = True


def collect_inputs() -> dict:
    """Read the keyed widget values back out of session_state."""
    s = st.session_state
    raw = {"LIMIT_BAL": s.limit, "SEX": s.sex, "EDUCATION": s.edu,
           "MARRIAGE": s.marr, "AGE": s.age}
    for i in range(1, 7):
        raw[f"PAY_{i}"] = s[f"pay_{i}"]
        raw[f"BILL_AMT{i}"] = s[f"bill_{i}"]
        raw[f"PAY_AMT{i}"] = s[f"amt_{i}"]
    return raw


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def gauge_figure(p: float) -> go.Figure:
    """Risk dial: emerald below the cost-optimal threshold, red above it."""
    risk_color = DANGER if p >= THRESHOLD else EMERALD
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "font": {"size": 46, "color": risk_color}},
        gauge={
            "axis": {
                "range": [0, 100], "tickwidth": 1, "tickcolor": GRID,
                "tickfont": {"color": SLATE, "size": 11},
                "ticksuffix": "%",
            },
            "bar": {"color": risk_color, "thickness": 0.68},
            "bgcolor": "rgba(148,163,184,0.08)",
            "borderwidth": 0,
            # Tinted bands make the safe/decline zones readable even before
            # the needle is interpreted.
            "steps": [
                {"range": [0, THRESHOLD * 100], "color": "rgba(16,185,129,0.18)"},
                {"range": [THRESHOLD * 100, 100], "color": "rgba(220,38,38,0.18)"},
            ],
            "threshold": {
                "line": {"color": INK, "width": 3},
                "thickness": 0.9,
                "value": THRESHOLD * 100,
            },
        },
    ))
    fig.update_layout(
        height=250,
        margin=dict(t=20, b=10, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
    )
    return fig


def waterfall_figure(X: pd.DataFrame, k: int = 7) -> go.Figure:
    """SHAP waterfall: baseline risk -> this applicant, in model log-odds."""
    explainer = get_explainer(MODEL)
    sv = _positive_class(explainer.shap_values(X.to_numpy())).ravel()

    base = np.ravel(explainer.expected_value)[-1]
    base = float(base)

    order = np.argsort(-np.abs(sv))[:k]
    labels = [f"{FEATURES[i]} = {X.iloc[0, i]:g}" for i in order]
    values = [float(sv[i]) for i in order]

    remainder = float(sv.sum() - sum(values))
    if abs(remainder) > 1e-6:
        labels.append(f"{len(sv) - len(values)} other features")
        values.append(remainder)

    y = ["Baseline risk"] + labels + ["This applicant"]
    x = [base] + values + [0.0]
    measure = ["absolute"] + ["relative"] * len(values) + ["total"]
    text = ([f"{base:.2f}"]
            + [f"{v:+.2f}" for v in values]
            + [f"{base + sum(values):.2f}"])

    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=measure,
        y=y,
        x=x,
        text=text,
        textposition="outside",
        textfont={"color": INK, "size": 11},
        increasing={"marker": {"color": DANGER}},   # pushes toward default
        decreasing={"marker": {"color": EMERALD}},  # pushes toward repayment
        totals={"marker": {"color": SLATE}},
        connector={"line": {"color": GRID, "width": 1}},
        hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>",
    ))
    fig.update_layout(
        height=380,
        margin=dict(t=10, b=40, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12),
        showlegend=False,
        xaxis=dict(title=dict(text="Contribution to default log-odds",
                              font=dict(color=SLATE, size=11)),
                   gridcolor=GRID, zerolinecolor=GRID,
                   tickfont=dict(color=SLATE, size=11)),
        # Keep the baseline at the top, reading downward to the final score.
        yaxis=dict(autorange="reversed", tickfont=dict(color=INK, size=11)),
    )
    return fig


def render_result(p: float, X: pd.DataFrame):
    """Decision badge + metric cards + gauge + waterfall + reason codes."""
    declined = p >= THRESHOLD
    decision = "Decline" if declined else "Approve"

    with st.container(border=True):
        st.badge(
            f"{decision} — {p:.1%} default risk",
            icon=":material/gpp_bad:" if declined else ":material/verified_user:",
            color="red" if declined else "green",
        )

        with st.container(horizontal=True):
            st.metric(
                "Default probability", f"{p:.1%}",
                delta=f"{(p - THRESHOLD) * 100:+.1f} pts vs threshold",
                delta_color="inverse", border=True,
                help="Predicted probability this applicant defaults next month.",
            )
            st.metric(
                "Decision", decision, border=True,
                help="Applied at the cost-optimal threshold, not 0.50.",
            )
            st.metric(
                "Cost-optimal threshold", f"{THRESHOLD:.1%}", border=True,
                help="Chosen on validation under a 5:1 false-negative to "
                     "false-positive cost ratio.",
            )

        left, right = st.columns([1, 1.45], gap="medium")
        with left:
            st.caption("Risk gauge")
            st.plotly_chart(gauge_figure(p), width="stretch", theme=None,
                            config={"displayModeBar": False})
        with right:
            st.caption("What drove this score")
            st.plotly_chart(waterfall_figure(X), width="stretch", theme=None,
                            config={"displayModeBar": False})

        st.caption(
            "Red bars push the applicant toward default, emerald bars away "
            "from it. Values are SHAP contributions in model log-odds, summing "
            "from the portfolio baseline to this applicant's score."
        )

    with st.container(border=True):
        st.markdown("**Adverse-action reason codes**")
        codes = top_reason_codes(MODEL, X.to_numpy()[0], FEATURES, k=5)
        st.dataframe(
            pd.DataFrame([
                {"Rank": i, "Feature": f, "Applicant value": round(v, 3),
                 "Impact on risk": round(s, 3),
                 "Direction": "Increases risk" if s > 0 else "Reduces risk"}
                for i, (f, s, v) in enumerate(codes, start=1)
            ]),
            width="stretch", hide_index=True,
            column_config={
                "Impact on risk": st.column_config.NumberColumn(format="%+.3f"),
            },
        )
        st.caption(
            "These are the specific reasons a lender must disclose for an "
            "adverse decision under fair-lending rules."
        )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("FairCredit — explainable, fairness-audited credit risk")
st.caption(
    f"Winner model **{WINNER}**  ·  cost-optimal threshold **{THRESHOLD:.3f}**  "
    f"·  ML Bubble 2026"
)

tab_score, tab_models, tab_fairness = st.tabs([
    "Score applicant", "Model comparison", "Fairness audit",
])

with tab_score:
    # Declared first so the result renders ABOVE the form, but filled in after
    # the widgets below have produced their values.
    result_slot = st.container()

    st.markdown("**Load an example**")
    with st.container(horizontal=True):
        st.button("Load high-risk example", icon=":material/trending_up:",
                  on_click=apply_preset, args=("high",))
        st.button("Load low-risk example", icon=":material/trending_down:",
                  on_click=apply_preset, args=("low",))

    with st.container(border=True):
        st.markdown("**Applicant**")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Credit limit (NT$)", 10000, 1000000, step=10000,
                        key="limit")
        c2.number_input("Age", 18, 90, step=1, key="age")
        c3.selectbox("Sex", [1, 2], key="sex",
                     format_func=lambda v: "Male" if v == 1 else "Female",
                     help="Used only for the fairness audit, never as an excuse "
                          "to drop the attribute.")
        c1.selectbox("Education", [1, 2, 3, 4], key="edu",
                     format_func=lambda v: {1: "Grad school", 2: "University",
                                            3: "High school", 4: "Other"}[v])
        c2.selectbox("Marital status", [1, 2, 3], key="marr",
                     format_func=lambda v: {1: "Married", 2: "Single",
                                            3: "Other"}[v])

        with st.expander("Repayment history (last 6 months)",
                         icon=":material/history:"):
            st.caption("Repayment status — −1 = paid duly, positive = months delayed")
            pay_cols = st.columns(6)
            for i in range(1, 7):
                pay_cols[i - 1].number_input(f"PAY_{i}", -2, 8, step=1,
                                             key=f"pay_{i}")

            st.caption("Bill amount (NT$)")
            bill_cols = st.columns(6)
            for i in range(1, 7):
                bill_cols[i - 1].number_input(f"BILL_{i}", 0, 1000000, step=5000,
                                              key=f"bill_{i}")

            st.caption("Amount paid (NT$)")
            amt_cols = st.columns(6)
            for i in range(1, 7):
                amt_cols[i - 1].number_input(f"PAID_{i}", 0, 1000000, step=500,
                                             key=f"amt_{i}")

        st.button("Score applicant", type="primary",
                  icon=":material/bolt:", on_click=mark_scored)

    if st.session_state.scored:
        X = build_features(collect_inputs(), FEATURES)
        p = float(MODEL.predict_proba(X)[:, 1][0])
        with result_slot:
            render_result(p, X)
    else:
        with result_slot:
            st.info(
                "Load an example or fill in the form, then select "
                "**Score applicant** to see the decision, risk gauge and "
                "SHAP explanation.",
                icon=":material/lightbulb:",
            )

with tab_models:
    st.subheader("Comparative analysis (test set)")
    st.dataframe(pd.DataFrame(RESULTS["comparison"]),
                 width="stretch", hide_index=True)
    st.image(str(ROOT / "outputs" / "roc_pr_curves.png"))

with tab_fairness:
    st.subheader("Fairness audit across sex")
    f = RESULTS["fairness"]
    cA, cB = st.columns(2)
    with cA:
        with st.container(border=True):
            st.markdown("**Before — single threshold**")
            st.dataframe(pd.DataFrame(f["before"]), width="stretch",
                         hide_index=True)
            st.json(f["gaps_before"])
    with cB:
        with st.container(border=True):
            st.markdown("**After — equal-opportunity thresholds**")
            st.dataframe(pd.DataFrame(f["after"]), width="stretch",
                         hide_index=True)
            st.json(f["gaps_after"])
    st.image(str(ROOT / "outputs" / "fairness_audit.png"))
