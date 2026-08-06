# FairCredit — Explainable, Fairness-Audited Credit Default Prediction

**ML Bubble 2026 · TE-BE (Advanced) Track · FinTech domain**

Predicting credit-card default is a solved-looking problem where the interesting
engineering lives *after* the AUC: is the model's decision **cost-aware**, is it
**fair** across protected groups, and can it **explain** why an applicant was
declined? This project builds a four-model comparative pipeline and then adds the
three things a real lender is actually regulated on — cost-sensitive decisions,
a bias audit with mitigation, and per-applicant reason codes.

---

## Headline results (held-out test set, n = 6,000)

Metrics at the cost-optimal threshold **0.375**, which was chosen on validation.

| Model | AUROC | AUPRC | Brier | Recall (defaulters) | F1 |
|---|---|---|---|---|---|
| **XGBoost (winner)** | **0.7799** | **0.5612** | 0.1758 | 0.7679 | 0.4917 |
| Random Forest | 0.7792 | 0.5556 | 0.1800 | 0.7792 | 0.4881 |
| LightGBM | 0.7724 | 0.5478 | 0.1735 | 0.7438 | 0.4984 |
| Logistic Regression | 0.7441 | 0.5003 | 0.1930 | 0.7249 | 0.4821 |

- **Cost-sensitive threshold** (FN:FP = 5:1): moved the operating point from 0.50
  to **0.375**, cutting expected cost per applicant from 0.5667 to 0.5565 —
  a **1.8%** reduction on the test set.
- **Fairness audit** across sex found a modest baseline gap (equal-opportunity
  difference **0.0161**, disparate impact **0.94**). Per-group equal-opportunity
  thresholds narrow it to **0.0129** and lift disparate impact to **0.97**,
  with no loss of ranking power — but they *widen* the FPR gap
  (0.0200 → 0.0256), which is the honest argument for moving to equalized odds.
- **Explainability:** SHAP reason codes surface recent repayment delay (`PAY_1`)
  and delinquency months as the dominant default drivers.

*Every number above is emitted by `outputs/results.json`. Reproduce with a fixed
seed: `py main.py` (about 20 seconds).*

---

## Why this wins on the rubric

| Requirement | How it is met |
|---|---|
| Working trained model | 4 imbalance-aware models, saved to `models/` in portable native formats |
| Performance metrics | AUROC, **AUPRC** (right metric for 22% positives), Brier, F1 |
| Comparative analysis | Linear → bagging → 2× boosting, with a narrative for *why* |
| Deployment considerations | FastAPI scoring API + Streamlit dashboard (live demo) |
| Reproducibility | Seeded, one command, models verified to reload bit-identically |
| Documentation | This README + `docs/PROJECT_DOCUMENTATION.md` |

The differentiator: most teams stop at AUC. FairCredit adds the **fairness audit +
mitigation** and **adverse-action reason codes** that make a credit model
deployable under fair-lending rules.

---

## Quickstart

```bash
py -m pip install -r requirements.txt
py main.py                           # reproduces all metrics + figures (~20s)

py -m uvicorn app.api:app --port 8000   # scoring API -> http://localhost:8000/docs
py -m streamlit run app/dashboard.py    # interactive demo dashboard
```

`py main.py` must be run first: it trains the models and writes
`models/feature_names.json`, which both the API and the dashboard load.

Score an applicant via the API:

```bash
curl -X POST localhost:8000/score -H "Content-Type: application/json" -d '{
  "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
  "PAY_1": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 2, "PAY_5": 2, "PAY_6": 2,
  "BILL_AMT1": 19000, "BILL_AMT2": 18500, "BILL_AMT3": 18000
}'
```

---

## Project structure

```
faircredit/
├── data/            UCI "Default of Credit Card Clients" (30k clients, Taiwan 2005)
├── src/
│   ├── data_prep.py       cleaning + behavioural feature engineering + splits
│   ├── train.py           4-model factory (imbalance-aware)
│   ├── evaluate.py        metrics, ROC/PR curves, comparison plots
│   ├── cost_threshold.py  cost-sensitive threshold optimisation
│   ├── fairness.py        bias audit + equal-opportunity mitigation
│   ├── explain.py         SHAP global summary + local reason codes
│   └── persistence.py     portable model save/load + canonical feature order
├── app/
│   ├── api.py             FastAPI scoring microservice
│   └── dashboard.py       Streamlit demo
├── outputs/         metrics (results.json, CSV) + all figures
├── models/          trained models (.json/.txt/.joblib) + feature_names.json
├── docs/            full project documentation
├── main.py          runs the whole experiment
└── run.sh           one-command reproduction
```

### Model persistence

Models are **never pickled**. A pickled XGBoost/LightGBM estimator embeds the
library's internal booster buffer, which is not portable across library versions
or platforms — loading one elsewhere fails with *"input stream corrupted"*.
Each model is saved in the format its own library guarantees to be portable:

| Model | Format |
|---|---|
| XGBoost | `xgboost.json` (native JSON) |
| LightGBM | `lightgbm.txt` (native text) |
| Logistic Regression, Random Forest | `.joblib` |

`main.py` reloads every model after saving and asserts its predictions match the
in-memory estimator to within 1e-6, so a broken artifact fails the run loudly
instead of at demo time. The training column order is written to
`models/feature_names.json`; serving code reindexes to it rather than trusting
its own field order.

## Dataset

UCI *Default of Credit Card Clients* (Yeh & Lien, 2009): 30,000 credit-card
holders of a Taiwanese bank, April–September 2005. Demographics + a six-month
panel of repayment status, bill amounts, and payments; binary target = default
next month (22.1% positive). See `docs/PROJECT_DOCUMENTATION.md` for the full
data dictionary and methodology.

## License / attribution

Dataset © UCI Machine Learning Repository. Code released for the ML Bubble 2026
submission.
