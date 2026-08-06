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

| Model | AUROC | AUPRC | Recall (defaulters) | F1 |
|---|---|---|---|---|
| **XGBoost (winner)** | **0.779** | **0.563** | **0.795** | 0.487 |
| Random Forest | 0.779 | 0.555 | 0.797 | 0.483 |
| LightGBM | 0.772 | 0.548 | 0.774 | 0.492 |
| Logistic Regression | 0.744 | 0.500 | 0.754 | 0.459 |

- **Cost-sensitive threshold** (FN:FP = 5:1): moved the operating point from 0.50
  to **0.355**, cutting expected cost per applicant by **3.2%** on the test set.
- **Fairness mitigation:** equal-opportunity gap across sex **0.028 → 0.001**,
  disparate-impact ratio **0.94 → 0.99** — with no loss of ranking power.
- **Explainability:** SHAP reason codes correctly surface recent repayment delay
  and delinquency months as the dominant default drivers.

*(Metrics are reproducible with a fixed seed; run `python main.py`.)*

---

## Why this wins on the rubric

| Requirement | How it is met |
|---|---|
| Working trained model | 4 imbalance-aware models, saved to `models/` |
| Performance metrics | AUROC, **AUPRC** (right metric for 22% positives), Brier, F1 |
| Comparative analysis | Linear → bagging → 2× boosting, with a narrative for *why* |
| Deployment considerations | FastAPI scoring API + Streamlit dashboard (live demo) |
| Reproducibility | Seeded, one-command `run.sh`, clean module structure |
| Documentation | This README + `docs/PROJECT_DOCUMENTATION.md` |

The differentiator: most teams stop at AUC. FairCredit adds the **fairness audit +
mitigation** and **adverse-action reason codes** that make a credit model
deployable under fair-lending rules.

---

## Quickstart

```bash
pip install -r requirements.txt
python main.py                       # reproduces all metrics + figures

uvicorn app.api:app --port 8000      # scoring API  -> http://localhost:8000/docs
streamlit run app/dashboard.py       # interactive demo dashboard
```

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
credit-risk/
├── data/            UCI "Default of Credit Card Clients" (30k clients, Taiwan 2005)
├── src/
│   ├── data_prep.py       cleaning + behavioural feature engineering + splits
│   ├── train.py           4-model factory (imbalance-aware)
│   ├── evaluate.py        metrics, ROC/PR curves, comparison plots
│   ├── cost_threshold.py  cost-sensitive threshold optimisation
│   ├── fairness.py        bias audit + equal-opportunity mitigation
│   └── explain.py         SHAP global summary + local reason codes
├── app/
│   ├── api.py             FastAPI scoring microservice
│   └── dashboard.py       Streamlit demo
├── outputs/         metrics (results.json, CSV) + all figures
├── models/          serialised trained models
├── docs/            full project documentation
├── main.py          runs the whole experiment
└── run.sh           one-command reproduction
```

## Dataset

UCI *Default of Credit Card Clients* (Yeh & Lien, 2009): 30,000 credit-card
holders of a Taiwanese bank, April–September 2005. Demographics + a six-month
panel of repayment status, bill amounts, and payments; binary target = default
next month (22.1% positive). See `docs/PROJECT_DOCUMENTATION.md` for the full
data dictionary and methodology.

## License / attribution

Dataset © UCI Machine Learning Repository. Code released for the ML Bubble 2026
submission.
