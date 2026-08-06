# FairCredit — Project Documentation

**ML Bubble 2026 · TE-BE Advanced Track · FinTech**

---

## 1. Problem statement

Card-issuing banks must decide, at origination and on an ongoing basis, whether a
customer will default. A model that is merely *accurate* is not enough to deploy:
it must make **cost-aware** decisions (the two error types cost very different
amounts), be **fair** across protected groups (fair-lending law), and produce
**reasons** for adverse decisions (regulatory requirement). This project delivers
a default-prediction system engineered around those three constraints.

## 2. Why machine learning

Default is driven by non-linear interactions across a six-month behavioural panel
(repayment status, utilisation, payment ratios) that hand-written rules capture
poorly. Gradient-boosted trees model these interactions directly, rank applicants
by risk, and — via SHAP — expose the drivers behind each score. The task is a
supervised binary classification problem with class imbalance.

## 3. Dataset

**Source:** UCI *Default of Credit Card Clients* (Yeh & Lien, 2009), 30,000
clients of a Taiwanese bank, April–September 2005.

| Group | Fields |
|---|---|
| Demographics | `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` |
| Repayment status | `PAY_1`…`PAY_6` (−1 = paid duly, positive = months delayed) |
| Bill amounts | `BILL_AMT1`…`BILL_AMT6` |
| Payments | `PAY_AMT1`…`PAY_AMT6` |
| Target | `default` (1 = default next month) — **22.1% positive** |

**Cleaning.** `EDUCATION` codes 0/5/6 (undocumented) merged into "other"; `MARRIAGE`
code 0 merged into "other"; raw `PAY_0` renamed to `PAY_1` (most recent month).

**Feature engineering (42 features).** Behavioural signals added on top of the raw
panel:
- **Utilisation** per month (`BILL_AMT / LIMIT_BAL`) and mean utilisation.
- **Repayment ratio** per month (`PAY_AMT / previous BILL`) and mean.
- **Delinquency**: months in arrears, max delay, mean delay.
- **Exposure/trend**: total bill, total paid, six-month bill trend.

## 4. Methodology

**Splits.** Stratified **64/16/20** train/validation/test, seeded (`random_state=42`).
The **test set is touched exactly once**, at the end. Model selection and the
cost threshold are decided on validation — no leakage.

**Models (comparative spine).** Four models of increasing capacity, all
imbalance-aware:

| Model | Role | Imbalance handling |
|---|---|---|
| Logistic Regression | transparent linear baseline | balanced class weights |
| Random Forest | non-linear bagging baseline | balanced class weights |
| XGBoost | boosting (credit-scoring standard) | `scale_pos_weight` |
| LightGBM | boosting, leaf-wise/faster | `scale_pos_weight` |

**Metrics.** On a 22%-positive target, accuracy is misleading (predicting "no
default" for everyone already scores ~78%). We report **AUPRC** (primary), AUROC,
Brier (calibration), and precision/recall/F1 at the chosen operating threshold.

## 5. Results

### 5.1 Model comparison (test set, at cost-optimal threshold 0.355)

| Model | AUROC | AUPRC | Brier | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.779** | **0.563** | 0.176 | 0.351 | 0.795 | 0.487 |
| Random Forest | 0.779 | 0.555 | 0.176 | 0.346 | 0.797 | 0.483 |
| LightGBM | 0.772 | 0.548 | 0.174 | 0.361 | 0.774 | 0.492 |
| Logistic Regression | 0.744 | 0.500 | 0.193 | 0.330 | 0.754 | 0.459 |

**Narrative.** The linear baseline is already respectable (AUROC 0.744), but the
boosters add ~3.5 AUROC points and ~6 AUPRC points by modelling interactions in
the repayment panel. XGBoost and Random Forest are statistically neck-and-neck;
**XGBoost wins on validation AUPRC (0.558)** and is selected. LightGBM is close
and would be preferred where inference latency matters.

### 5.2 Cost-sensitive decisioning

A default at 0.5 assumes both errors cost the same. In lending they do not:
approving a defaulter (false negative) loses principal, worth far more than
rejecting a good customer (false positive). Under a **5:1** cost ratio, sweeping
the threshold on validation gives an optimal cutoff of **0.355**, which lowers
expected cost per applicant from **0.569 → 0.551** on the test set — a **3.2%**
reduction purely from choosing the operating point deliberately.

### 5.3 Fairness audit (protected attribute: sex)

Fairness is measured on the winning model at the cost-optimal threshold:

| | Selection rate | TPR (recall) | FPR |
|---|---|---|---|
| Male | 0.519 | 0.811 | 0.430 |
| Female | 0.488 | 0.783 | 0.408 |

| Gap metric | Before | After mitigation |
|---|---|---|
| Demographic-parity difference | 0.031 | **0.004** |
| Equal-opportunity difference (TPR gap) | 0.028 | **0.001** |
| Disparate-impact ratio | 0.94 | **0.99** |

**Mitigation.** We equalise **true-positive rate** across groups by choosing a
per-group decision threshold on validation (male 0.375, female 0.355) — a
deploy-time control that leaves the trained model untouched. The equal-opportunity
gap collapses to ~0.001 with no loss of ranking quality.

### 5.4 Explainability (SHAP)

- **Global:** across the portfolio, the most recent repayment status (`PAY_1`),
  number of delinquent months, and worst delay dominate default risk — consistent
  with credit intuition.
- **Local (reason codes):** for the highest-risk applicant (p = 0.96) the system
  outputs ranked drivers — `PAY_1 = 3` (three months delayed), `DELINQ_MONTHS = 6`,
  `MAX_DELAY = 7` — exactly the "adverse-action reasons" a lender must disclose.

## 6. Deployment considerations

- **Scoring API** (`app/api.py`, FastAPI): a `/score` endpoint returns
  probability, an approve/decline decision at the cost-optimal threshold, and
  reason codes. Stateless and container-ready; auto-generated OpenAPI docs.
- **Demo dashboard** (`app/dashboard.py`, Streamlit): score an applicant live,
  browse the model comparison, and inspect the fairness audit.
- **Production notes:** models serialised in `models/`; the decision threshold and
  cost ratio are configuration, not code, so risk teams can retune them; SHAP
  reason codes support compliance; the fairness thresholds should be re-audited on
  each retrain and monitored for drift.

## 7. Limitations & future work

- Single-country, 2005 data — retrain on current, local data before any real use.
- `SEX` is the only protected attribute available; a production audit would also
  cover age bands and intersectional groups.
- Threshold-based mitigation is simple and transparent; in-processing methods
  (e.g. adversarial debiasing) could push fairness further, at some AUC cost.
- No probability calibration layer yet (Platt/Isotonic) — worth adding since the
  decision is threshold-based.

## 8. Reproduction

```bash
pip install -r requirements.txt
python main.py     # writes outputs/results.json, all figures, and models/
```

All numbers in this document are produced by that single command with a fixed
seed.
