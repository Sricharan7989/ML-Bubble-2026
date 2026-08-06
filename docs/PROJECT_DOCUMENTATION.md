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

**Feature engineering (19 engineered + 23 raw = 42 features).** Behavioural
signals added on top of the raw panel:
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

### 5.1 Model comparison (test set, at cost-optimal threshold 0.375)

| Model | AUROC | AUPRC | Brier | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.7799** | **0.5612** | 0.1758 | 0.3616 | 0.7679 | 0.4917 |
| Random Forest | 0.7792 | 0.5556 | 0.1800 | 0.3553 | 0.7792 | 0.4881 |
| LightGBM | 0.7724 | 0.5478 | 0.1735 | 0.3747 | 0.7438 | 0.4984 |
| Logistic Regression | 0.7441 | 0.5003 | 0.1930 | 0.3611 | 0.7249 | 0.4821 |

**Narrative.** The linear baseline is already respectable (AUROC 0.744), but the
boosters add ~3.5 AUROC points and ~6 AUPRC points by modelling interactions in
the repayment panel. XGBoost and Random Forest are statistically neck-and-neck
on test (ΔAUROC 0.0007); **XGBoost wins on validation AUPRC (0.5582 vs 0.5508)**
and is selected on that basis alone, before the test set is touched. LightGBM is
close and would be preferred where inference latency matters.

Validation AUPRC used for selection: XGBoost 0.5582, LightGBM 0.5555,
Random Forest 0.5508, Logistic Regression 0.4930.

### 5.2 Cost-sensitive decisioning

A default at 0.5 assumes both errors cost the same. In lending they do not:
approving a defaulter (false negative) loses principal, worth far more than
rejecting a good customer (false positive). Under a **5:1** cost ratio, sweeping
the threshold on validation gives an optimal cutoff of **0.375**, which lowers
expected cost per applicant from **0.5667 → 0.5565** on the test set — a **1.8%**
reduction purely from choosing the operating point deliberately.

The gain is modest because the cost curve is flat near its minimum on this
dataset; the point is that the operating point is now a *derived business
decision* with a stated cost ratio, not an unexamined default. The ratio is
configuration (`COST_FN`/`COST_FP` in `src/cost_threshold.py`), so a risk team
can retune it without touching the model.

### 5.3 Fairness audit (protected attribute: sex)

Fairness is measured on the winning model at the cost-optimal threshold 0.375:

| | n | Selection rate | TPR (recall) | FPR |
|---|---|---|---|---|
| Male | 2,402 | 0.4858 | 0.7772 | 0.3971 |
| Female | 3,598 | 0.4589 | 0.7611 | 0.3771 |

| Gap metric | Before | After mitigation |
|---|---|---|
| Demographic-parity difference | 0.0269 | **0.0148** |
| Equal-opportunity difference (TPR gap) | 0.0161 | **0.0129** |
| FPR difference | 0.0200 | 0.0256 |
| Disparate-impact ratio | 0.9446 | **0.9694** |

**Mitigation.** We equalise **true-positive rate** across groups by choosing a
per-group decision threshold on validation (male 0.385, female 0.360) — a
deploy-time control that leaves the trained model untouched. The target TPR is
itself measured on **validation** (0.7919), never on test, so the mitigation
carries no test-set information.

**Honest reading of this result.** The baseline disparity is real but small, and
the mitigation is a partial success: it narrows the TPR gap by ~20% and the
selection-rate gap by ~45%, lifting disparate impact from 0.94 to 0.97 (both
already clear the 0.8 four-fifths rule). But equalising TPR alone **widens the
FPR gap**, from 0.0200 to 0.0256 — good applicants of one group are now flagged
slightly more often. This is the textbook limitation of optimising equal
opportunity in isolation, and it is the motivation for moving to an
**equalized-odds** criterion that constrains TPR and FPR jointly. We report the
regression rather than the single flattering metric.

### 5.4 Explainability (SHAP)

- **Global:** across the portfolio, the most recent repayment status (`PAY_1`),
  number of delinquent months, and worst delay dominate default risk — consistent
  with credit intuition.
- **Local (reason codes):** for the highest-risk applicant in the test set
  (p = 0.9618) the system outputs ranked drivers — `PAY_1 = 3` (three months
  delayed, SHAP +0.94), `DELINQ_MONTHS = 6` (+0.57), `MAX_DELAY = 3` (+0.39),
  `MEAN_DELAY = 2.17` (+0.20), `PAYRATIO_MEAN = 0.0` (+0.14) — exactly the
  "adverse-action reasons" a lender must disclose.

## 6. Deployment considerations

- **Scoring API** (`app/api.py`, FastAPI): a `/score` endpoint returns
  probability, an approve/decline decision at the cost-optimal threshold, and
  reason codes. Stateless and container-ready; auto-generated OpenAPI docs.
- **Demo dashboard** (`app/dashboard.py`, Streamlit): score an applicant live,
  browse the model comparison, and inspect the fairness audit.
- **Model artifacts:** serialised in each library's own portable format —
  XGBoost as native JSON, LightGBM as native text, scikit-learn via joblib.
  Pickling a booster embeds a version- and platform-specific binary buffer that
  fails to load elsewhere with *"input stream corrupted"*; native formats are the
  only artifacts safe to hand to a container or a colleague. `main.py` reloads
  every model after saving and fails the run if predictions drift by more
  than 1e-6.
- **Train/serve consistency:** the training column order is exported to
  `models/feature_names.json`, and all serving paths build their feature frame
  through the single `data_prep.build_features()` entry point, which reindexes to
  it. Column order at the API boundary therefore cannot affect a score.
- **Production notes:** the decision threshold and cost ratio are configuration,
  not code, so risk teams can retune them; SHAP reason codes support compliance;
  the fairness thresholds should be re-audited on each retrain and monitored for
  drift.

## 7. Limitations & future work

- Single-country, 2005 data — retrain on current, local data before any real use.
- `SEX` is the only protected attribute available; a production audit would also
  cover age bands and intersectional groups.
- **Equal opportunity alone is not enough.** Our per-group thresholds narrow the
  TPR gap but widen the FPR gap (§5.3). The right next step is an
  **equalized-odds** constraint that bounds both simultaneously.
- Threshold-based mitigation is simple and transparent; in-processing methods
  (e.g. adversarial debiasing) could push fairness further, at some AUC cost.
- No probability calibration layer yet (Platt/Isotonic) — worth adding since the
  decision is threshold-based and Brier is only ~0.176.
- Single fixed split; cross-validated metrics with confidence intervals would
  show whether the XGBoost-vs-Random-Forest margin (ΔAUROC 0.0007) is real. On
  present evidence it is not distinguishable from noise, and we say so rather
  than claiming a decisive winner.

## 8. Reproduction

```bash
py -m pip install -r requirements.txt
py main.py     # writes outputs/results.json, all figures, and models/
```

Every number in this document is emitted by that single command with a fixed
seed (`random_state=42`), and is read from `outputs/results.json`. The run takes
roughly 20 seconds on a laptop.
