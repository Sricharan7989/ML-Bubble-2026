# FairCredit — 3-Minute Pitch Script

Roughly 450 words ≈ 3 minutes at a calm pace. Times are cumulative. Say the
**bold** numbers slowly — they are what judges remember.

---

**[0:00 — Slide 1, Title]**
"Predicting credit-card default looks like a solved problem. Everyone can hit a
decent AUC. So we asked a different question: what would it take to actually
*deploy* this model at a bank? That question is where FairCredit lives."

**[0:20 — Slide 2, The problem]**
"A bank can't ship a model that's only accurate. It's regulated on three things
accuracy ignores. One: cost — approving someone who defaults costs far more than
rejecting a good customer. Two: fairness — a high-AUC model can still reject
protected groups at different rates, which is illegal. Three: explainability —
lenders must give the *reason* for a rejection. We built for all three."

**[0:45 — Slides 3–4, Data & method]**
"We used the UCI Taiwan dataset — **30,000** real clients, a genuinely imbalanced
**22%** default rate. We engineered **19** behavioural features out of the
six-month repayment panel — utilisation, repayment ratios, delinquency counts —
for **42** in total. The whole pipeline is leakage-safe: model choice, the cost
threshold, and the fairness thresholds are all decided on validation, and the
test set is touched exactly once. Every number you're about to see is honest."

**[1:10 — Slide 5, Comparative analysis]**
"We compared four models, simplest to strongest. The linear baseline is fine, but
gradient boosting adds real signal. **XGBoost wins** — AUROC **0.78**, and more
importantly AUPRC **0.56**. We lead with AUPRC because on a 22%-positive problem,
plain accuracy is a trap — you'd score 78% just predicting 'no default' for
everyone. And we picked the winner on *validation*, before we ever opened the
test set."

**[1:40 — Slide 6, Cost]**
"Now the deployment thinking. A 0.5 threshold is an accident, not a decision.
Under a realistic 5-to-1 cost ratio, the optimal cutoff is **0.375** — worth a
**1.8%** cut in expected cost per applicant on the test set. Modest, because the
cost curve is flat near the minimum — but the operating point is now a business
decision with a stated cost ratio, not an unexamined default."

**[2:05 — Slide 7, Fairness — slow down here]**
"This is our differentiator. We audited across sex and found a real gap:
equal-opportunity difference **0.016**, disparate impact **0.94**. Per-group
thresholds — a deploy-time control that never touches the model — narrow that to
**0.013**, and lift disparate impact to **0.97**.

Here's the part we want to be straight about: equalising true-positive rate
alone made the *false*-positive gap slightly worse, 0.020 to 0.026. That's the
known limitation of optimising equal opportunity in isolation, and it's why the
next step is equalized odds, which constrains both rates together. We'd rather
show you the metric that moved the wrong way than hide it."

**[2:30 — Slide 8, Explainability]**
"And every decision is explainable. SHAP turns the model into ranked reason
codes — for a high-risk applicant, it points to recent payment delays and months
in arrears. That's exactly the adverse-action disclosure a lender is required to
give."

**[2:45 — Slide 9 + 11, Deploy & close]**
"It's not a notebook — it's a FastAPI service and a live dashboard you can click.
Most teams stop at AUC. We shipped the model a bank could actually deploy:
compared, cost-aware, fair, explainable, and reproducible. That's FairCredit.
Thank you."

---

### Q&A prep — likely judge questions

- **"Why not just drop SEX to be fair?"** → "Fairness-through-unawareness fails —
  proxies leak the attribute. You have to keep it, measure the bias, and correct
  it, which is what we did."
- **"Isn't 0.78 AUROC low?"** → "It's in line with the published benchmark for
  this dataset (Yeh & Lien). We optimised for honest, leakage-free numbers and for
  AUPRC/cost/fairness, not for a vanity AUC."
- **"Does the fairness fix hurt accuracy?"** → "Ranking quality is untouched —
  AUPRC is identical, because we only move per-group operating points, not the
  model. The cost is a slightly wider FPR gap, which we report openly."
- **"Your fairness gain looks small."** → "It is — because the baseline gap was
  already small (0.016, and disparate impact 0.94 already passes the
  four-fifths rule). We'd rather report a real 20% narrowing than manufacture a
  dramatic one. The contribution is the audit-and-mitigate *machinery*, which
  works the same way when the gap is large."
- **"Would this work in India?"** → "The method transfers directly; you'd retrain
  on local, current data and re-run the same audit."
