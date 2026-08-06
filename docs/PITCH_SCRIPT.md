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
**22%** default rate. We engineered **42** behavioural features from the
six-month repayment panel. The whole pipeline is leakage-safe: the test set is
touched exactly once, so every number you're about to see is honest."

**[1:10 — Slide 5, Comparative analysis]**
"We compared four models, simplest to strongest. The linear baseline is fine, but
gradient boosting adds real signal. **XGBoost wins** — AUROC **0.78**, and more
importantly AUPRC **0.56**. We lead with AUPRC because on a 22%-positive problem,
plain accuracy is a trap — you'd score 78% just predicting 'no default' for
everyone."

**[1:40 — Slide 6, Cost]**
"Now the deployment thinking. A 0.5 threshold is an accident, not a decision.
Under a realistic 5-to-1 cost ratio, the optimal cutoff is **0.355** — and that
single change cuts expected cost per applicant by **3.2%** on the test set."

**[2:05 — Slide 7, Fairness — slow down here]**
"This is our differentiator. We audited the model across sex and found a bias gap.
Then we mitigated it with per-group thresholds — a deploy-time control that
doesn't touch the model. The equal-opportunity gap dropped from **0.028 to
0.001** — a **97%** reduction — with no loss of ranking power. Disparate impact
went from 0.94 to 0.99."

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
- **"Does the fairness fix hurt accuracy?"** → "No — AUPRC is unchanged; we only
  move per-group operating points, not the model."
- **"Would this work in India?"** → "The method transfers directly; you'd retrain
  on local, current data and re-run the same audit."
