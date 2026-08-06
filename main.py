"""
main.py — end-to-end pipeline for the ML Bubble 2026 submission.

Runs: data prep -> train 4 models -> comparative evaluation -> cost-sensitive
threshold -> fairness audit + mitigation -> SHAP explainability. Writes every
model, figure, and a machine-readable results.json into ./models and ./outputs.

Reproduce:  python main.py
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import prepare, split, PROTECTED
from src.train import build_models, pos_weight_from, fit_all
from src.evaluate import proba, compare, rank_metrics, threshold_metrics, \
    plot_curves, plot_metric_bars
from src.cost_threshold import optimize, savings_vs_default, plot_cost_curve, \
    COST_FN, COST_FP
from src.fairness import audit, summary_gaps, equal_opportunity_thresholds, \
    audit_grouped_thresholds, plot_fairness
from src.explain import global_summary, top_reason_codes

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
MODELS = Path("models"); MODELS.mkdir(exist_ok=True)
DATA = "data/UCI_Credit_Card.csv"


def main():
    results = {}

    # ---- 1. Data ----------------------------------------------------------
    df, X, y, cols = prepare(DATA)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y)
    sex_val = Xva[PROTECTED].to_numpy()
    sex_te = Xte[PROTECTED].to_numpy()
    results["dataset"] = {
        "n_samples": int(len(X)), "n_features": int(len(cols)),
        "default_rate": round(float(y.mean()), 4),
        "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
    }
    print(f"[data] {len(X)} rows, {len(cols)} features, "
          f"default rate {y.mean():.3f}")

    # ---- 2. Train four models --------------------------------------------
    models = build_models(pos_weight_from(ytr))
    fitted = fit_all(models, Xtr, ytr)
    for name, m in fitted.items():
        with open(MODELS / f"{name.replace(' ', '_').lower()}.pkl", "wb") as f:
            pickle.dump(m, f)
    print(f"[train] fitted + saved {len(fitted)} models")

    # ---- 3. Pick winner by validation AUPRC ------------------------------
    val_scores = {n: rank_metrics(yva, proba(m, Xva))["AUPRC"]
                  for n, m in fitted.items()}
    winner = max(val_scores, key=val_scores.get)
    results["val_auprc"] = val_scores
    results["winner"] = winner
    print(f"[select] winner by val AUPRC: {winner} ({val_scores[winner]:.4f})")

    win_model = fitted[winner]
    p_val = proba(win_model, Xva)
    p_te = proba(win_model, Xte)

    # ---- 4. Cost-sensitive threshold (tuned on val, applied to test) ------
    thr_opt, _, grid, costs = optimize(yva, p_val, COST_FN, COST_FP)
    plot_cost_curve(grid, costs, thr_opt, OUT / "cost_threshold.png")
    results["cost_threshold"] = {
        "ratio_fn_fp": f"{COST_FN:.0f}:{COST_FP:.0f}",
        "optimal_threshold": round(thr_opt, 4),
        "test": savings_vs_default(yte, p_te, thr_opt, COST_FN, COST_FP),
    }
    print(f"[cost] optimal threshold {thr_opt:.3f} | "
          f"savings vs 0.5 on test: "
          f"{results['cost_threshold']['test']['relative_savings']*100:.1f}%")

    # ---- 5. Comparative evaluation on test (all at cost-opt threshold) ----
    thr_map = {n: thr_opt for n in fitted}
    comp = compare(fitted, Xte, yte, thr_map)
    comp.to_csv(OUT / "model_comparison.csv", index=False)
    plot_curves(fitted, Xte, yte, OUT / "roc_pr_curves.png")
    plot_metric_bars(comp, OUT / "metric_bars.png")
    results["comparison"] = comp.to_dict(orient="records")
    print("[eval] comparison table:\n", comp.to_string(index=False))

    # ---- 6. Fairness audit + mitigation (winner) -------------------------
    before = audit(yte, p_te, sex_te, thr_opt)
    gaps_before = summary_gaps(before)
    # target TPR = overall recall at cost-opt threshold, equalised across groups
    overall_tpr = threshold_metrics(yte, p_te, thr_opt)["recall"]
    eo_thr = equal_opportunity_thresholds(yva, p_val, sex_val, overall_tpr)
    after = audit_grouped_thresholds(yte, p_te, sex_te, eo_thr)
    gaps_after = summary_gaps(after.rename(columns={}))
    plot_fairness(before, after, OUT / "fairness_audit.png")
    results["fairness"] = {
        "target_tpr": round(float(overall_tpr), 4),
        "group_thresholds": {int(k): v for k, v in eo_thr.items()},
        "before": before.to_dict(orient="records"),
        "after": after.to_dict(orient="records"),
        "gaps_before": gaps_before,
        "gaps_after": gaps_after,
    }
    print(f"[fairness] equal-opportunity gap: "
          f"{gaps_before['equal_opportunity_diff']:.3f} -> "
          f"{gaps_after['equal_opportunity_diff']:.3f}")

    # ---- 7. SHAP explainability ------------------------------------------
    try:
        expl_model = win_model  # tree model expected
        Xsamp = Xte.sample(min(1000, len(Xte)), random_state=42)
        global_summary(expl_model, Xsamp, OUT / "shap_summary.png")
        # a concrete high-risk applicant -> reason codes
        idx = int(np.argmax(p_te))
        codes = top_reason_codes(expl_model, Xte.iloc[idx].to_numpy(), cols, k=5)
        results["explainability"] = {
            "example_applicant_risk": round(float(p_te[idx]), 4),
            "reason_codes": [{"feature": f, "shap": round(s, 4),
                              "value": round(v, 4)} for f, s, v in codes],
        }
        print("[shap] global summary + reason codes written")
    except Exception as e:  # SHAP only supports tree winners cleanly
        results["explainability"] = {"error": str(e)}
        print(f"[shap] skipped: {e}")

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[done] artifacts in ./outputs and ./models")
    return results


if __name__ == "__main__":
    main()
