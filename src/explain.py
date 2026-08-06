"""
explain.py
----------
SHAP explanations for the winning tree model. Two levels:
  * Global  : which features drive default risk across the whole portfolio
              (beeswarm summary).
  * Local   : per-applicant reason codes. Under fair-lending rules a lender must
              give an applicant the specific reasons for an adverse decision;
              SHAP values give exactly that, turning a black box into a set of
              "reason codes".
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


def tree_explainer(model, X_background):
    return shap.TreeExplainer(model)


def global_summary(model, X_sample, out_path: str, max_display: int = 15):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):          # some versions return [neg, pos]
        sv = sv[1]
    plt.figure()
    shap.summary_plot(sv, X_sample, max_display=max_display, show=False)
    plt.title("Global feature impact on default risk (SHAP)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    return sv


def top_reason_codes(model, x_row, feature_names, k: int = 5):
    """Return the k features pushing a single applicant toward default."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(x_row.reshape(1, -1))
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv).ravel()
    order = np.argsort(-np.abs(sv))[:k]
    return [(feature_names[i], float(sv[i]), float(x_row[i])) for i in order]
