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


def _tree_model(model):
    """Unwrap to the object SHAP can explain.

    Models reloaded from LightGBM's portable text format arrive wrapped in
    ``persistence.LightGBMNative``, which exposes the raw Booster via
    ``shap_model()``. Everything else is passed through untouched.
    """
    unwrap = getattr(model, "shap_model", None)
    return unwrap() if callable(unwrap) else model


def _positive_class(sv):
    """Normalise SHAP output to a 2-D (n_samples, n_features) array for class 1.

    TreeExplainer's return shape depends on the model family and SHAP version:
    XGBoost gives (n, features); scikit-learn's RandomForest gives a per-class
    (n, features, 2); older versions give a ``[neg, pos]`` list. Collapsing all
    three here keeps the callers honest — the previous code handled only the list
    case, so Random Forest reason codes raised IndexError.
    """
    if isinstance(sv, list):              # [neg, pos]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                      # (n, features, n_classes)
        sv = sv[..., 1]
    return sv


def tree_explainer(model, X_background):
    return shap.TreeExplainer(_tree_model(model))


def global_summary(model, X_sample, out_path: str, max_display: int = 15):
    explainer = shap.TreeExplainer(_tree_model(model))
    sv = _positive_class(explainer.shap_values(X_sample))
    plt.figure()
    shap.summary_plot(sv, X_sample, max_display=max_display, show=False)
    plt.title("Global feature impact on default risk (SHAP)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    return sv


def top_reason_codes(model, x_row, feature_names, k: int = 5):
    """Return the k features pushing a single applicant toward default."""
    x_row = np.asarray(x_row, dtype=float).ravel()
    explainer = shap.TreeExplainer(_tree_model(model))
    sv = _positive_class(explainer.shap_values(x_row.reshape(1, -1))).ravel()
    if sv.shape[0] != len(feature_names):
        raise ValueError(
            f"SHAP returned {sv.shape[0]} values for {len(feature_names)} "
            f"features — unexpected explainer output shape.")
    order = np.argsort(-np.abs(sv))[:k]
    return [(feature_names[i], float(sv[i]), float(x_row[i])) for i in order]
