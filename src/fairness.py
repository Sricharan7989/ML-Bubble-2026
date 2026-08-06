"""
fairness.py
-----------
Bias audit across the protected attribute SEX (1 = male, 2 = female) and a
lightweight mitigation. Most credit-scoring submissions optimise AUC and stop.
The reason lending models are regulated (ECOA in the US, RBI fair-lending
expectations in India) is that a high-AUC model can still deny protected groups
at systematically different rates. We measure that and then reduce it.

Group fairness metrics reported (positive class = "will default"):
  * Selection rate      : P(flagged as default | group)   -> demographic parity
  * TPR (equal opp.)    : recall among true defaulters      -> equal opportunity
  * FPR                 : good customers wrongly flagged
  * Disparate impact    : min(selection_rate) / max(selection_rate)  (>=0.8 rule)

Mitigation: per-group decision thresholds chosen to equalise TPR (equal
opportunity) while holding the global operating point fixed. This shrinks the
TPR gap without touching the trained model — a deploy-time control.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

SEX_LABELS = {1: "Male", 2: "Female"}


def _group_stats(y_true, yhat) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    n = tn + fp + fn + tp
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    sel = (tp + fp) / n if n else 0.0
    return {"n": int(n), "selection_rate": sel, "TPR": tpr, "FPR": fpr}


def audit(y_true, p, sex, thr: float) -> pd.DataFrame:
    yhat = (p >= thr).astype(int)
    rows = []
    for code, label in SEX_LABELS.items():
        m = sex == code
        s = _group_stats(np.asarray(y_true)[m], yhat[m])
        rows.append({"group": label, **{k: round(v, 4) if isinstance(v, float) else v
                                        for k, v in s.items()}})
    return pd.DataFrame(rows)


def summary_gaps(tbl: pd.DataFrame) -> dict:
    sel = tbl["selection_rate"].values
    tpr = tbl["TPR"].values
    fpr = tbl["FPR"].values
    di = (sel.min() / sel.max()) if sel.max() > 0 else 1.0
    return {
        "demographic_parity_diff": round(float(abs(sel[0] - sel[1])), 4),
        "equal_opportunity_diff": round(float(abs(tpr[0] - tpr[1])), 4),
        "fpr_diff": round(float(abs(fpr[0] - fpr[1])), 4),
        "disparate_impact_ratio": round(float(di), 4),
    }


def equal_opportunity_thresholds(y_true, p, sex, target_tpr: float):
    """Pick a per-group threshold that lands each group closest to target_tpr."""
    out = {}
    y_true = np.asarray(y_true)
    for code in SEX_LABELS:
        m = sex == code
        yt, pg = y_true[m], p[m]
        best_thr, best_gap = 0.5, 1e9
        for thr in np.linspace(0.05, 0.95, 181):
            yhat = (pg >= thr).astype(int)
            tp = int(((yhat == 1) & (yt == 1)).sum())
            fn = int(((yhat == 0) & (yt == 1)).sum())
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            gap = abs(tpr - target_tpr)
            if gap < best_gap:
                best_gap, best_thr = gap, thr
        out[code] = round(float(best_thr), 4)
    return out


def audit_grouped_thresholds(y_true, p, sex, thr_by_group: dict) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    rows = []
    for code, label in SEX_LABELS.items():
        m = sex == code
        yhat = (p[m] >= thr_by_group[code]).astype(int)
        s = _group_stats(y_true[m], yhat)
        rows.append({"group": label, "threshold": thr_by_group[code],
                     **{k: round(v, 4) if isinstance(v, float) else v
                        for k, v in s.items()}})
    return pd.DataFrame(rows)


def plot_fairness(before: pd.DataFrame, after: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    metrics = [("selection_rate", "Selection rate\n(demographic parity)"),
               ("TPR", "TPR / Recall\n(equal opportunity)"),
               ("FPR", "FPR\n(good customers flagged)")]
    xb = np.arange(2)
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(xb - 0.2, before[col], 0.4, label="Single threshold",
               color="#94a3b8")
        ax.bar(xb + 0.2, after[col], 0.4, label="Equal-opportunity thresholds",
               color="#10b981")
        ax.set_xticks(xb)
        ax.set_xticklabels(before["group"])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(before[col].max(), after[col].max()) * 1.25 + 1e-3)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Fairness Audit Across Sex — Before vs After Mitigation",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
