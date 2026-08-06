"""
evaluate.py
-----------
Metrics and comparison plots. We deliberately foreground AUPRC and recall on the
positive (default) class, because on a 22%-positive problem, plain accuracy is
misleading: a model predicting "no default" for everyone already scores ~78%.
Reported per model:
    AUROC, AUPRC (average precision), Brier (calibration),
    plus precision / recall / F1 at a chosen operating threshold.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)

PALETTE = {
    "Logistic Regression": "#6366f1",
    "Random Forest": "#10b981",
    "XGBoost": "#f59e0b",
    "LightGBM": "#ef4444",
}


def proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def threshold_metrics(y_true, p, thr: float) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
    return {
        "threshold": round(thr, 4),
        "precision": round(precision_score(y_true, yhat, zero_division=0), 4),
        "recall": round(recall_score(y_true, yhat, zero_division=0), 4),
        "f1": round(f1_score(y_true, yhat, zero_division=0), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def rank_metrics(y_true, p) -> dict:
    return {
        "AUROC": round(roc_auc_score(y_true, p), 4),
        "AUPRC": round(average_precision_score(y_true, p), 4),
        "Brier": round(brier_score_loss(y_true, p), 4),
    }


def compare(fitted: dict, X_test, y_test, thresholds: dict) -> pd.DataFrame:
    rows = []
    for name, model in fitted.items():
        p = proba(model, X_test)
        r = rank_metrics(y_test, p)
        thr = thresholds.get(name, 0.5)
        tm = threshold_metrics(y_test, p, thr)
        rows.append({"model": name, **r,
                     "precision@thr": tm["precision"],
                     "recall@thr": tm["recall"],
                     "f1@thr": tm["f1"],
                     "threshold": tm["threshold"]})
    df = pd.DataFrame(rows).sort_values("AUPRC", ascending=False).reset_index(drop=True)
    return df


def plot_curves(fitted: dict, X_test, y_test, out_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    base = float(np.mean(y_test))
    for name, model in fitted.items():
        p = proba(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, p)
        prec, rec, _ = precision_recall_curve(y_test, p)
        c = PALETTE.get(name, "#333")
        ax1.plot(fpr, tpr, color=c, lw=2,
                 label=f"{name} (AUROC {roc_auc_score(y_test, p):.3f})")
        ax2.plot(rec, prec, color=c, lw=2,
                 label=f"{name} (AUPRC {average_precision_score(y_test, p):.3f})")
    ax1.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax1.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
            title="ROC Curves")
    ax1.legend(fontsize=8, loc="lower right")
    ax2.axhline(base, ls="--", color="#999", lw=1,
                label=f"No-skill ({base:.2f})")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax2.legend(fontsize=8, loc="upper right")
    fig.suptitle("Model Comparison — Discrimination & Ranking Quality",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(comp: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ["AUROC", "AUPRC", "recall@thr", "f1@thr"]
    x = np.arange(len(comp))
    w = 0.2
    colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444"]
    for i, m in enumerate(metrics):
        ax.bar(x + i * w, comp[m], w, label=m, color=colors[i])
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(comp["model"], rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Comparative Performance Across Models", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
