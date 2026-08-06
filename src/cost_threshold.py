"""
cost_threshold.py
-----------------
A default at 0.5 is an accident of implementation, not a business decision. In
lending the two error types cost very different amounts:

    False Negative (approve someone who defaults) -> lose principal   : COST_FN
    False Positive (reject a customer who repays)  -> lose margin/opp. : COST_FP

We sweep the threshold on the VALIDATION set, compute expected cost per applicant
under a cost ratio, and pick the minimiser. The chosen threshold is then applied
unchanged to the untouched TEST set — so the number we report is honest.
Default ratio COST_FN : COST_FP = 5 : 1 (a bad loan hurts ~5x a lost-margin
rejection); this is a single tunable business input, exposed here.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

COST_FN = 5.0
COST_FP = 1.0


def expected_cost(y_true, p, thr, cost_fn=COST_FN, cost_fp=COST_FP) -> float:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    return (cost_fn * fn + cost_fp * fp) / len(y_true)


def optimize(y_val, p_val, cost_fn=COST_FN, cost_fp=COST_FP):
    grid = np.linspace(0.05, 0.95, 181)
    costs = [expected_cost(y_val, p_val, t, cost_fn, cost_fp) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i]), grid, np.array(costs)


def savings_vs_default(y_true, p, thr_opt, cost_fn=COST_FN, cost_fp=COST_FP):
    c_default = expected_cost(y_true, p, 0.5, cost_fn, cost_fp)
    c_opt = expected_cost(y_true, p, thr_opt, cost_fn, cost_fp)
    rel = (c_default - c_opt) / c_default if c_default else 0.0
    return {"cost@0.5": round(c_default, 4), "cost@opt": round(c_opt, 4),
            "relative_savings": round(rel, 4)}


def plot_cost_curve(grid, costs, thr_opt, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, costs, color="#6366f1", lw=2)
    ax.axvline(0.5, ls="--", color="#94a3b8", label="Default threshold (0.5)")
    ax.axvline(thr_opt, ls="--", color="#ef4444",
               label=f"Cost-optimal ({thr_opt:.2f})")
    ax.set(xlabel="Decision threshold", ylabel="Expected cost per applicant",
           title="Cost-Sensitive Threshold Selection (COST_FN : COST_FP = 5 : 1)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
