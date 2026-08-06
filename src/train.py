"""
train.py
--------
Model factory and training. Four models of increasing capacity form the
comparative-analysis spine of the project:

    Logistic Regression  -> transparent linear baseline
    Random Forest        -> non-linear bagging baseline
    XGBoost              -> gradient boosting (industry credit-scoring standard)
    LightGBM             -> gradient boosting, faster / leaf-wise

Every model is class-imbalance aware (~22% positives): linear/bagging models use
balanced class weights; boosters use scale_pos_weight. Logistic Regression is
wrapped in a scaling Pipeline; tree models are scale-invariant and are not.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .data_prep import RANDOM_STATE


def build_models(pos_weight: float) -> dict:
    """Return the four candidate estimators, imbalance-aware."""
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                C=0.1, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=pos_weight, eval_metric="aucpr",
            n_jobs=-1, random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(
            n_estimators=600, max_depth=5, num_leaves=31, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=pos_weight, n_jobs=-1, verbosity=-1,
            random_state=RANDOM_STATE),
    }


def pos_weight_from(y) -> float:
    """scale_pos_weight = (#negatives / #positives)."""
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    return neg / max(pos, 1.0)


def fit_all(models: dict, X_train, y_train) -> dict:
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted
