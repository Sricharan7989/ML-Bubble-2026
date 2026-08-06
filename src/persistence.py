"""
persistence.py
--------------
Portable model persistence.

Pickling a fitted XGBoost or LightGBM estimator embeds the library's internal
binary booster buffer. That buffer is not stable across library versions or
platforms, so a pickle written on one machine routinely fails to load on another
with "input stream corrupted" — which is exactly the failure this project hit.

The fix is to serialise each model in the format its own library guarantees to be
portable:

    XGBoost      -> native JSON  (XGBClassifier.save_model / load_model)
    LightGBM     -> native text  (Booster.save_model / lgb.Booster(model_file=))
    scikit-learn -> joblib       (no native format; joblib is the supported route)

The canonical training feature order is written alongside the models as
``feature_names.json``. Serving code (API, dashboard) builds its feature frame
against that file, so inference can never silently drift from training column
order.
"""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

FEATURES_FILE = "feature_names.json"


def slug(name: str) -> str:
    """'Random Forest' -> 'random_forest' (the on-disk model basename)."""
    return name.replace(" ", "_").lower()


class LightGBMNative:
    """scikit-learn-style wrapper around a native LightGBM ``Booster``.

    LightGBM can reload a Booster from its portable text format but not back into
    an ``LGBMClassifier``, so we supply the small bit of sklearn surface the rest
    of the project uses.

    ``shap_model()`` hands out the raw Booster, because ``shap.TreeExplainer``
    needs the real thing rather than this wrapper. It is deliberately *not*
    called ``booster``: XGBClassifier already exposes a ``booster`` attribute (an
    unset hyper-parameter, i.e. ``None``), so a plain ``getattr(m, "booster")``
    unwrap silently breaks SHAP for XGBoost.
    """

    def __init__(self, booster: lgb.Booster):
        self._booster = booster

    def shap_model(self) -> lgb.Booster:
        return self._booster

    def predict_proba(self, X) -> np.ndarray:
        p = np.asarray(self._booster.predict(X), dtype=float).ravel()
        return np.column_stack([1.0 - p, p])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def save_model(name: str, model, models_dir: Path) -> Path:
    """Serialise one fitted model in its most portable format. Returns the path."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    base = models_dir / slug(name)

    if isinstance(model, XGBClassifier):
        path = base.with_suffix(".json")
        model.save_model(str(path))
    elif isinstance(model, LGBMClassifier):
        path = base.with_suffix(".txt")
        model.booster_.save_model(str(path))
    else:                                   # sklearn estimators / Pipelines
        path = base.with_suffix(".joblib")
        joblib.dump(model, path)
    return path


def load_model(name: str, models_dir: Path):
    """Load a model saved by :func:`save_model`, dispatching on file extension."""
    models_dir = Path(models_dir)
    base = models_dir / slug(name)

    xgb_path = base.with_suffix(".json")
    if xgb_path.exists():
        model = XGBClassifier()
        model.load_model(str(xgb_path))
        return model

    lgbm_path = base.with_suffix(".txt")
    if lgbm_path.exists():
        return LightGBMNative(lgb.Booster(model_file=str(lgbm_path)))

    sk_path = base.with_suffix(".joblib")
    if sk_path.exists():
        return joblib.load(sk_path)

    raise FileNotFoundError(
        f"No saved model for {name!r} in {models_dir} "
        f"(looked for {base.name}.json/.txt/.joblib). Run `py main.py` first.")


def save_feature_names(feature_names, models_dir: Path) -> Path:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / FEATURES_FILE
    path.write_text(json.dumps(list(feature_names), indent=2), encoding="utf-8")
    return path


def load_feature_names(models_dir: Path) -> list[str]:
    path = Path(models_dir) / FEATURES_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — run `py main.py` to train and export models.")
    return json.loads(path.read_text(encoding="utf-8"))
