"""Feature attributions from XGBoost's built-in TreeSHAP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from crimebase.evaluate import CLASS_NAMES
from crimebase.models import Fitted


def contributions(model: Fitted, X: pd.DataFrame) -> np.ndarray:
    """TreeSHAP values, shape (rows, classes, features + 1); the last slot is the bias.

    Per class, the values plus the bias sum to that class's raw margin, so each one is the
    feature's push on the log-odds of the class, relative to the average prediction.
    """
    if model.name != "xgboost":
        raise ValueError("contributions() needs an xgboost model")
    booster = model.model.get_booster()
    return booster.predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)


def importance(contrib: np.ndarray, features: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature: one column per class and a `total` averaged over classes."""
    mean_abs = np.abs(contrib[:, :, :-1]).mean(axis=0)  # (classes, features)
    table = pd.DataFrame(mean_abs.T, index=features, columns=CLASS_NAMES)
    table["total"] = table.mean(axis=1)
    return table.sort_values("total", ascending=False)


def signed_effect(contrib: np.ndarray, X: pd.DataFrame, class_index: int) -> pd.DataFrame:
    """For each numeric feature, how a high value pushes one class.

    Returns the rank correlation between the feature's value and its SHAP value for the
    class (constant features are skipped). Positive means higher values push the class up,
    negative means down. A tree model has no single sign, so this is a summary, not a
    coefficient.
    """
    rows = {}
    for j, col in enumerate(X.columns):
        if isinstance(X[col].dtype, pd.CategoricalDtype) or X[col].nunique() < 2:
            continue  # categories have no order; a constant feature has no correlation
        shap_col = pd.Series(contrib[:, class_index, j], index=X.index)
        rows[col] = X[col].corr(shap_col, method="spearman")
    return pd.Series(rows, name=CLASS_NAMES[class_index]).dropna().to_frame()
