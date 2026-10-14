"""One interface over XGBoost, LightGBM, CatBoost and logistic regression.

Every model takes the same feature frame (numeric columns plus pandas `category` columns),
trains on balanced class weights, early-stops on a validation log loss when given one, and
returns an (n, 3) probability matrix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

N_CLASSES = 3
N_JOBS = os.cpu_count() or 1
MODEL_NAMES = ["xgboost", "lightgbm", "catboost", "logreg"]
EARLY_STOP = 100
MAX_TREES = 3000

DEFAULTS: dict[str, dict[str, Any]] = {
    "xgboost": dict(
        learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.7,
        min_child_weight=10, reg_lambda=5.0,
    ),
    "lightgbm": dict(
        learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, min_child_samples=40, reg_lambda=5.0,
    ),
    "catboost": dict(learning_rate=0.05, depth=6, l2_leaf_reg=3.0),
    "logreg": dict(C=0.1),
}  # fmt: skip


@dataclass
class Fitted:
    """A trained model plus the tree count early stopping chose (None for logreg)."""

    name: str
    model: Any
    best_iteration: int | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.name == "catboost":
            X = _catboost_frame(X)
        proba = np.asarray(self.model.predict_proba(X), dtype=float)
        # XGBoost computes in float32, so rows sum to 1 only within ~1e-7
        return proba / proba.sum(axis=1, keepdims=True)


def balanced_weights(y) -> np.ndarray:
    """Per-row weights that make every class count equally."""
    return compute_sample_weight("balanced", np.asarray(y))


def _categorical_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def _catboost_frame(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost wants categoricals as strings (a missing category becomes 'nan')."""
    X = X.copy()
    for col in _categorical_columns(X):
        X[col] = X[col].astype(str)
    return X


def fit(
    name: str,
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame | None = None,
    y_val=None,
    params: dict[str, Any] | None = None,
    seed: int = 42,
    n_estimators: int | None = None,
) -> Fitted:
    """Train `name`. With a validation set, trees are added until its log loss stops improving.

    Without one, `n_estimators` trees are fitted (use the best_iteration from an earlier run).
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    p = {**DEFAULTS[name], **(params or {})}
    y_train = np.asarray(y_train).astype(int)
    if set(np.unique(y_train)) != set(range(N_CLASSES)):
        raise ValueError(f"training labels must contain all {N_CLASSES} classes")
    weights = balanced_weights(y_train)
    stop = X_val is not None
    trees = MAX_TREES if stop else (n_estimators or 300)

    if name == "xgboost":
        model = xgb.XGBClassifier(
            objective="multi:softprob", tree_method="hist",
            enable_categorical=True, n_estimators=trees, eval_metric="mlogloss",
            early_stopping_rounds=EARLY_STOP if stop else None,
            n_jobs=N_JOBS, random_state=seed, **p,
        )  # fmt: skip
        kw = {"eval_set": [(X_val, np.asarray(y_val).astype(int))]} if stop else {}
        model.fit(X_train, y_train, sample_weight=weights, verbose=False, **kw)
        best = int(model.best_iteration) + 1 if stop else trees
    elif name == "lightgbm":
        model = lgb.LGBMClassifier(
            objective="multiclass", n_estimators=trees, n_jobs=N_JOBS, random_state=seed,
            verbose=-1, **p,
        )  # fmt: skip
        kw = {}
        if stop:
            kw = {
                "eval_X": X_val,
                "eval_y": np.asarray(y_val).astype(int),
                "callbacks": [lgb.early_stopping(EARLY_STOP, verbose=False)],
            }
        model.fit(X_train, y_train, sample_weight=weights, **kw)
        best = int(model.best_iteration_) if stop else trees
    elif name == "catboost":
        model = CatBoostClassifier(
            loss_function="MultiClass", iterations=trees, thread_count=N_JOBS, random_seed=seed,
            allow_writing_files=False, verbose=False,
            od_type="Iter" if stop else None, od_wait=EARLY_STOP if stop else None, **p,
        )  # fmt: skip
        Xt = _catboost_frame(X_train)
        cats = _categorical_columns(X_train)
        kw = {}
        if stop:
            kw = {"eval_set": (_catboost_frame(X_val), np.asarray(y_val).astype(int))}
        model.fit(Xt, y_train, cat_features=cats, sample_weight=weights, **kw)
        best = int(model.get_best_iteration()) + 1 if stop else trees
    else:
        model, best = _logreg(X_train, p, seed), None
        # these matrices are tiny; one BLAS thread is ~6x faster than four fighting over them
        with threadpool_limits(limits=1):
            model.fit(X_train, y_train, clf__sample_weight=weights)
    return Fitted(name, model, best)


def _logreg(X: pd.DataFrame, p: dict[str, Any], seed: int) -> Pipeline:
    """Linear baseline: median-impute with missing flags, quantile-normalise, one-hot the rest.

    The quantile transform keeps heavy-tailed ratios (up to 10^4) from dominating the fit.
    """
    cats = _categorical_columns(X)
    nums = [c for c in X.columns if c not in cats]
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        (
                            "quantile",
                            QuantileTransformer(
                                n_quantiles=min(1000, len(X)),
                                output_distribution="normal",
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
                nums,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats),
        ]
    )
    clf = LogisticRegression(max_iter=2000, random_state=seed, **p)
    return Pipeline([("prep", prep), ("clf", clf)])
