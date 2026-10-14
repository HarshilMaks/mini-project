"""Hyper-parameter search with Optuna over expanding-window time folds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss

from crimebase.config import Config
from crimebase.features import FEATURES
from crimebase.models import DEFAULTS, balanced_weights, fit
from crimebase.splits import cv_folds

optuna.logging.set_verbosity(optuna.logging.WARNING)

# a one-parameter search converges long before 40 trials
MAX_TRIALS = {"logreg": 12}


def search_space(name: str, trial: optuna.Trial) -> dict[str, Any]:
    """Parameters to tune per model. Ranges bracket the DEFAULTS in models.py."""
    if name == "xgboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        }
    if name == "lightgbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        }
    if name == "catboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
        }
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 10, log=True)}
    raise ValueError(f"no search space for {name!r}")


def default_trial(name: str) -> dict[str, Any]:
    """The DEFAULTS as a trial inside the search space (an unset L1 penalty becomes its minimum)."""
    base = {k: v for k, v in DEFAULTS[name].items() if k != "subsample_freq"}
    if name in ("xgboost", "lightgbm"):
        base["reg_alpha"] = 1e-3
    return base


@dataclass
class TuneResult:
    name: str
    best_params: dict[str, Any]
    best_value: float  # mean CV balanced log loss of the best trial
    default_value: float  # the same score for the default parameters (trial 0)
    trials: pd.DataFrame
    n_folds: int


def cv_score(
    name: str, X: pd.DataFrame, y: pd.Series, folds, params: dict[str, Any], seed: int,
    report=None,
) -> float:  # fmt: skip
    """Mean class-balanced log loss over the folds (lower is better).

    Each fold early-stops on its own validation window, a mild optimism that is the same for
    every candidate; the held-out test years take no part in any of it.
    """
    scores = []
    for i, (train, val) in enumerate(folds):
        model = fit(name, X.iloc[train], y.iloc[train], X.iloc[val], y.iloc[val], params, seed)
        proba = model.predict_proba(X.iloc[val])
        y_val = y.iloc[val].astype(int).to_numpy()
        scores.append(
            log_loss(y_val, proba, labels=[0, 1, 2], sample_weight=balanced_weights(y_val))
        )
        if report is not None:
            report(float(np.mean(scores)), i)
    return float(np.mean(scores))


def tune(
    name: str, m: pd.DataFrame, cfg: Config, target: str = "y_trend", n_trials: int = 40,
    seed: int | None = None,
) -> TuneResult:  # fmt: skip
    seed = cfg.seed if seed is None else seed
    n_trials = min(n_trials, MAX_TRIALS.get(name, n_trials))
    folds = cv_folds(m, cfg.split, target)
    X, y = m[FEATURES], m[target]

    def objective(trial: optuna.Trial) -> float:
        params = search_space(name, trial)

        def report(value: float, step: int) -> None:
            trial.report(value, step)
            if step < len(folds) - 1 and trial.should_prune():
                raise optuna.TrialPruned()

        return cv_score(name, X, y, folds, params, seed, report)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.enqueue_trial(default_trial(name))
    study.optimize(objective, n_trials=n_trials)

    trials = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    trials.columns = [c.replace("params_", "") for c in trials.columns]
    first = study.trials[0]
    return TuneResult(
        name=name,
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        default_value=float(first.value),
        trials=trials,
        n_folds=len(folds),
    )
