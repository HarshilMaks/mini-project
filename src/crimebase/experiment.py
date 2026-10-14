"""The evaluation protocol: fit on the past, score once on the held-out future."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from crimebase.config import Config
from crimebase.evaluate import (
    binned_proba,
    mean_reversion_pred,
    multiclass_scores,
    persistence_pred,
    prior_proba,
)
from crimebase.features import FEATURES
from crimebase.models import fit
from crimebase.splits import dev_mask, holdout_mask, train_mask, val_mask


@dataclass
class Holdout:
    """The held-out rows and the columns every comparison needs."""

    index: pd.Index
    y: np.ndarray
    g1: pd.Series
    year: np.ndarray
    groups: np.ndarray  # state + year, the unit the bootstrap resamples


def holdout_of(m: pd.DataFrame, cfg: Config, target: str) -> Holdout:
    rows = m[holdout_mask(m, cfg.split, target)]
    return Holdout(
        index=rows.index,
        y=rows[target].astype(int).to_numpy(),
        g1=rows["g1"],
        year=rows["year"].to_numpy(),
        groups=(rows["state_code"].astype(str) + "-" + rows["year"].astype(str)).to_numpy(),
    )


def final_fit_predict(
    name: str, m: pd.DataFrame, cfg: Config, target: str, params: dict[str, Any] | None = None
) -> tuple[np.ndarray, int | None]:
    """Fit `name` the way the report does and return its holdout probabilities.

    1. Train on the training years and early-stop on the validation years to fix the tree
       count. 2. Refit on train + validation with that count. 3. Predict the held-out years,
       which no earlier step has seen.
    """
    X, y, s = m[FEATURES], m[target], cfg.split
    tr, va, dev, te = (f(m, s, target) for f in (train_mask, val_mask, dev_mask, holdout_mask))
    stopped = fit(name, X[tr], y[tr], X[va], y[va], params, cfg.seed)
    trees = stopped.best_iteration
    final = fit(name, X[dev], y[dev], params=params, seed=cfg.seed, n_estimators=trees)
    return final.predict_proba(X[te]), trees


def baseline_results(
    m: pd.DataFrame, cfg: Config, target: str = "y_trend"
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """Scores for the no-ML opponents on the holdout, plus probabilities where they exist.

    `majority` is the training class mix; `persistence` and `mean_reversion` are fixed
    rules on last year's growth; `binned_g1` learns the class mix per last-year direction.
    """
    s, band = cfg.split, cfg.task.trend_band
    dev, te = dev_mask(m, s, target), holdout_mask(m, s, target)
    y_dev = m.loc[dev, target].astype(int).to_numpy()
    y_te = m.loc[te, target].astype(int).to_numpy()
    g1_te = m.loc[te, "g1"]
    fallback = int(np.bincount(y_dev).argmax())

    probas = {
        "majority": prior_proba(y_dev, len(y_te)),
        "binned_g1": binned_proba(m.loc[dev, "g1"], y_dev, g1_te, band),
    }
    scores = {name: multiclass_scores(y_te, proba=p) for name, p in probas.items()}
    scores["persistence"] = multiclass_scores(y_te, pred=persistence_pred(g1_te, band, fallback))
    scores["mean_reversion"] = multiclass_scores(
        y_te, pred=mean_reversion_pred(g1_te, band, fallback)
    )
    return scores, probas


def rule_as_proba(pred: np.ndarray) -> np.ndarray:
    """A hard rule prediction as a proper probability matrix, so it can be bootstrapped."""
    return np.eye(3)[pred] * 0.94 + 0.02
