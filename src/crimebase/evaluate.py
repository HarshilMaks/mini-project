"""Metrics, baselines and bootstrap intervals for the trend classifiers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

CLASS_NAMES = ["down", "stable", "up"]
LABELS = [0, 1, 2]
Metric = Callable[[np.ndarray, np.ndarray], float]


def multiclass_scores(y_true, proba=None, pred=None) -> dict[str, float]:
    """Scores for the 3-class task. Give `proba` (n x 3) for the full set, or only `pred`.

    Log loss and one-vs-rest AUC need probabilities and are NaN for hard predictions.
    """
    y_true = np.asarray(y_true)
    if proba is not None:
        proba = np.asarray(proba)
        pred = proba.argmax(axis=1)
    pred = np.asarray(pred)
    per_class = f1_score(y_true, pred, labels=LABELS, average=None, zero_division=0)
    scores = {
        "accuracy": accuracy_score(y_true, pred),
        "precision_macro": precision_recall_fscore_support(
            y_true, pred, labels=LABELS, average="macro", zero_division=0
        )[0],
        "recall_macro": balanced_accuracy_score(y_true, pred),  # macro recall is balanced accuracy
        "macro_f1": f1_score(y_true, pred, labels=LABELS, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        **{f"f1_{name}": float(f) for name, f in zip(CLASS_NAMES, per_class, strict=True)},
        "log_loss": np.nan,
        "auc_ovr": np.nan,
    }
    if proba is not None:
        scores["log_loss"] = log_loss(y_true, proba, labels=LABELS)
        scores["auc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr", labels=LABELS)
    return {k: float(v) for k, v in scores.items()}


def binary_scores(y_true, p) -> dict[str, float]:
    """Scores for a rare-event view (drop / surge). `p` is the event probability."""
    y_true, p = np.asarray(y_true), np.asarray(p)
    base = float(y_true.mean())
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "base_rate": base,
        "pr_lift": float(average_precision_score(y_true, p) / base),
        "brier": float(brier_score_loss(y_true, p)),
    }


# --- baselines -------------------------------------------------------------------------


def prior_proba(y_train, n: int) -> np.ndarray:
    """Training class mix for every row: the majority-class baseline in probability form."""
    prior = np.bincount(np.asarray(y_train).astype(int), minlength=3) / len(y_train)
    return np.tile(prior, (n, 1))


def direction_of(g1: pd.Series, band: float) -> np.ndarray:
    """Last year's move as 0 down / 1 stable / 2 up, or -1 when unknown."""
    g = g1.to_numpy(dtype=float)
    out = np.where(g > band, 2, np.where(g < -band, 0, 1))
    return np.where(np.isnan(g), -1, out)


def persistence_pred(g1: pd.Series, band: float, fallback: int) -> np.ndarray:
    """Predict that next year moves the same way as last year."""
    d = direction_of(g1, band)
    return np.where(d < 0, fallback, d)


def mean_reversion_pred(g1: pd.Series, band: float, fallback: int) -> np.ndarray:
    """Predict that next year undoes last year's move (up becomes down, down becomes up)."""
    d = direction_of(g1, band)
    flipped = np.where(d == 0, 2, np.where(d == 2, 0, d))
    return np.where(d < 0, fallback, flipped)


def binned_proba(train_g1, y_train, test_g1, band: float, alpha: float = 1.0) -> np.ndarray:
    """One-feature model: class mix conditional on last year's direction (plus 'unknown').

    This is the fair "no machine learning" opponent: it learns persistence or mean
    reversion from the training years instead of assuming one.
    """
    d_train, d_test = direction_of(train_g1, band), direction_of(test_g1, band)
    y = np.asarray(y_train).astype(int)
    table = {}
    for d in (-1, 0, 1, 2):
        counts = np.bincount(y[d_train == d], minlength=3) + alpha
        table[d] = counts / counts.sum()
    return np.vstack([table[d] for d in d_test])


# --- bootstrap -------------------------------------------------------------------------


def _resample_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    codes, uniques = pd.factorize(groups)
    by_group = [np.flatnonzero(codes == i) for i in range(len(uniques))]
    pick = rng.integers(0, len(uniques), len(uniques))
    return np.concatenate([by_group[i] for i in pick])


def bootstrap_ci(
    y_true, scores, metric: Metric, groups, n_boot: int = 500, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile interval of `metric(y_true, scores)` resampling whole `groups`.

    Rows of one state and year share shocks, so resampling single rows would make the
    interval too narrow; callers pass `groups` such as state_code + year.
    """
    y_true, scores, groups = np.asarray(y_true), np.asarray(scores), np.asarray(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = _resample_indices(groups, rng)
        try:
            value = metric(y_true[idx], scores[idx])
        except ValueError:  # a resample missing a class can make a metric undefined
            continue
        if not np.isnan(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    lo, hi = np.quantile(values, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(
    y_true, scores_a, scores_b, metric: Metric, groups, n_boot: int = 500, seed: int = 0
) -> dict[str, float]:
    """metric(A) - metric(B) on the same resamples, with a 95% interval and P(A > B).

    Using the same resamples for both models removes the shared noise, which is what lets
    a small but real gap show up (or be shown to be indistinguishable from zero).
    """
    y_true, a, b = np.asarray(y_true), np.asarray(scores_a), np.asarray(scores_b)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _resample_indices(groups, rng)
        try:
            d = metric(y_true[idx], a[idx]) - metric(y_true[idx], b[idx])
        except ValueError:
            continue
        if not np.isnan(d):
            diffs.append(d)
    point = float(metric(y_true, a) - metric(y_true, b))
    if not diffs:
        return {"diff": point, "lo": float("nan"), "hi": float("nan"), "p_a_better": float("nan")}
    diffs = np.asarray(diffs)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "diff": point,
        "lo": float(lo),
        "hi": float(hi),
        "p_a_better": float((diffs > 0).mean()),
    }


def macro_f1_from_proba(y_true, proba) -> float:
    """Macro-F1 of the argmax class; a class absent from a small sample scores 0 quietly."""
    pred = np.asarray(proba).argmax(axis=1)
    return f1_score(y_true, pred, labels=LABELS, average="macro", zero_division=0)


def auc_from_proba(y_true, proba) -> float:
    """One-vs-rest ROC-AUC averaged over the three classes (bootstrap-friendly signature).

    NaN when a class is absent from `y_true`, where AUC is undefined.
    """
    if len(np.unique(y_true)) < len(LABELS):
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(proba), multi_class="ovr", labels=LABELS))


def classification_table(y_true, pred) -> pd.DataFrame:
    """Per-class precision, recall, F1 and support, laid out like scikit-learn's report.

    Rows: down, stable, up, then accuracy, macro average and weighted average.
    """
    p, r, f, n = precision_recall_fscore_support(
        y_true, pred, labels=LABELS, average=None, zero_division=0
    )
    table = pd.DataFrame(
        {"precision": p, "recall": r, "f1-score": f, "support": n}, index=CLASS_NAMES
    )
    total = int(n.sum())
    macro = precision_recall_fscore_support(
        y_true, pred, labels=LABELS, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, pred, labels=LABELS, average="weighted", zero_division=0
    )
    table.loc["accuracy"] = [np.nan, np.nan, accuracy_score(y_true, pred), total]
    table.loc["macro avg"] = [*macro[:3], total]
    table.loc["weighted avg"] = [*weighted[:3], total]
    table["support"] = table["support"].astype(int)
    return table


def roc_points(y_true, proba) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """One-vs-rest ROC curve per class: {class: (false positive rate, true positive rate, AUC)}."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    out = {}
    for k, name in enumerate(CLASS_NAMES):
        hit = (y_true == k).astype(int)
        fpr, tpr, _ = roc_curve(hit, proba[:, k])
        out[name] = (fpr, tpr, float(roc_auc_score(hit, proba[:, k])))
    return out
