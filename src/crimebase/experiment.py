"""The evaluation protocol: fit on the past, score once on the held-out future."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from crimebase.config import Config
from crimebase.evaluate import (
    auc_from_proba,
    binned_proba,
    bootstrap_ci,
    macro_f1_from_proba,
    mean_reversion_pred,
    multiclass_scores,
    paired_bootstrap_diff,
    persistence_pred,
    prior_proba,
)
from crimebase.features import FEATURES, build_matrix
from crimebase.models import MODEL_NAMES, Fitted, fit
from crimebase.splits import dev_mask, holdout_mask, train_mask, val_mask
from crimebase.tuning import TuneResult, tune


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


def final_fit(
    name: str, m: pd.DataFrame, cfg: Config, target: str, params: dict[str, Any] | None = None
) -> tuple[Fitted, int | None]:
    """Fit `name` the way the report does; return the final model and its tree count.

    1. Train on the training years and early-stop on the validation years to fix the tree
       count. 2. Refit on train + validation with that count. The held-out years play no part.
    """
    X, y, s = m[FEATURES], m[target], cfg.split
    tr, va, dev = (f(m, s, target) for f in (train_mask, val_mask, dev_mask))
    stopped = fit(name, X[tr], y[tr], X[va], y[va], params, cfg.seed)
    trees = stopped.best_iteration
    return fit(name, X[dev], y[dev], params=params, seed=cfg.seed, n_estimators=trees), trees


def final_fit_predict(
    name: str, m: pd.DataFrame, cfg: Config, target: str, params: dict[str, Any] | None = None
) -> tuple[np.ndarray, int | None]:
    """`final_fit`, then predict the held-out years once (no earlier step has seen them)."""
    final, trees = final_fit(name, m, cfg, target, params)
    te = holdout_mask(m, cfg.split, target)
    return final.predict_proba(m.loc[te, FEATURES]), trees


def rule_predictions(
    m: pd.DataFrame, cfg: Config, target: str = "y_trend"
) -> dict[str, np.ndarray]:
    """Hard predictions of the two fixed rules on the holdout, from last year's growth."""
    s, band = cfg.split, cfg.task.trend_band
    y_dev = m.loc[dev_mask(m, s, target), target].astype(int).to_numpy()
    g1_te = m.loc[holdout_mask(m, s, target), "g1"]
    fallback = int(np.bincount(y_dev).argmax())
    return {
        "persistence": persistence_pred(g1_te, band, fallback),
        "mean_reversion": mean_reversion_pred(g1_te, band, fallback),
    }


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

    probas = {
        "majority": prior_proba(y_dev, len(y_te)),
        "binned_g1": binned_proba(m.loc[dev, "g1"], y_dev, m.loc[te, "g1"], band),
    }
    scores = {name: multiclass_scores(y_te, proba=p) for name, p in probas.items()}
    for name, pred in rule_predictions(m, cfg, target).items():
        scores[name] = multiclass_scores(y_te, pred=pred)
    return scores, probas


def rule_as_proba(pred: np.ndarray) -> np.ndarray:
    """A hard rule prediction as a proper probability matrix, so it can be bootstrapped."""
    return np.eye(3)[pred] * 0.94 + 0.02


ENSEMBLE_MEMBERS = ["xgboost", "lightgbm", "catboost"]
RULES = ("persistence", "mean_reversion")
# (a, b): is a better than b? Only pairs whose two sides were run are reported.
PAIRS = [
    ("xgboost_tuned", "mean_reversion"),
    ("xgboost_tuned", "xgboost_default"),
    ("xgboost_tuned", "logreg_tuned"),
    ("lightgbm_tuned", "xgboost_tuned"),
    ("catboost_tuned", "xgboost_tuned"),
    ("ensemble", "xgboost_tuned"),
]


@dataclass
class Comparison:
    """Everything the report needs from one run, all measured on the held-out years."""

    target: str
    holdout: Holdout
    scores: pd.DataFrame  # one row per entry: metrics, 95% intervals, tree count
    probas: dict[str, np.ndarray]  # holdout probabilities for every entry that has them
    tuning: dict[str, TuneResult]
    paired: pd.DataFrame  # paired bootstrap differences between entries


def run_comparison(
    m: pd.DataFrame,
    cfg: Config,
    target: str = "y_trend",
    n_trials: int = 40,
    models: tuple[str, ...] = tuple(MODEL_NAMES),
    n_boot: int = 500,
    log: Callable[[str], None] | None = None,
) -> Comparison:
    """Baselines, default and tuned models, and a soft-voting ensemble, scored on the holdout."""
    log = log or (lambda msg: None)
    h = holdout_of(m, cfg, target)
    scores, probas = baseline_results(m, cfg, target)
    rules = rule_predictions(m, cfg, target)
    trees: dict[str, int | None] = {}
    tuning: dict[str, TuneResult] = {}

    for name in models:
        log(f"{name}: default parameters")
        probas[f"{name}_default"], trees[f"{name}_default"] = final_fit_predict(
            name, m, cfg, target
        )
        log(f"{name}: tuning {n_trials} trials over {len(cfg.split.cv_windows)} time folds")
        tuning[name] = tune(name, m, cfg, target, n_trials)
        probas[f"{name}_tuned"], trees[f"{name}_tuned"] = final_fit_predict(
            name, m, cfg, target, tuning[name].best_params
        )
    members = [f"{n}_tuned" for n in ENSEMBLE_MEMBERS if n in models]
    if len(members) >= 2:
        probas["ensemble"] = np.mean([probas[k] for k in members], axis=0)
        log(f"ensemble of {', '.join(members)}")

    for name, proba in probas.items():
        if name not in scores:
            scores[name] = multiclass_scores(h.y, proba=proba)

    order = ["majority", *RULES, "binned_g1"]
    order += [f"{name}_{kind}" for name in models for kind in ("default", "tuned")]
    order += ["ensemble"] if "ensemble" in probas else []
    table = pd.DataFrame.from_dict(scores, orient="index").loc[order]
    table.index.name = "entry"
    for entry in table.index:
        proba = probas.get(entry, rule_as_proba(rules[entry]) if entry in rules else None)
        lo, hi = bootstrap_ci(h.y, proba, macro_f1_from_proba, h.groups, n_boot, cfg.seed)
        table.loc[entry, ["f1_lo", "f1_hi"]] = lo, hi
        if entry in probas:
            lo, hi = bootstrap_ci(h.y, probas[entry], auc_from_proba, h.groups, n_boot, cfg.seed)
            table.loc[entry, ["auc_lo", "auc_hi"]] = lo, hi
        table.loc[entry, "trees"] = trees.get(entry, np.nan)

    def scorable(entry: str) -> np.ndarray:
        return probas[entry] if entry in probas else rule_as_proba(rules[entry])

    rows = []
    for a, b in PAIRS:
        if a not in table.index or b not in table.index:
            continue
        for metric_name, fn in (("macro_f1", macro_f1_from_proba), ("auc_ovr", auc_from_proba)):
            if metric_name == "auc_ovr" and (a in rules or b in rules):
                continue  # hard rules have no probabilities, so no AUC
            diff = paired_bootstrap_diff(
                h.y, scorable(a), scorable(b), fn, h.groups, n_boot, cfg.seed
            )
            rows.append({"a": a, "b": b, "metric": metric_name, **diff})
    return Comparison(target, h, table, probas, tuning, pd.DataFrame(rows))


def min_cases_sensitivity(
    panel: pd.DataFrame,
    cfg: Config,
    values: tuple[int, ...] = (20, 50, 100, 200, 500),
    target: str = "y_trend",
    n_boot: int = 300,
) -> pd.DataFrame:
    """How the model's edge over the mean-reversion rule moves with the minimum-count rule.

    Each threshold rebuilds the matrix, so the labelled rows change; at each one the default
    XGBoost and the rule are scored on the same holdout rows, with a paired bootstrap interval
    on the difference in macro-F1.
    """
    rows = []
    for min_cases in values:
        task = replace(cfg.task, min_cases=min_cases)
        cfg_v = replace(cfg, task=task)
        m = build_matrix(panel, task)
        h = holdout_of(m, cfg_v, target)
        proba, trees = final_fit_predict("xgboost", m, cfg_v, target)
        rule = rule_as_proba(rule_predictions(m, cfg_v, target)["mean_reversion"])
        diff = paired_bootstrap_diff(
            h.y, proba, rule, macro_f1_from_proba, h.groups, n_boot, cfg.seed
        )
        rows.append(
            {
                "min_cases": min_cases,
                "n_dev": int(dev_mask(m, cfg.split, target).sum()),
                "n_test": len(h.y),
                "model_f1": macro_f1_from_proba(h.y, proba),
                "rule_f1": macro_f1_from_proba(h.y, rule),
                "model_auc": auc_from_proba(h.y, proba),
                "diff": diff["diff"],
                "diff_lo": diff["lo"],
                "diff_hi": diff["hi"],
            }
        )
    return pd.DataFrame(rows)


def split_performance(
    m: pd.DataFrame,
    cfg: Config,
    target: str = "y_trend",
    name: str = "xgboost",
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """One model scored on its training, validation and test rows: the overfitting check.

    The model is fitted on the training years and early-stopped on the validation years, so all
    three rows come from the same fitted model. The headline held-out scores refit on train +
    validation first, so they are not the same numbers as this table's test row (more recent
    training data helps when the test years differ from the training years).
    """
    X, y, s = m[FEATURES], m[target], cfg.split
    tr, va, te = (f(m, s, target) for f in (train_mask, val_mask, holdout_mask))
    model = fit(name, X[tr], y[tr], X[va], y[va], params, cfg.seed)
    rows = {}
    for label, mask in (("Train", tr), ("Validation", va), ("Test", te)):
        years = m.loc[mask, "year"]
        scores = multiclass_scores(y[mask].astype(int), proba=model.predict_proba(X[mask]))
        rows[label] = {
            "years": f"{years.min()}-{years.max()}",
            "rows": int(mask.sum()),
            **scores,
        }
    return pd.DataFrame(rows).T
