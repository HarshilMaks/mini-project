"""Write a Comparison to reports/metrics/ and read it back. The train stage is the only writer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from crimebase.config import Config
from crimebase.evaluate import CLASS_NAMES
from crimebase.experiment import Comparison

ID_COLUMNS = ["state_code", "scope", "head_key", "year"]


def _path(cfg: Config, target: str, name: str) -> Path:
    return cfg.paths.metrics / f"{target}_{name}"


def save_comparison(
    cmp: Comparison, m: pd.DataFrame, cfg: Config, n_trials: int, n_boot: int
) -> Path:
    """Write scores, paired tests, tuning history and holdout predictions; return the scores file."""
    out = cfg.paths.metrics
    out.mkdir(parents=True, exist_ok=True)
    t = cmp.target
    cmp.scores.to_csv(_path(cfg, t, "scores.csv"))
    cmp.paired.to_csv(_path(cfg, t, "paired.csv"), index=False)

    tuning = {}
    for name, res in cmp.tuning.items():
        res.trials.to_csv(_path(cfg, t, f"trials_{name}.csv"), index=False)
        tuning[name] = {
            "best_params": res.best_params,
            "best_cv_log_loss": res.best_value,
            "default_cv_log_loss": res.default_value,
            "n_trials": len(res.trials),
            "n_folds": res.n_folds,
        }
    _path(cfg, t, "tuning.json").write_text(json.dumps(tuning, indent=2) + "\n")

    preds = m.loc[cmp.holdout.index, ID_COLUMNS].reset_index(names="row")
    preds["y_true"] = cmp.holdout.y
    for entry, proba in cmp.probas.items():
        for k, label in enumerate(CLASS_NAMES):
            preds[f"p_{entry}_{label}"] = proba[:, k]
    preds.to_parquet(_path(cfg, t, "holdout_predictions.parquet"), index=False)

    meta = {
        "target": t,
        "n_trials": n_trials,
        "n_boot": n_boot,
        "seed": cfg.seed,
        "min_cases": cfg.task.min_cases,
        "trend_band": cfg.task.trend_band,
        "train_to": cfg.split.train_to,
        "val_years": list(cfg.split.val_years),
        "test_from": cfg.split.test_from,
        "holdout_rows": len(cmp.holdout.y),
        "holdout_clusters": len(set(cmp.holdout.groups)),
    }
    _path(cfg, t, "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return _path(cfg, t, "scores.csv")


@dataclass
class SavedComparison:
    scores: pd.DataFrame
    paired: pd.DataFrame
    tuning: dict[str, Any]
    predictions: pd.DataFrame
    meta: dict[str, Any]

    def proba(self, entry: str) -> pd.DataFrame:
        """Holdout probabilities of one entry as columns down / stable / up."""
        cols = {f"p_{entry}_{label}": label for label in CLASS_NAMES}
        return self.predictions[list(cols)].rename(columns=cols)


def load_comparison(cfg: Config, target: str = "y_trend") -> SavedComparison:
    if not _path(cfg, target, "scores.csv").exists():
        raise FileNotFoundError(
            f"no saved results for {target} in {cfg.paths.metrics}; "
            "run `python -m crimebase train` first"
        )
    return SavedComparison(
        scores=pd.read_csv(_path(cfg, target, "scores.csv"), index_col="entry"),
        paired=pd.read_csv(_path(cfg, target, "paired.csv")),
        tuning=json.loads(_path(cfg, target, "tuning.json").read_text()),
        predictions=pd.read_parquet(_path(cfg, target, "holdout_predictions.parquet")),
        meta=json.loads(_path(cfg, target, "meta.json").read_text()),
    )
