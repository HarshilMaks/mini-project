"""Time-ordered splits. Rows are split by feature year t, never at random."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crimebase.config import Split


def labelled(m: pd.DataFrame, target: str) -> pd.Series:
    """Rows that may be used for `target`: eligible and carrying a label."""
    return m["eligible"] & m[target].notna()


def train_mask(m: pd.DataFrame, split: Split, target: str) -> pd.Series:
    return labelled(m, target) & (m["year"] <= split.train_to)


def val_mask(m: pd.DataFrame, split: Split, target: str) -> pd.Series:
    return labelled(m, target) & m["year"].isin(split.val_years)


def dev_mask(m: pd.DataFrame, split: Split, target: str) -> pd.Series:
    """Everything before the test years: what the final model is fitted on."""
    return labelled(m, target) & (m["year"] < split.test_from)


def holdout_mask(m: pd.DataFrame, split: Split, target: str) -> pd.Series:
    return labelled(m, target) & (m["year"] >= split.test_from)


def cv_folds(m: pd.DataFrame, split: Split, target: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds as (train positions, validation positions).

    Fold k validates on the years in `cv_windows[k]` and trains on every labelled row from
    earlier years. A training label for year a-1 looks at year a, which is already known
    when the validation features for year a are built, so no purge gap is needed. Windows
    reaching into the test years are refused, and empty folds are skipped.
    """
    ok = labelled(m, target)
    folds = []
    for first, last in split.cv_windows:
        if last >= split.test_from:
            raise ValueError(f"cv window {(first, last)} reaches the test years")
        train = np.flatnonzero(ok & (m["year"] < first))
        val = np.flatnonzero(ok & m["year"].between(first, last))
        if len(train) and len(val):
            folds.append((train, val))
    return folds
