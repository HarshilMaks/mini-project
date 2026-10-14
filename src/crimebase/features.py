"""Features (years <= t only) and labels (year t+1) for the year-ahead trend task."""

from __future__ import annotations

import numpy as np
import pandas as pd

SERIES = ["state_code", "scope", "head_key"]


def value_at(panel: pd.DataFrame, col: str, delta: int) -> np.ndarray:
    """`col` of the same series `delta` years from each row's year; NaN if that year is absent.

    Joining on the exact year (not on row position) means a gap in a series gives NaN
    instead of being read as "the previous year".
    """
    right = panel[[*SERIES, "year", col]].copy()
    right["year"] = right["year"] - delta
    return panel[[*SERIES, "year"]].merge(right, on=[*SERIES, "year"], how="left")[col].to_numpy()


def add_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Level, growth and rolling statistics of each series up to and including year t."""
    out = panel.sort_values([*SERIES, "year"], ignore_index=True)
    out["lv"] = np.log1p(out["cases"])
    lag = {k: value_at(out, "lv", -k) for k in (1, 2, 3)}
    out["g1"] = out["lv"] - lag[1]
    out["g2"] = lag[1] - lag[2]
    out["g3"] = lag[2] - lag[3]

    window = pd.DataFrame({"t": out["lv"], "l1": lag[1], "l2": lag[2], "l3": lag[3]})
    out["mean4"] = window.mean(axis=1)
    out["std4"] = window.std(axis=1)  # NaN with fewer than two observed years
    out["dev4"] = out["lv"] - out["mean4"]
    out["z4"] = out["dev4"] / (out["std4"] + 0.05)

    by_series = out.groupby(SERIES, sort=False)["lv"]
    expanding_mean = by_series.cumsum() / (by_series.cumcount() + 1)
    out["dev_exp"] = out["lv"] - expanding_mean
    out["n_hist"] = by_series.cumcount()
    return out
