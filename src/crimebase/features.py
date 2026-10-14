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


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional context at year t.

    Compares a state with the same head in other states, and with its own other heads.
    Every statistic groups rows of one year only, so nothing from t+1 leaks in.
    """
    out = df.copy()
    head_year = out.groupby(["scope", "head_key", "year"])
    out["g1_med"] = head_year["g1"].transform("median")
    out["g1_rel"] = out["g1"] - out["g1_med"]
    out["lv_rel"] = out["lv"] - head_year["lv"].transform("median")
    out["n_states"] = head_year["state_code"].transform("nunique")
    out["share"] = out["cases"] / head_year["cases"].transform("sum").replace(0, np.nan)

    state_year = out.groupby(["state_code", "scope", "year"])
    out["st_g1_mean"] = state_year["g1"].transform("mean")
    out["st_nheads"] = state_year["head_key"].transform("nunique")
    return out
