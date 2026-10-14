"""Features (years <= t only) and labels (year t+1) for the year-ahead trend task."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crimebase.config import Task

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


def add_ratios_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Same-year justice-pipeline ratios and coarse head-label flags.

    Companion metrics are often unreported (NaN), and a NaN stays NaN so tree models can
    use the missingness. The +1 keeps a zero denominator finite.
    """
    out = df.copy()
    out["arr_ratio"] = out["persons_arrested"] / (out["cases"] + 1)
    out["cs_ratio"] = out["cases_chargesheeted"] / (out["cases"] + 1)
    out["conv_ratio"] = out["persons_convicted"] / (out["persons_arrested"] + 1)
    out["vic_ratio"] = out["victims"] / (out["cases"] + 1)
    out["is_total"] = out["head_key"].str.contains("total").astype("int8")
    out["is_other"] = out["head_key"].str.contains(r"\bother").astype("int8")
    out["is_child"] = out["head_key"].str.contains("child|minor|girl").astype("int8")
    return out


REASONS = [
    "no_next_year",
    "non_crime_head",
    "below_min_cases",
    "boundary_break",
    "source_switch",
    "definition_break",
]


def add_labels(df: pd.DataFrame, task: Task) -> pd.DataFrame:
    """Attach next-year growth, an eligibility reason and the three labels.

    Nothing is dropped: a row that cannot be labelled keeps `eligible=False` and the first
    failing reason from `REASONS` (checked in that order), and its labels are NA.
    `y_trend` is 0 down / 1 stable / 2 up around +-`trend_band`; `y_drop` and `y_surge`
    are the binary views at +-`event_band`.
    """
    out = df.copy()
    out["next_cases"] = value_at(out, "cases", 1)
    next_source = value_at(out, "source", 1)
    safe_cases = out["cases"].replace(0, np.nan)
    ratio = out["next_cases"] / safe_cases
    # (next - this) / this, not ratio - 1: 110/100 - 1 is 0.10000000000000009 and would
    # tip a case that is exactly +10% over the strict band edge
    out["growth"] = (out["next_cases"] - out["cases"]) / safe_cases

    breaks = {f"{state}|{year}" for state, year in task.boundary_breaks}
    state_year = out["state_code"] + "|" + out["year"].astype(str)
    failing = [
        out["next_cases"].isna(),
        out["head_key"].str.startswith(tuple(task.non_crime_head_prefixes)),
        out["cases"] < task.min_cases,
        state_year.isin(breaks),
        out["source"].to_numpy() != next_source,
        (ratio > task.max_ratio) | (ratio < 1 / task.max_ratio),
    ]
    out["reason"] = np.select(failing, REASONS, default="ok")
    out["eligible"] = out["reason"] == "ok"

    growth = out["growth"].where(out["eligible"])
    trend = np.where(growth > task.trend_band, 2, np.where(growth < -task.trend_band, 0, 1))
    out["y_trend"] = pd.Series(trend, index=out.index).astype("Int8").where(out["eligible"])
    out["y_drop"] = (growth < -task.event_band).astype("Int8").where(out["eligible"])
    out["y_surge"] = (growth > task.event_band).astype("Int8").where(out["eligible"])
    return out


NUMERIC_FEATURES = [
    "lv", "g1", "g2", "g3", "mean4", "std4", "dev4", "z4", "dev_exp", "n_hist",
    "g1_med", "g1_rel", "lv_rel", "n_states", "share", "st_g1_mean", "st_nheads",
    "arr_ratio", "cs_ratio", "conv_ratio", "vic_ratio",
    "is_total", "is_other", "is_child", "disputed",
]  # fmt: skip
CATEGORICAL_FEATURES = ["scope", "crime_category", "state_code", "source"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LABELS = ["y_trend", "y_drop", "y_surge"]
ID_COLUMNS = [*SERIES, "year"]


def build_matrix(panel: pd.DataFrame, task: Task) -> pd.DataFrame:
    """Chain every step into the model matrix: ids, features, labels and eligibility."""
    out = add_labels(add_ratios_and_flags(add_context(add_history(panel))), task)
    out["disputed"] = out["disputed"].astype("int8")
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("category")
    # state_code and scope are both ids and categorical features; list each column once
    keep = [*ID_COLUMNS, *FEATURES, "cases", "next_cases", "growth", *LABELS, "reason", "eligible"]
    return out[list(dict.fromkeys(keep))]
