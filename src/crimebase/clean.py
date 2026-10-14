"""Turn the long state-year table into one tidy panel row per (state, scope, head, year)."""

from __future__ import annotations

import pandas as pd

from crimebase.config import Task

PANEL_KEY = ["state_code", "scope", "head_key", "year"]
KEY = [*PANEL_KEY, "metric"]


def head_key(heads: pd.Series) -> pd.Series:
    """Normalise a crime-head label so spelling variants of one head share a key.

    NCRB prints the same head in different case and punctuation across editions
    ("MURDER", "Murder", "SC/ST (Prev. of Atrocities) Act" vs "...Prev of..."). The
    ' | ' hierarchy separator is kept so a sub-head never merges with its parent.
    """
    return (
        heads.str.lower()
        .str.replace(r"[^a-z0-9| ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def dedupe_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Make `KEY` unique by keeping one row per group. Never sums.

    Spelling variants of one head usually carry the same number from different files
    (97.5% of duplicate groups). Where they differ (141 groups) the smaller value is a
    parse glitch (for example 1 against 695), and the larger row is never flagged
    disputed. So the kept row is the largest value; ties prefer an undisputed row, then
    the `state_table` source, then the alphabetically first category, which makes the
    result independent of input order. `disputed` is True if any variant was disputed.
    """
    order = df.sort_values(
        ["value", "disputed", "source", "crime_category"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    kept = order.groupby(KEY, sort=False).tail(1).set_index(KEY)
    kept["disputed"] = df.groupby(KEY)["disputed"].any()
    return kept.reset_index().sort_values(KEY, ignore_index=True)


def build_panel(df: pd.DataFrame, task: Task) -> pd.DataFrame:
    """One row per (state, scope, head, year) that has the forecast metric.

    `cases` is the forecast metric; the companion metrics from the same year become
    columns (NaN where the table did not report them). `crime_category`, `source` and
    `disputed` describe the forecast-metric row. Heads are never summed across scopes:
    the scopes overlap, so each (scope, head) stays its own series.
    """
    d = dedupe_metrics(df.assign(head_key=head_key(df["crime_head"])))
    wide = d.set_index(KEY)["value"].unstack("metric")
    meta = d[d["metric"] == task.metric].set_index(PANEL_KEY)[
        ["crime_category", "source", "disputed"]
    ]
    values = wide.reindex(columns=[task.metric, *task.companions]).rename(
        columns={task.metric: "cases"}
    )
    panel = meta.join(values)
    panel.columns.name = None
    return panel.reset_index().sort_values(PANEL_KEY, ignore_index=True)
