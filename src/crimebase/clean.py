"""Turn the long state-year table into one tidy panel row per (state, scope, head, year)."""

from __future__ import annotations

import pandas as pd


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
