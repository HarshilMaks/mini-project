import numpy as np
import pandas as pd
import pytest


def make_panel(seed: int = 0, years=range(2005, 2021)) -> pd.DataFrame:
    """Small synthetic panel in the shape build_panel() returns.

    Random-walk counts for 3 states x 2 scopes x 3 heads, with a few years dropped from
    some series so gap handling is exercised.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for state in ("MH", "KA", "TN"):
        for scope in ("ipc", "women"):
            for head in ("murder", "theft", "rape"):
                level = rng.uniform(80, 400)
                for year in years:
                    level *= float(np.exp(rng.normal(0, 0.2)))
                    rows.append(
                        dict(
                            state_code=state,
                            scope=scope,
                            head_key=head,
                            year=year,
                            crime_category="murder" if head == "murder" else "other",
                            source="state_table",
                            disputed=False,
                            cases=float(round(level)),
                            persons_arrested=float(round(level * 0.8)),
                            cases_chargesheeted=float(round(level * 0.7)),
                            persons_convicted=float(round(level * 0.1)),
                            victims=float(round(level * 0.05)) if year % 2 == 0 else np.nan,
                        )
                    )
    panel = pd.DataFrame(rows)
    drop = (panel.state_code == "KA") & (panel.head_key == "theft") & (panel.year == 2012)
    return panel[~drop].reset_index(drop=True)


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_panel()
