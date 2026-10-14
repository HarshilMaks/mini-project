import numpy as np
import pandas as pd
import pytest

from crimebase.features import SERIES, add_context, add_history


def one_series(years, cases, **over):
    base = dict(state_code="MH", scope="ipc", head_key="murder")
    return pd.DataFrame({**base, "year": years, "cases": cases, **over})


def test_growth_uses_exact_previous_year_and_gaps_give_nan():
    # 2017 is missing, so 2018 has no previous year and must not borrow 2016
    df = add_history(one_series([2015, 2016, 2018], [100.0, 200.0, 400.0]))
    g1 = dict(zip(df.year, df.g1, strict=True))
    assert g1[2015] != g1[2015]  # NaN: first year
    assert g1[2016] == pytest.approx(np.log1p(200) - np.log1p(100))
    assert np.isnan(g1[2018])


def test_rolling_stats_use_only_observed_years():
    df = add_history(one_series([2015, 2016, 2017, 2018], [10.0, 20.0, 40.0, 80.0]))
    last = df[df.year == 2018].iloc[0]
    lv = np.log1p([80.0, 40.0, 20.0, 10.0])
    assert last.mean4 == pytest.approx(lv.mean())
    assert last.std4 == pytest.approx(lv.std(ddof=1))
    first = df[df.year == 2015].iloc[0]
    assert first.mean4 == pytest.approx(np.log1p(10.0)) and np.isnan(first.std4)


def test_expanding_mean_and_history_count():
    df = add_history(one_series([2015, 2016, 2017], [10.0, 30.0, 20.0]))
    lv = np.log1p([10.0, 30.0, 20.0])
    assert list(df.n_hist) == [0, 1, 2]
    assert df.dev_exp.iloc[2] == pytest.approx(lv[2] - lv.mean())


def test_history_features_do_not_depend_on_the_future(panel):
    cutoff = 2014
    base = add_history(panel)
    shifted = panel.copy()
    shifted.loc[shifted.year > cutoff, "cases"] *= 7  # rewrite everything after the cutoff
    other = add_history(shifted)
    cols = ["lv", "g1", "g2", "g3", "mean4", "std4", "dev4", "z4", "dev_exp", "n_hist"]
    keep = base.year <= cutoff
    pd.testing.assert_frame_equal(
        base.loc[keep, [*SERIES, "year", *cols]].reset_index(drop=True),
        other.loc[keep, [*SERIES, "year", *cols]].reset_index(drop=True),
    )


def head_year_frame(g1, cases, states=("MH", "KA", "TN"), head="murder", year=2018):
    return pd.DataFrame(
        {
            "state_code": list(states),
            "scope": "ipc",
            "head_key": head,
            "year": year,
            "cases": cases,
            "lv": np.log1p(cases),
            "g1": g1,
        }
    )


def test_context_compares_a_state_with_the_same_head_elsewhere():
    out = add_context(head_year_frame(g1=[0.5, 0.1, -0.3], cases=[100.0, 300.0, 600.0]))
    assert out.g1_med.tolist() == [0.1, 0.1, 0.1]
    assert out.g1_rel.tolist() == pytest.approx([0.4, 0.0, -0.4])
    assert out.n_states.tolist() == [3, 3, 3]
    assert out.share.tolist() == pytest.approx([0.1, 0.3, 0.6])
    assert out.share.sum() == pytest.approx(1.0)


def test_context_shares_are_nan_when_the_head_total_is_zero():
    out = add_context(head_year_frame(g1=[0.0, 0.0, 0.0], cases=[0.0, 0.0, 0.0]))
    assert out.share.isna().all()


def test_state_level_context_averages_over_heads():
    df = pd.concat(
        [
            head_year_frame(g1=[0.2, 0.0, 0.0], cases=[10.0, 10.0, 10.0], head="a"),
            head_year_frame(g1=[0.4, 0.0, 0.0], cases=[10.0, 10.0, 10.0], head="b"),
        ],
        ignore_index=True,
    )
    mh = add_context(df).query("state_code == 'MH'")
    assert mh.st_g1_mean.tolist() == pytest.approx([0.3, 0.3])
    assert mh.st_nheads.tolist() == [2, 2]


def test_context_does_not_mix_years():
    df = pd.concat(
        [
            head_year_frame(g1=[1.0, 1.0, 1.0], cases=[10.0, 10.0, 10.0], year=2017),
            head_year_frame(g1=[-1.0, -1.0, -1.0], cases=[10.0, 10.0, 10.0], year=2018),
        ],
        ignore_index=True,
    )
    out = add_context(df)
    assert set(out.loc[out.year == 2017, "g1_med"]) == {1.0}
    assert set(out.loc[out.year == 2018, "g1_med"]) == {-1.0}


def test_context_features_do_not_depend_on_the_future(panel):
    cutoff = 2014
    base = add_context(add_history(panel))
    shifted = panel.copy()
    shifted.loc[shifted.year > cutoff, "cases"] *= 7
    other = add_context(add_history(shifted))
    cols = ["g1_med", "g1_rel", "lv_rel", "n_states", "share", "st_g1_mean", "st_nheads"]
    keep = base.year <= cutoff
    pd.testing.assert_frame_equal(
        base.loc[keep, [*SERIES, "year", *cols]].reset_index(drop=True),
        other.loc[keep, [*SERIES, "year", *cols]].reset_index(drop=True),
    )
