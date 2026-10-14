from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from crimebase.config import load_config
from crimebase.features import (
    SERIES,
    add_context,
    add_history,
    add_labels,
    add_ratios_and_flags,
)


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


def test_ratios_use_plus_one_and_keep_missing_as_nan():
    df = pd.DataFrame(
        {
            "head_key": ["murder", "murder"],
            "cases": [99.0, 0.0],
            "persons_arrested": [198.0, 0.0],
            "cases_chargesheeted": [49.0, np.nan],
            "persons_convicted": [19.0, np.nan],
            "victims": [np.nan, 5.0],
        }
    )
    out = add_ratios_and_flags(df)
    assert out.arr_ratio.tolist() == [pytest.approx(1.98), 0.0]
    assert out.cs_ratio[0] == pytest.approx(0.49) and np.isnan(out.cs_ratio[1])
    assert out.conv_ratio[0] == pytest.approx(0.0955, abs=1e-3) and np.isnan(out.conv_ratio[1])
    assert np.isnan(out.vic_ratio[0]) and out.vic_ratio[1] == pytest.approx(5.0)
    assert np.isfinite(out[["arr_ratio"]]).all().all()


def test_head_flags():
    heads = [
        "total cognizable ipc crimes",
        "other ipc crimes",
        "mother in law cruelty",  # contains "other" inside a word, so not an "other" head
        "kidnapping of children",
        "murder",
    ]
    cols = ["cases", "persons_arrested", "cases_chargesheeted", "persons_convicted", "victims"]
    df = pd.DataFrame({"head_key": heads, **dict.fromkeys(cols, 1.0)})
    out = add_ratios_and_flags(df)
    assert out.is_total.tolist() == [1, 0, 0, 0, 0]
    assert out.is_other.tolist() == [0, 1, 0, 0, 0]
    assert out.is_child.tolist() == [0, 0, 0, 1, 0]


def labelled(rows, **task_over):
    """rows: (head, cases_t, cases_t1[, state, source_t, source_t1]); one 2018->2019 pair each."""
    task = replace(load_config().task, min_cases=50, **task_over)
    frames = []
    for i, r in enumerate(rows):
        head, c0, c1 = r[:3]
        state = r[3] if len(r) > 3 else "MH"
        s0, s1 = (r[4], r[5]) if len(r) > 4 else ("state_table", "state_table")
        frames.append(
            pd.DataFrame(
                {
                    "state_code": state,
                    "scope": "ipc",
                    "head_key": head + str(i) if head == "x" else head,
                    "year": [2018, 2019],
                    "cases": [c0, c1],
                    "source": [s0, s1],
                }
            )
        )
    out = add_labels(pd.concat(frames, ignore_index=True), task)
    return out[out.year == 2018].reset_index(drop=True)


def test_each_ineligibility_reason_triggers():
    out = labelled(
        [
            ("murder", 100.0, 110.0),  # ok
            ("oral complaints | total oral", 100.0, 110.0),  # not a crime head
            ("theft", 10.0, 12.0),  # below min_cases
            ("rape", 100.0, 110.0, "AP"),  # ok here; boundary only applies for year 2013
            ("arson", 100.0, 110.0, "MH", "state_table", "district_total"),  # source switch
            ("riots", 100.0, 2000.0),  # 20x jump is a definition break
            ("fraud", 100.0, 0.0),  # fell to zero is a definition break too
        ]
    )
    assert out.reason.tolist() == [
        "ok",
        "non_crime_head",
        "below_min_cases",
        "ok",
        "source_switch",
        "definition_break",
        "definition_break",
    ]


def test_last_year_has_no_next_year_and_gaps_are_not_bridged():
    df = pd.DataFrame(
        {
            "state_code": "MH",
            "scope": "ipc",
            "head_key": "murder",
            "year": [2015, 2016, 2018],  # 2017 missing
            "cases": [100.0, 100.0, 100.0],
            "source": "state_table",
        }
    )
    out = add_labels(df, load_config().task)
    assert out.reason.tolist() == ["ok", "no_next_year", "no_next_year"]


def test_boundary_break_flags_only_the_configured_state_year():
    rows = [("murder", 100.0, 50.0, "AP"), ("murder", 100.0, 50.0, "KA")]
    task = load_config().task
    frames = [
        pd.DataFrame(
            {
                "state_code": s,
                "scope": "ipc",
                "head_key": "murder",
                "year": [2013, 2014],
                "cases": [c0, c1],
                "source": "state_table",
            }
        )
        for _, c0, c1, s in rows
    ]
    out = add_labels(pd.concat(frames, ignore_index=True), task)
    out = out[out.year == 2013].set_index("state_code")
    assert out.loc["AP", "reason"] == "boundary_break"
    assert out.loc["KA", "reason"] == "ok"


def test_first_failing_reason_wins():
    # below min_cases AND a 20x jump: min_cases is checked first
    out = labelled([("theft", 10.0, 500.0)])
    assert out.reason.iloc[0] == "below_min_cases"


def test_trend_labels_and_band_edges():
    out = labelled(
        [
            ("a", 100.0, 120.0),  # +20% up
            ("b", 100.0, 80.0),  # -20% down
            ("c", 100.0, 105.0),  # +5% stable
            ("d", 100.0, 110.0),  # exactly +10% is still stable (strict band)
            ("e", 100.0, 90.0),  # exactly -10% is still stable
        ]
    )
    assert out.y_trend.tolist() == [2, 0, 1, 1, 1]


def test_binary_views_use_the_wider_event_band():
    out = labelled([("a", 100.0, 130.0), ("b", 100.0, 70.0), ("c", 100.0, 110.0)])
    assert out.y_surge.tolist() == [1, 0, 0]
    assert out.y_drop.tolist() == [0, 1, 0]


def test_ineligible_rows_have_na_labels():
    out = labelled([("theft", 10.0, 12.0)])
    assert out.y_trend.isna().all() and out.y_drop.isna().all() and out.y_surge.isna().all()
    assert not out.eligible.iloc[0]


def test_growth_is_next_over_this_minus_one():
    out = labelled([("a", 200.0, 250.0)])
    assert out.growth.iloc[0] == pytest.approx(0.25)
    assert out.next_cases.iloc[0] == 250.0
