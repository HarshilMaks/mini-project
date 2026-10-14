import pandas as pd

from crimebase.clean import KEY, PANEL_KEY, build_panel, dedupe_metrics, head_key
from crimebase.config import load_config


def test_head_key_merges_case_and_punctuation_variants():
    heads = pd.Series(
        [
            "MURDER",
            "Murder",
            "SC/ST (Prev of Atrocities) Act | For SCs@",
            "SC/ST (Prev. Of Atrocities) Act | For SCs",
        ]
    )
    keys = head_key(heads)
    assert keys[0] == keys[1] == "murder"
    assert keys[2] == keys[3] == "sc st prev of atrocities act | for scs"


def test_head_key_keeps_hierarchy_and_digits():
    keys = head_key(pd.Series(["Murder (Sec.302 IPC) | A) Murder with Rape/POCSO", "Murder"]))
    assert keys[0] == "murder sec 302 ipc | a murder with rape pocso"
    assert keys[0] != keys[1]


def test_head_key_collapses_whitespace():
    assert head_key(pd.Series(["  Dowry   Deaths \n"]))[0] == "dowry deaths"


def row(value, *, disputed=False, source="state_table", cat="murder", **over):
    base = dict(
        state_code="MH",
        scope="ipc",
        head_key="murder",
        year=2019,
        metric="cases_registered",
        value=value,
        disputed=disputed,
        source=source,
        crime_category=cat,
    )
    return {**base, **over}


def test_dedupe_does_not_sum_identical_duplicates():
    out = dedupe_metrics(pd.DataFrame([row(120.0), row(120.0)]))
    assert len(out) == 1 and out.value.iloc[0] == 120.0


def test_dedupe_keeps_larger_value_and_flags_dispute():
    # modelled on AP 2013 SC/ST atrocities, where a parse glitch read 1 instead of 695
    good = row(695.0, cat="crime_against_sc_st")
    glitch = row(1.0, disputed=True, cat="other")
    out = dedupe_metrics(pd.DataFrame([good, glitch]))
    assert len(out) == 1
    assert out.value.iloc[0] == 695.0
    assert out.crime_category.iloc[0] == "crime_against_sc_st"  # category follows the kept row
    assert bool(out.disputed.iloc[0]) is True


def test_dedupe_result_does_not_depend_on_row_order():
    rows = [
        row(50.0, disputed=True, source="district_total"),
        row(50.0, source="state_table"),
        row(50.0, source="district_total", cat="other"),
    ]
    a = dedupe_metrics(pd.DataFrame(rows))
    b = dedupe_metrics(pd.DataFrame(rows[::-1]))
    pd.testing.assert_frame_equal(a, b)
    assert a.source.iloc[0] == "state_table"
    assert bool(a.disputed.iloc[0]) is True


def test_dedupe_passes_unique_rows_through():
    out = dedupe_metrics(
        pd.DataFrame(
            [
                row(120.0),
                row(100.0, metric="persons_arrested"),
                row(130.0, year=2020),
            ]
        )
    )
    assert len(out) == 3
    assert not out.duplicated(KEY).any()


def test_build_panel_pivots_companions_and_drops_heads_without_main_metric():
    task = load_config().task
    df = pd.DataFrame(
        [
            row(120.0),
            row(100.0, metric="persons_arrested"),
            row(40.0, metric="victims", head_key="rape", cat="sexual_offence"),
        ]
    ).assign(crime_head=lambda d: d.head_key.str.upper())
    panel = build_panel(df, task)
    assert len(panel) == 1  # "rape" has no cases_registered row, so it is not a series
    r = panel.iloc[0]
    assert (r.cases, r.persons_arrested) == (120.0, 100.0)
    assert pd.isna(r.victims) and pd.isna(r.persons_convicted)  # unreported companions stay NaN
    assert list(panel.columns[:4]) == PANEL_KEY


def test_build_panel_merges_spelling_variants_without_double_counting():
    task = load_config().task
    df = pd.DataFrame([row(120.0), row(120.0)]).assign(crime_head=["MURDER", "Murder"])
    panel = build_panel(df, task)
    assert len(panel) == 1 and panel.cases.iloc[0] == 120.0


def test_build_panel_key_is_unique_and_sorted():
    task = load_config().task
    rows = [row(10.0 + y, year=y) for y in (2021, 2019, 2020)]
    panel = build_panel(pd.DataFrame(rows).assign(crime_head="Murder"), task)
    assert not panel.duplicated(PANEL_KEY).any()
    assert list(panel.year) == [2019, 2020, 2021]
