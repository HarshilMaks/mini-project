import pandas as pd

from crimebase.clean import KEY, dedupe_metrics, head_key


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
