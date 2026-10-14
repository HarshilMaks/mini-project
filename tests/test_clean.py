import pandas as pd

from crimebase.clean import head_key


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
