import zipfile
from dataclasses import replace

import pandas as pd
import pytest

from crimebase.config import load_config
from crimebase.data import load_india_year, load_state_year, read_table

GOOD_STATE_CSV = (
    "state_code,state,year,scope,crime_head,crime_category,metric,value,disputed,source,table_id,file\n"
    "MH,Maharashtra,2019,ipc,MURDER,murder,cases_registered,120.0,False,state_table,,a.xls\n"
    "MH,Maharashtra,2020,ipc,MURDER,murder,cases_registered,130.0,False,state_table,,a.xls\n"
)


def make_cfg(tmp_path, files):
    zip_path = tmp_path / "raw.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, body in files.items():
            zf.writestr(name, body)
    cfg = load_config()
    return replace(cfg, paths=replace(cfg.paths, raw_zip=zip_path))


def test_state_year_reads_from_zip_without_extracting(tmp_path):
    cfg = make_cfg(tmp_path, {"crime_state_year.csv": GOOD_STATE_CSV})
    df = load_state_year(cfg)
    assert list(df.year) == [2019, 2020]
    assert not any(p.name.endswith(".csv") for p in tmp_path.iterdir())


def test_missing_columns_raise(tmp_path):
    cfg = make_cfg(tmp_path, {"crime_state_year.csv": "state_code,year,value\nMH,2019,1\n"})
    with pytest.raises(ValueError, match="missing columns"):
        load_state_year(cfg)


def test_non_numeric_value_raises(tmp_path):
    bad = GOOD_STATE_CSV.replace("120.0", "abc").replace("130.0", "abc")
    cfg = make_cfg(tmp_path, {"crime_state_year.csv": bad})
    with pytest.raises(ValueError, match="numeric"):
        load_state_year(cfg)


def test_missing_member_and_missing_zip(tmp_path):
    cfg = make_cfg(tmp_path, {"other.csv": "a\n1\n"})
    with pytest.raises(KeyError, match="crime_state_year.csv"):
        load_state_year(cfg)
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "nope.zip", "x.csv")


def test_parquet_member(tmp_path):
    zip_path = tmp_path / "p.zip"
    df = pd.DataFrame({"a": [1, 2]})
    src = tmp_path / "t.parquet"
    df.to_parquet(src)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(src, "t.parquet")
    assert read_table(zip_path, "t.parquet").equals(df)


def test_india_year_requires_checked_column(tmp_path):
    csv = (
        "year,scope,crime_head,crime_category,metric,value,disputed\n"
        "2019,ipc,X,other,rate,1.0,False\n"
    )
    cfg = make_cfg(tmp_path, {"crime_india_year.csv": csv})
    with pytest.raises(ValueError, match="checked"):
        load_india_year(cfg)
