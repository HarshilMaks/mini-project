import zipfile
from dataclasses import replace

import pandas as pd
import pytest

from conftest import make_panel
from crimebase.clean import PANEL_KEY
from crimebase.config import load_config
from crimebase.features import FEATURES, build_matrix
from crimebase.pipeline import load_matrix, run_clean, run_features

METRICS = {
    "cases": "cases_registered",
    "persons_arrested": "persons_arrested",
    "cases_chargesheeted": "cases_chargesheeted",
    "persons_convicted": "persons_convicted",
    "victims": "victims",
}


def to_long(panel: pd.DataFrame) -> pd.DataFrame:
    """Turn a panel back into the raw long table the zip holds."""
    long = panel.melt(
        id_vars=["state_code", "scope", "head_key", "year", "crime_category", "source", "disputed"],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    long["metric"] = long["metric"].map(METRICS)
    return long.rename(columns={"head_key": "crime_head"}).assign(
        state=lambda d: d.state_code, table_id=None, file="x.xlsx"
    )


@pytest.fixture
def cfg(tmp_path):
    zip_path = tmp_path / "raw.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("crime_state_year.csv", to_long(make_panel()).to_csv(index=False))
    base = load_config()
    paths = replace(
        base.paths,
        raw_zip=zip_path,
        interim=tmp_path / "interim",
        processed=tmp_path / "processed",
        metrics=tmp_path / "metrics",
        figures=tmp_path / "figures",
        models=tmp_path / "models",
        report=tmp_path / "results.md",
    )
    return replace(base, paths=paths)


def test_clean_then_features_write_the_contract_files(cfg):
    panel_file = run_clean(cfg)
    matrix_file = run_features(cfg)
    panel = pd.read_parquet(panel_file)
    matrix = load_matrix(cfg)
    assert matrix_file.name == "model_matrix.parquet"
    assert not panel.duplicated(PANEL_KEY).any()
    assert len(matrix) == len(panel) == len(make_panel())
    assert set(FEATURES) <= set(matrix.columns)
    assert matrix.eligible.sum() > 0


def test_stage_output_matches_the_in_memory_chain(cfg):
    run_clean(cfg)
    run_features(cfg)
    expected = build_matrix(make_panel(), cfg.task)
    got = load_matrix(cfg)
    pd.testing.assert_frame_equal(
        got.sort_values(["state_code", "scope", "head_key", "year"], ignore_index=True),
        expected.sort_values(["state_code", "scope", "head_key", "year"], ignore_index=True),
        check_dtype=False,
        check_categorical=False,
    )


def test_stages_are_idempotent(cfg):
    run_clean(cfg)
    first = run_features(cfg).read_bytes()
    run_clean(cfg)
    second = run_features(cfg).read_bytes()
    assert first == second


def test_features_without_panel_gives_a_helpful_error(cfg):
    with pytest.raises(FileNotFoundError, match="python -m crimebase clean"):
        run_features(cfg)
    with pytest.raises(FileNotFoundError, match="python -m crimebase features"):
        load_matrix(cfg)
