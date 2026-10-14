"""Read tables straight from the raw zip. The zip is never extracted or modified."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from crimebase.config import Config

STATE_YEAR_COLUMNS = {
    "state_code",
    "state",
    "year",
    "scope",
    "crime_head",
    "crime_category",
    "metric",
    "value",
    "disputed",
    "source",
}
INDIA_YEAR_COLUMNS = {
    "year",
    "scope",
    "crime_head",
    "crime_category",
    "metric",
    "value",
    "disputed",
    "checked",
}


def read_table(zip_path: Path, name: str) -> pd.DataFrame:
    """Load one CSV or Parquet member of the zip."""
    if not zip_path.exists():
        raise FileNotFoundError(f"raw dataset not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        if name not in zf.namelist():
            raise KeyError(f"{name!r} is not in {zip_path.name}; found {sorted(zf.namelist())}")
        if name.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(zf.read(name)))
        with zf.open(name) as fh:
            return pd.read_csv(fh)


def _require(df: pd.DataFrame, columns: set[str], name: str) -> pd.DataFrame:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns {sorted(missing)}")
    if not pd.api.types.is_numeric_dtype(df["value"]):
        raise ValueError(f"{name}: 'value' must be numeric, got {df['value'].dtype}")
    return df


def load_state_year(cfg: Config) -> pd.DataFrame:
    df = read_table(cfg.paths.raw_zip, cfg.tables.state_year)
    return _require(df, STATE_YEAR_COLUMNS, cfg.tables.state_year)


def load_india_year(cfg: Config) -> pd.DataFrame:
    df = read_table(cfg.paths.raw_zip, cfg.tables.india_year)
    return _require(df, INDIA_YEAR_COLUMNS, cfg.tables.india_year)


def load_news(cfg: Config) -> pd.DataFrame:
    return read_table(cfg.paths.raw_zip, cfg.tables.news)
