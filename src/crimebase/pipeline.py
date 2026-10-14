"""Stages that read one contract file and write the next. Each output has one writer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crimebase.clean import build_panel
from crimebase.config import Config
from crimebase.data import load_state_year
from crimebase.features import build_matrix

PANEL_FILE = "panel.parquet"
MATRIX_FILE = "model_matrix.parquet"


def panel_path(cfg: Config) -> Path:
    return cfg.paths.interim / PANEL_FILE


def matrix_path(cfg: Config) -> Path:
    return cfg.paths.processed / MATRIX_FILE


def run_clean(cfg: Config) -> Path:
    """raw zip -> data/interim/panel.parquet"""
    panel = build_panel(load_state_year(cfg), cfg.task)
    out = panel_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    return out


def run_features(cfg: Config) -> Path:
    """data/interim/panel.parquet -> data/processed/model_matrix.parquet"""
    src = panel_path(cfg)
    if not src.exists():
        raise FileNotFoundError(f"{src} not found; run `python -m crimebase clean` first")
    matrix = build_matrix(pd.read_parquet(src), cfg.task)
    out = matrix_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(out, index=False)
    return out


def load_matrix(cfg: Config) -> pd.DataFrame:
    """Read the model matrix written by `run_features`."""
    path = matrix_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m crimebase features` first")
    return pd.read_parquet(path)
