"""Command line: `python -m crimebase <stage>`."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from crimebase.config import Config, load_config
from crimebase.experiment import min_cases_sensitivity, run_comparison
from crimebase.models import MODEL_NAMES
from crimebase.pipeline import load_matrix, panel_path, run_clean, run_features
from crimebase.report import build_report
from crimebase.results import save_comparison


def _train(cfg: Config, args: argparse.Namespace) -> Path:
    """data/processed/model_matrix.parquet -> reports/metrics/*"""
    m = load_matrix(cfg)
    cmp = run_comparison(m, cfg, args.target, args.trials, tuple(args.models), args.boot, log=print)
    return save_comparison(cmp, m, cfg, args.trials, args.boot)


def _sensitivity(cfg: Config, args: argparse.Namespace) -> Path:
    """data/interim/panel.parquet -> reports/metrics/<target>_sensitivity.csv"""
    if not panel_path(cfg).exists():
        raise FileNotFoundError(
            f"{panel_path(cfg)} not found; run `python -m crimebase clean` first"
        )
    panel = pd.read_parquet(panel_path(cfg))
    table = min_cases_sensitivity(panel, cfg, tuple(args.thresholds), args.target, args.boot)
    out = cfg.paths.metrics / f"{args.target}_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return out


def _report(cfg: Config, args: argparse.Namespace) -> Path:
    """reports/metrics/* -> reports/results.md, reports/figures/*, models/*"""
    return build_report(cfg, args.target)["report"]


STAGES: dict[str, Callable[[Config, argparse.Namespace], Path]] = {
    "clean": lambda cfg, args: run_clean(cfg),
    "features": lambda cfg, args: run_features(cfg),
    "train": _train,
    "sensitivity": _sensitivity,
    "report": _report,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crimebase", description=__doc__)
    parser.add_argument("--config", help="path to a config YAML (default: configs/crimebase.yaml)")
    parser.add_argument("stage", choices=sorted(STAGES), help="pipeline stage to run")
    train = parser.add_argument_group("train options")
    train.add_argument("--target", default="y_trend", choices=["y_trend"])
    train.add_argument("--trials", type=int, default=40, help="Optuna trials per model")
    train.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    train.add_argument("--boot", type=int, default=500, help="bootstrap resamples for intervals")
    train.add_argument(
        "--thresholds", nargs="+", type=int, default=[20, 50, 100, 200, 500],
        help="min_cases values for the sensitivity stage",
    )  # fmt: skip
    args = parser.parse_args(argv)
    out = STAGES[args.stage](load_config(args.config), args)
    print(f"{args.stage}: wrote {out}")
    return 0
