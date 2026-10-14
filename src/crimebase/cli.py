"""Command line: `python -m crimebase <stage>`."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from crimebase.config import Config, load_config
from crimebase.experiment import run_comparison
from crimebase.models import MODEL_NAMES
from crimebase.pipeline import load_matrix, run_clean, run_features
from crimebase.results import save_comparison


def _train(cfg: Config, args: argparse.Namespace) -> Path:
    """data/processed/model_matrix.parquet -> reports/metrics/*"""
    m = load_matrix(cfg)
    cmp = run_comparison(m, cfg, args.target, args.trials, tuple(args.models), args.boot, log=print)
    return save_comparison(cmp, m, cfg, args.trials, args.boot)


STAGES: dict[str, Callable[[Config, argparse.Namespace], Path]] = {
    "clean": lambda cfg, args: run_clean(cfg),
    "features": lambda cfg, args: run_features(cfg),
    "train": _train,
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
    args = parser.parse_args(argv)
    out = STAGES[args.stage](load_config(args.config), args)
    print(f"{args.stage}: wrote {out}")
    return 0
