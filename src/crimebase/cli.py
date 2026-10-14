"""Command line: `python -m crimebase <stage>`."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from crimebase.config import Config, load_config
from crimebase.pipeline import run_clean, run_features

STAGES: dict[str, Callable[[Config], Path]] = {
    "clean": run_clean,
    "features": run_features,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crimebase", description=__doc__)
    parser.add_argument("--config", help="path to a config YAML (default: configs/crimebase.yaml)")
    parser.add_argument("stage", choices=sorted(STAGES), help="pipeline stage to run")
    args = parser.parse_args(argv)
    out = STAGES[args.stage](load_config(args.config))
    print(f"{args.stage}: wrote {out}")
    return 0
