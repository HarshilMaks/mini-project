"""Typed access to configs/crimebase.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "crimebase.yaml"


@dataclass(frozen=True)
class Paths:
    raw_zip: Path
    interim: Path
    processed: Path
    models: Path
    metrics: Path
    figures: Path
    report: Path


@dataclass(frozen=True)
class Tables:
    state_year: str
    india_year: str
    news: str


@dataclass(frozen=True)
class Task:
    metric: str
    companions: tuple[str, ...]
    min_cases: int
    trend_band: float
    event_band: float
    max_ratio: float
    non_crime_head_prefixes: tuple[str, ...]
    boundary_breaks: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Split:
    train_to: int
    val_years: tuple[int, ...]
    test_from: int
    cv_windows: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Config:
    paths: Paths
    tables: Tables
    task: Task
    split: Split
    seed: int


def load_config(path: str | Path | None = None) -> Config:
    """Read the YAML file and resolve every path against the repository root."""
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    paths = Paths(**{k: ROOT / v for k, v in raw["paths"].items()})
    t = raw["task"]
    task = Task(
        metric=t["metric"],
        companions=tuple(t["companions"]),
        min_cases=int(t["min_cases"]),
        trend_band=float(t["trend_band"]),
        event_band=float(t["event_band"]),
        max_ratio=float(t["max_ratio"]),
        non_crime_head_prefixes=tuple(t["non_crime_head_prefixes"]),
        boundary_breaks=tuple((b["state"], int(b["year"])) for b in t["boundary_breaks"]),
    )
    s = raw["split"]
    split = Split(
        train_to=int(s["train_to"]),
        val_years=tuple(int(y) for y in s["val_years"]),
        test_from=int(s["test_from"]),
        cv_windows=tuple((int(a), int(b)) for a, b in s["cv_windows"]),
    )
    return Config(paths, Tables(**raw["tables"]), task, split, int(raw["seed"]))
