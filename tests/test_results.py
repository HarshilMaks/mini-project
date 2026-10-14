from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from conftest import make_panel
from crimebase.config import load_config
from crimebase.experiment import run_comparison
from crimebase.features import build_matrix
from crimebase.results import load_comparison, save_comparison


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    base = load_config()
    out = tmp_path_factory.mktemp("metrics")
    cfg = replace(
        base,
        paths=replace(base.paths, metrics=out),
        split=replace(
            base.split,
            train_to=2013,
            val_years=(2014, 2015),
            test_from=2016,
            cv_windows=((2010, 2011), (2012, 2013), (2014, 2015)),
        ),
    )
    m = build_matrix(make_panel(), cfg.task)
    cmp = run_comparison(m, cfg, models=("xgboost", "lightgbm"), n_trials=2, n_boot=30)
    save_comparison(cmp, m, cfg, n_trials=2, n_boot=30)
    return cfg, m, cmp, load_comparison(cfg)


def test_files_are_written_under_the_metrics_directory(saved):
    cfg, *_ = saved
    names = sorted(p.name for p in cfg.paths.metrics.iterdir())
    assert names == [
        "y_trend_holdout_predictions.parquet",
        "y_trend_meta.json",
        "y_trend_paired.csv",
        "y_trend_scores.csv",
        "y_trend_trials_lightgbm.csv",
        "y_trend_trials_xgboost.csv",
        "y_trend_tuning.json",
    ]


def test_scores_round_trip(saved):
    _, _, cmp, back = saved
    pd.testing.assert_frame_equal(back.scores, cmp.scores, check_exact=False, rtol=1e-9)
    assert len(back.paired) == len(cmp.paired)


def test_tuning_json_records_defaults_versus_best(saved):
    _, _, cmp, back = saved
    for name, res in cmp.tuning.items():
        entry = back.tuning[name]
        assert entry["best_params"] == res.best_params
        assert entry["best_cv_log_loss"] <= entry["default_cv_log_loss"]
        assert entry["n_trials"] == 2 and entry["n_folds"] == 3


def test_predictions_keep_ids_and_probabilities_for_every_entry(saved):
    _, m, cmp, back = saved
    preds = back.predictions
    assert len(preds) == len(cmp.holdout.y)
    assert {"state_code", "scope", "head_key", "year", "y_true", "row"} <= set(preds.columns)
    assert (preds.year >= back.meta["test_from"]).all()
    p = back.proba("xgboost_tuned")
    assert list(p.columns) == ["down", "stable", "up"]
    assert np.allclose(p.to_numpy(), cmp.probas["xgboost_tuned"])
    assert np.allclose(p.sum(axis=1), 1)
    # rows can be traced back to the model matrix
    assert (
        m.loc[preds.row, "state_code"].astype(str).to_numpy()
        == preds.state_code.astype(str).to_numpy()
    ).all()


def test_meta_records_the_setup(saved):
    cfg, _, cmp, back = saved
    assert back.meta["holdout_rows"] == len(cmp.holdout.y)
    assert back.meta["min_cases"] == cfg.task.min_cases and back.meta["n_boot"] == 30


def test_loading_before_training_names_the_command(tmp_path):
    cfg = load_config()
    cfg = replace(cfg, paths=replace(cfg.paths, metrics=tmp_path))
    with pytest.raises(FileNotFoundError, match="python -m crimebase train"):
        load_comparison(cfg)
