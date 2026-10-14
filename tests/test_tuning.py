from dataclasses import replace

import numpy as np
import optuna
import pytest

from crimebase.config import load_config
from crimebase.models import DEFAULTS
from crimebase.tuning import default_trial, search_space, tune
from test_splits import matrix, split  # noqa: F401  (fixtures: synthetic matrix + shifted windows)


@pytest.fixture
def cfg(split):  # noqa: F811
    return replace(load_config(), split=split)


@pytest.mark.parametrize("name", ["xgboost", "lightgbm", "catboost", "logreg"])
def test_default_trial_lies_inside_the_search_space(name):
    study = optuna.create_study()
    study.enqueue_trial(default_trial(name))

    def objective(trial):
        search_space(name, trial)
        return 0.0

    study.optimize(objective, n_trials=1)
    trial = study.trials[0]
    assert set(trial.params) == set(default_trial(name))
    for key, value in default_trial(name).items():
        dist = trial.distributions[key]
        assert dist.low <= value <= dist.high, (
            f"{name}.{key}={value} outside [{dist.low}, {dist.high}]"
        )
        assert trial.params[key] == pytest.approx(value)


def test_tuning_scores_defaults_as_trial_zero_and_never_does_worse(matrix, cfg):  # noqa: F811
    res = tune("xgboost", matrix, cfg, n_trials=4)
    assert res.n_folds == 3
    assert np.isfinite(res.best_value) and np.isfinite(res.default_value)
    assert res.best_value <= res.default_value
    assert set(res.best_params) == set(default_trial("xgboost"))
    assert len(res.trials) == 4
    assert res.trials.loc[0, "learning_rate"] == DEFAULTS["xgboost"]["learning_rate"]


def test_tuning_is_reproducible_for_a_seed(matrix, cfg):  # noqa: F811
    a = tune("lightgbm", matrix, cfg, n_trials=3, seed=5)
    b = tune("lightgbm", matrix, cfg, n_trials=3, seed=5)
    assert a.best_params == b.best_params
    assert a.best_value == pytest.approx(b.best_value)


def test_one_parameter_logreg_search_is_capped(matrix, cfg):  # noqa: F811
    res = tune("logreg", matrix, cfg, n_trials=50)
    assert len(res.trials) == 12
