from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crimebase.models import MODEL_NAMES, balanced_weights, fit


def toy(n=1500, seed=0):
    """Three classes driven by x1 (strongly) and a category; x3 is mostly missing."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], n)
    y = np.digitize(x1 + (cat == "c") * 1.5 + rng.normal(scale=0.4, size=n), [-0.7, 0.7])
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": rng.normal(size=n),
            "x3": np.where(rng.random(n) < 0.7, np.nan, rng.normal(size=n)),
            "cat": pd.Categorical(cat, categories=["a", "b", "c"]),
        }
    )
    return X, y


@pytest.fixture(scope="module")
def data():
    X, y = toy()
    return X.iloc[:1000], y[:1000], X.iloc[1000:1250], y[1000:1250], X.iloc[1250:], y[1250:]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_learns_a_strong_signal_and_returns_probabilities(name, data):
    Xtr, ytr, Xva, yva, Xte, yte = data
    m = fit(name, Xtr, ytr, Xva, yva)
    proba = m.predict_proba(Xte)
    assert proba.shape == (len(Xte), 3)
    assert np.allclose(proba.sum(axis=1), 1, atol=1e-12, rtol=0)
    assert (proba.argmax(axis=1) == yte).mean() > 0.7


@pytest.mark.parametrize("name", ["xgboost", "lightgbm", "catboost"])
def test_early_stopping_picks_a_tree_count_and_refit_reproduces_it(name, data):
    Xtr, ytr, Xva, yva, Xte, _ = data
    stopped = fit(name, Xtr, ytr, Xva, yva)
    assert 1 <= stopped.best_iteration < 3000
    refit = fit(name, Xtr, ytr, n_estimators=stopped.best_iteration)
    assert refit.best_iteration == stopped.best_iteration
    assert refit.predict_proba(Xte).shape == (len(Xte), 3)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_same_seed_gives_identical_probabilities(name, data):
    Xtr, ytr, Xva, yva, Xte, _ = data
    a = fit(name, Xtr, ytr, Xva, yva, seed=7).predict_proba(Xte)
    b = fit(name, Xtr, ytr, Xva, yva, seed=7).predict_proba(Xte)
    assert np.allclose(a, b)


def test_params_override_defaults(data):
    Xtr, ytr, Xva, yva, _, _ = data
    shallow = fit("xgboost", Xtr, ytr, Xva, yva, params={"max_depth": 1})
    assert shallow.model.get_params()["max_depth"] == 1


def test_missing_class_and_unknown_model_are_rejected(data):
    Xtr, ytr, *_ = data
    with pytest.raises(ValueError, match="all 3 classes"):
        fit("xgboost", Xtr[ytr < 2], ytr[ytr < 2])
    with pytest.raises(ValueError, match="unknown model"):
        fit("random_forest", Xtr, ytr)


def test_balanced_weights_equalise_class_totals():
    y = np.array([0] * 80 + [1] * 15 + [2] * 5)
    w = balanced_weights(y)
    totals = [w[y == c].sum() for c in range(3)]
    assert totals == pytest.approx([totals[0]] * 3)


def test_training_leaves_no_files_behind(data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Xtr, ytr, Xva, yva, *_ = data
    fit("catboost", Xtr, ytr, Xva, yva)
    assert not list(Path(tmp_path).iterdir())  # no catboost_info/ in the working directory
