import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from crimebase.explain import contributions, importance, signed_effect
from crimebase.models import fit
from test_models import toy


@pytest.fixture(scope="module")
def trained():
    X, y = toy()
    model = fit("xgboost", X.iloc[:1000], y[:1000], X.iloc[1000:1250], y[1000:1250])
    return model, X.iloc[1250:]


def test_contributions_add_up_to_the_models_raw_margins(trained):
    model, X = trained
    contrib = contributions(model, X)
    assert contrib.shape == (len(X), 3, X.shape[1] + 1)
    margin = model.model.get_booster().predict(
        xgb.DMatrix(X, enable_categorical=True), output_margin=True
    )
    assert np.allclose(contrib.sum(axis=2), margin, atol=1e-4)


def test_the_driving_features_rank_first(trained):
    model, X = trained
    imp = importance(contributions(model, X), list(X.columns))
    assert imp.index[:2].tolist() == ["x1", "cat"]  # y was built from x1 and cat only
    assert imp.loc["x2", "total"] < 0.2 * imp.loc["x1", "total"]
    assert list(imp.columns) == ["down", "stable", "up", "total"]
    assert imp.total.is_monotonic_decreasing


def test_signed_effect_matches_how_the_toy_target_was_built(trained):
    model, X = trained
    contrib = contributions(model, X)
    up = signed_effect(contrib, X, class_index=2)
    down = signed_effect(contrib, X, class_index=0)
    assert up.loc["x1", "up"] > 0.8 and down.loc["x1", "down"] < -0.8
    assert "cat" not in up.index  # categorical columns are skipped


def test_only_xgboost_models_are_supported():
    X, y = toy()
    lgbm = fit("lightgbm", X.iloc[:1000], y[:1000], X.iloc[1000:1250], y[1000:1250])
    with pytest.raises(ValueError, match="xgboost"):
        contributions(lgbm, X.iloc[1250:])


def test_importance_handles_a_pandas_feature_list(trained):
    model, X = trained
    imp = importance(contributions(model, X), list(X.columns))
    assert isinstance(imp, pd.DataFrame) and set(imp.index) == set(X.columns)
