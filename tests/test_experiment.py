from dataclasses import replace

import numpy as np
import pytest

from crimebase.config import load_config
from crimebase.experiment import baseline_results, final_fit_predict, holdout_of, rule_as_proba
from crimebase.splits import dev_mask, holdout_mask
from test_splits import matrix, split  # noqa: F401  (fixtures: synthetic matrix + shifted windows)


@pytest.fixture
def cfg(split):  # noqa: F811
    return replace(load_config(), split=split)


def test_holdout_contains_only_labelled_rows_after_the_test_year(matrix, cfg):  # noqa: F811
    h = holdout_of(matrix, cfg, "y_trend")
    assert len(h.y) == holdout_mask(matrix, cfg.split, "y_trend").sum()
    assert h.year.min() >= cfg.split.test_from
    assert set(np.unique(h.y)) <= {0, 1, 2}
    assert len(set(h.groups)) > 1 and h.groups[0].count("-") == 1  # "state-year"


def test_final_fit_predicts_every_holdout_row_with_probabilities(matrix, cfg):  # noqa: F811
    h = holdout_of(matrix, cfg, "y_trend")
    proba, trees = final_fit_predict("xgboost", matrix, cfg, "y_trend")
    assert proba.shape == (len(h.y), 3) and np.allclose(proba.sum(axis=1), 1)
    assert trees >= 1


def test_final_fit_never_trains_on_test_years(matrix, cfg, monkeypatch):  # noqa: F811
    seen = []
    import crimebase.experiment as exp

    real_fit = exp.fit

    def spy(name, X_train, y_train, *args, **kwargs):
        seen.append(matrix.loc[X_train.index, "year"].max())
        return real_fit(name, X_train, y_train, *args, **kwargs)

    monkeypatch.setattr(exp, "fit", spy)
    final_fit_predict("lightgbm", matrix, cfg, "y_trend")
    assert seen and max(seen) < cfg.split.test_from
    assert seen[-1] == matrix.year[dev_mask(matrix, cfg.split, "y_trend")].max()  # refit uses dev


def test_baselines_report_probabilities_only_where_they_have_them(matrix, cfg):  # noqa: F811
    scores, probas = baseline_results(matrix, cfg)
    assert set(scores) == {"majority", "binned_g1", "persistence", "mean_reversion"}
    assert set(probas) == {"majority", "binned_g1"}
    assert np.isnan(scores["persistence"]["log_loss"])
    assert np.isfinite(scores["majority"]["log_loss"]) and np.isfinite(
        scores["binned_g1"]["auc_ovr"]
    )
    assert scores["majority"]["auc_ovr"] == pytest.approx(0.5)


def test_rule_as_proba_is_a_valid_probability_matrix():
    p = rule_as_proba(np.array([0, 1, 2, 2]))
    assert np.allclose(p.sum(axis=1), 1) and (p.argmax(axis=1) == [0, 1, 2, 2]).all()
