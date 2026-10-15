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


def test_rule_predictions_are_mirror_images_on_directional_rows(matrix, cfg):  # noqa: F811
    from crimebase.experiment import rule_predictions

    rules = rule_predictions(matrix, cfg)
    h = holdout_of(matrix, cfg, "y_trend")
    both = (rules["persistence"] != 1) & (rules["mean_reversion"] != 1)
    assert both.any()
    assert (rules["persistence"][both] == 2 - rules["mean_reversion"][both]).all()
    assert len(rules["persistence"]) == len(h.y)


@pytest.fixture(scope="module")
def comparison():
    """One small end-to-end run (xgboost + lightgbm, 2 trials) shared by the tests below."""
    from crimebase.experiment import run_comparison

    base = load_config()
    cfg = replace(
        base,
        split=replace(
            base.split,
            train_to=2013,
            val_years=(2014, 2015),
            test_from=2016,
            cv_windows=((2010, 2011), (2012, 2013), (2014, 2015)),
        ),
    )
    from conftest import make_panel
    from crimebase.features import build_matrix

    m = build_matrix(make_panel(), cfg.task)
    messages = []
    cmp = run_comparison(
        m, cfg, models=("xgboost", "lightgbm"), n_trials=2, n_boot=40, log=messages.append
    )
    return cmp, messages


def test_comparison_rows_are_baselines_then_models_then_ensemble(comparison):
    cmp, _ = comparison
    assert list(cmp.scores.index) == [
        "majority",
        "persistence",
        "mean_reversion",
        "binned_g1",
        "xgboost_default",
        "xgboost_tuned",
        "lightgbm_default",
        "lightgbm_tuned",
        "ensemble",
    ]
    assert {"macro_f1", "f1_lo", "f1_hi", "auc_ovr", "auc_lo", "auc_hi", "trees"} <= set(
        cmp.scores.columns
    )


def test_intervals_are_ordered_and_rules_have_no_auc(comparison):
    cmp, _ = comparison
    assert (cmp.scores.f1_lo <= cmp.scores.f1_hi).all()
    assert cmp.scores.loc["mean_reversion", ["auc_ovr", "auc_lo", "auc_hi"]].isna().all()
    assert cmp.scores.loc["xgboost_tuned", "auc_lo"] <= cmp.scores.loc["xgboost_tuned", "auc_hi"]


def test_tuning_never_scores_worse_than_defaults_on_cv(comparison):
    cmp, _ = comparison
    assert set(cmp.tuning) == {"xgboost", "lightgbm"}
    assert all(t.best_value <= t.default_value for t in cmp.tuning.values())


def test_ensemble_is_the_mean_of_the_tuned_members(comparison):
    cmp, _ = comparison
    expected = (cmp.probas["xgboost_tuned"] + cmp.probas["lightgbm_tuned"]) / 2
    assert np.allclose(cmp.probas["ensemble"], expected)


def test_paired_rows_exist_only_for_pairs_that_were_run(comparison):
    cmp, _ = comparison
    pairs = {(r.a, r.b, r.metric) for r in cmp.paired.itertuples()}
    assert ("xgboost_tuned", "mean_reversion", "macro_f1") in pairs
    assert ("xgboost_tuned", "mean_reversion", "auc_ovr") not in pairs  # a rule has no AUC
    assert ("xgboost_tuned", "xgboost_default", "auc_ovr") in pairs
    assert ("ensemble", "xgboost_tuned", "macro_f1") in pairs
    assert not any(
        "catboost" in a or "logreg" in a or "catboost" in b or "logreg" in b for a, b, _ in pairs
    )


def test_paired_difference_equals_the_difference_of_scores(comparison):
    cmp, _ = comparison
    row = cmp.paired.query(
        "a == 'xgboost_tuned' and b == 'xgboost_default' and metric == 'macro_f1'"
    ).iloc[0]
    s = cmp.scores
    assert row["diff"] == pytest.approx(
        s.loc["xgboost_tuned", "macro_f1"] - s.loc["xgboost_default", "macro_f1"]
    )
    assert row.lo <= row.hi and 0 <= row.p_a_better <= 1


def test_progress_messages_name_each_step(comparison):
    _, messages = comparison
    assert any("xgboost: tuning 2 trials" in msg for msg in messages)
    assert any(msg.startswith("ensemble") for msg in messages)


def test_final_fit_returns_the_model_that_final_fit_predict_uses(matrix, cfg):  # noqa: F811
    from crimebase.experiment import final_fit
    from crimebase.features import FEATURES

    model, trees = final_fit("xgboost", matrix, cfg, "y_trend")
    te = holdout_mask(matrix, cfg.split, "y_trend")
    proba, trees2 = final_fit_predict("xgboost", matrix, cfg, "y_trend")
    assert trees == trees2 and model.best_iteration == trees
    assert np.allclose(model.predict_proba(matrix.loc[te, FEATURES]), proba)


def test_min_cases_sensitivity_changes_the_labelled_rows_and_reports_paired_diffs():
    from conftest import make_panel
    from crimebase.experiment import min_cases_sensitivity

    base = load_config()
    cfg = replace(
        base,
        split=replace(
            base.split,
            train_to=2013,
            val_years=(2014, 2015),
            test_from=2016,
            cv_windows=((2010, 2011), (2012, 2013), (2014, 2015)),
        ),
    )
    table = min_cases_sensitivity(make_panel(), cfg, values=(50, 300), n_boot=30)
    assert list(table.min_cases) == [50, 300]
    assert table.n_test.iloc[0] > table.n_test.iloc[1]  # a higher bar leaves fewer rows
    assert (table.diff_lo <= table.diff_hi).all()
    assert np.allclose(table["diff"], table.model_f1 - table.rule_f1)


def test_split_performance_scores_one_model_on_train_validation_and_test(matrix, cfg):  # noqa: F811
    from crimebase.experiment import split_performance

    table = split_performance(matrix, cfg)
    assert list(table.index) == ["Train", "Validation", "Test"]
    assert {"years", "rows", "accuracy", "precision_macro", "recall_macro", "macro_f1"} <= set(
        table.columns
    )
    assert table.loc["Train", "rows"] > table.loc["Validation", "rows"] > 0
    assert table.loc["Test", "years"].startswith(str(cfg.split.test_from))
    # the training rows were seen by the model: it should not do worse there than on test
    assert table.loc["Train", "log_loss"] <= table.loc["Test", "log_loss"] + 0.05
