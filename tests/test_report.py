import numpy as np
import pandas as pd

from crimebase.report import (
    _num,
    _signed,
    by_year_table,
    classification_report_table,
    drivers_sentence,
    headline,
    md_table,
    paired_table,
    prediction_mix_sentence,
    scores_table,
    sensitivity_sentence,
    split_table,
    tuning_table,
)
from crimebase.results import SavedComparison


def make_saved(tuned_edge_lo=0.003, tuned_vs_default=(-0.008, -0.015, -0.0004)):
    cols = [
        "macro_f1", "f1_lo", "f1_hi", "balanced_acc", "accuracy", "log_loss",
        "auc_ovr", "auc_lo", "auc_hi", "trees",
    ]  # fmt: skip
    rows = {
        "majority": [0.153, 0.145, 0.162, 0.333, 0.298, 1.107, 0.5, 0.5, 0.5, np.nan],
        "persistence": [0.302, 0.285, 0.319, 0.318, 0.316, np.nan, np.nan, np.nan, np.nan, np.nan],
        "mean_reversion": [
            0.387,
            0.371,
            0.404,
            0.403,
            0.394,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        "xgboost_default": [0.417, 0.398, 0.434, 0.426, 0.424, 1.069, 0.609, 0.585, 0.630, 115],
        "xgboost_tuned": [0.409, 0.391, 0.427, 0.418, 0.417, 1.065, 0.611, 0.589, 0.631, 181],
    }
    scores = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    scores.index.name = "entry"
    d, lo, hi = tuned_vs_default
    paired = pd.DataFrame(
        [
            ("xgboost_tuned", "mean_reversion", "macro_f1", 0.022, tuned_edge_lo, 0.042, 0.98),
            ("xgboost_tuned", "xgboost_default", "macro_f1", d, lo, hi, 0.02),
        ],
        columns=["a", "b", "metric", "diff", "lo", "hi", "p_a_better"],
    )
    tuning = {
        "xgboost": {
            "best_params": {"max_depth": 2},
            "best_cv_log_loss": 1.0379,
            "default_cv_log_loss": 1.0408,
            "n_trials": 40,
            "n_folds": 4,
        }
    }
    return SavedComparison(scores, paired, tuning, pd.DataFrame(), {})


def test_md_table_is_github_markdown():
    text = md_table(pd.DataFrame({"A": ["x", "y"], "B": ["1", "2"]}))
    assert text.splitlines() == ["| A | B |", "|---|---|", "| x | 1 |", "| y | 2 |"]


def test_headline_leads_with_the_model_fixed_in_advance_and_quotes_the_numbers():
    text = headline(make_saved())
    assert text.startswith("The model fixed in advance, **tuned XGBoost**, scores **0.409**")
    assert "(95% CI 0.391 to 0.427)" in text
    assert "mean-reversion rule scores 0.387" in text and "scores 0.153" in text
    assert "+0.022" in text and "(paired 95% CI +0.003 to +0.042)" in text
    assert "no winner is picked" in text  # it does not crown the best test score


def test_headline_verdict_follows_the_interval():
    clear = headline(make_saved(tuned_edge_lo=0.003))
    unclear = headline(make_saved(tuned_edge_lo=-0.004))
    assert "an interval that excludes zero" in clear
    assert "an interval that includes zero" in unclear
    # tuning effect: [-0.015, -0.0004] excludes zero; [-0.015, +0.004] includes it
    assert "which excludes zero" in clear
    assert "which includes zero" in headline(make_saved(tuned_vs_default=(-0.005, -0.015, 0.004)))


def test_headline_names_the_range_over_all_model_variants():
    text = headline(make_saved())
    assert "All 2 model variants land between 0.409 and 0.417" in text


def test_scores_table_has_the_standard_ml_columns_and_marks_rules_as_having_no_probabilities():
    table = scores_table(make_saved()).set_index("Model")
    assert list(table.columns) == [
        "Accuracy",
        "Precision (macro)",
        "Recall (macro)",
        "F1 (macro) [95% CI]",
        "ROC-AUC [95% CI]",
        "Log loss",
        "Trees",
    ]
    assert table.loc["Mean-reversion rule", "ROC-AUC [95% CI]"] == "n/a"
    assert table.loc["Mean-reversion rule", "Log loss"] == "n/a"
    assert table.loc["XGBoost (tuned)", "ROC-AUC [95% CI]"] == "0.611 [0.589, 0.631]"
    assert table.loc["XGBoost (tuned)", "Accuracy"] == "0.417"
    assert table.loc["XGBoost (tuned)", "Trees"] == "181"
    assert table.loc["Majority class", "Trees"] == ""
    assert table.loc["XGBoost (tuned)", "Precision (macro)"] == "n/a"  # not in a saved run


def test_scores_table_uses_recomputed_precision_and_recall_when_given():
    saved = make_saved()
    extra = saved.scores.assign(precision_macro=0.425, recall_macro=0.418)
    table = scores_table(saved, extra).set_index("Model")
    assert table.loc["XGBoost (tuned)", "Precision (macro)"] == "0.425"
    assert table.loc["XGBoost (tuned)", "Recall (macro)"] == "0.418"


def test_classification_report_table_reads_like_scikit_learns():
    from crimebase.evaluate import classification_table

    y = np.array([0, 0, 1, 2, 2, 2])
    table = classification_report_table(classification_table(y, np.array([0, 1, 1, 2, 2, 0])))
    assert list(table.columns) == ["Class", "Precision", "Recall", "F1-score", "Support"]
    assert table["Class"].tolist() == [
        "down",
        "stable",
        "up",
        "accuracy",
        "macro avg",
        "weighted avg",
    ]
    acc = table[table["Class"] == "accuracy"].iloc[0]
    assert acc["Precision"] == "" and acc["Recall"] == "" and acc["F1-score"] == "0.667"
    assert table.loc[table["Class"] == "weighted avg", "Support"].iloc[0] == "6"


def test_split_table_labels_each_split_with_its_years_and_rows():
    perf = pd.DataFrame(
        {
            "years": ["2001-2016", "2017-2018", "2019-2021"],
            "rows": [24114, 4012, 12268],
            "accuracy": [0.49, 0.42, 0.41],
            "precision_macro": [0.5, 0.44, 0.42],
            "recall_macro": [0.5, 0.43, 0.41],
            "macro_f1": [0.487, 0.413, 0.39],
            "auc_ovr": [0.685, 0.628, 0.604],
            "log_loss": [1.008, 1.054, 1.074],
        },
        index=["Train", "Validation", "Test"],
    )
    table = split_table(perf)
    assert table["Split"].tolist() == ["Train", "Validation", "Test"]
    assert table["Rows"].tolist() == ["24,114", "4,012", "12,268"]
    assert table.loc[0, "F1 (macro)"] == "0.487" and table.loc[2, "ROC-AUC"] == "0.604"


def test_paired_table_says_whether_zero_is_excluded():
    table = paired_table(make_saved()).set_index("B")
    assert table.loc["Mean-reversion rule", "Excludes 0"] == "yes"
    assert table.loc["XGBoost (default)", "A - B [95% CI]"] == "-0.008 [-0.015, -0.0004]"


def test_tuning_table_reports_the_cv_gain():
    row = tuning_table(make_saved()).iloc[0]
    assert (row["Default CV log loss"], row["Tuned CV log loss"], row["Gain"]) == (
        "1.0408", "1.0379", "0.0029",
    )  # fmt: skip


def test_sensitivity_sentence_splits_clear_from_unclear_thresholds():
    sens = pd.DataFrame({"min_cases": [20, 50, 500], "diff_lo": [0.011, 0.011, -0.05]})
    text = sensitivity_sentence(sens)
    assert "clearly positive at min_cases = [20, 50]" in text
    assert "includes zero at [500]" in text


def test_by_year_table_labels_each_row_with_its_target_year():
    table = by_year_table(pd.DataFrame({"mean_reversion": [0.389, 0.372]}, index=[2019, 2020]))
    assert table["Year"].tolist() == ["2019 to 2020", "2020 to 2021"]
    assert table["Mean-reversion rule"].tolist() == ["0.389", "0.372"]


def test_numbers_get_a_fourth_decimal_only_when_three_would_print_minus_zero():
    assert _num(0.0223) == "0.022" and _num(-0.008) == "-0.008" and _num(0) == "0.000"
    assert _num(-0.0004) == "-0.0004" and _num(0.0003) == "0.0003"
    assert _signed(0.022) == "+0.022" and _signed(-0.0004) == "-0.0004"


def test_headline_never_prints_a_zero_that_hides_the_sign():
    import re

    text = headline(make_saved())
    assert not re.search(r"-0\.000(?!\d)", text)  # a bare "-0.000" would hide how small it is
    assert "-0.0004" in text


def test_drivers_sentence_names_the_top_features_and_the_rank_of_last_years_growth():
    imp = pd.DataFrame(
        {"total": [0.09, 0.08, 0.07, 0.009]}, index=["lv", "crime_category", "mean4", "g1"]
    )
    text = drivers_sentence(imp)
    assert "`lv`, `crime_category` and `mean4`" in text
    assert "ranks 4 of 4" in text


def test_prediction_mix_compares_predicted_with_actual_shares():
    y = np.array([0, 0, 1, 2])
    proba = np.eye(3)[[0, 0, 0, 0]]  # always predicts down
    text = prediction_mix_sentence(y, proba)
    assert "down 50%, stable 25%, up 25%" in text
    assert "down 100%, stable 0%, up 0%" in text


def test_markdown_tables_never_contain_a_stray_pipe_in_a_header():
    from crimebase.report import md_table

    table = md_table(pd.DataFrame({"Feature": ["lv"], "Mean abs. SHAP": ["0.088"]}))
    assert all(line.count("|") == 3 for line in table.splitlines())


def test_overfit_sentence_states_the_train_to_test_gap():
    from crimebase.report import overfit_sentence

    perf = pd.DataFrame({"macro_f1": [0.487, 0.413, 0.390]}, index=["Train", "Validation", "Test"])
    text = overfit_sentence(perf)
    assert "0.487 on the training years, 0.413 on validation and 0.390 on test" in text
    assert "gap of 0.097" in text
