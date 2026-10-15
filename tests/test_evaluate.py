import numpy as np
import pandas as pd
import pytest

from crimebase.evaluate import (
    auc_from_proba,
    binary_scores,
    binned_proba,
    bootstrap_ci,
    direction_of,
    macro_f1_from_proba,
    mean_reversion_pred,
    multiclass_scores,
    paired_bootstrap_diff,
    persistence_pred,
    prior_proba,
)

ONE_HOT = np.eye(3)


def test_perfect_probabilities_score_one():
    y = np.array([0, 1, 2, 2, 1, 0])
    s = multiclass_scores(y, proba=ONE_HOT[y] * 0.94 + 0.02)
    assert s["macro_f1"] == s["accuracy"] == s["balanced_acc"] == 1.0
    assert s["auc_ovr"] == 1.0 and s["log_loss"] < 0.05


def test_uniform_probabilities_have_log_loss_ln3_and_auc_half():
    y = np.array([0, 1, 2, 0, 1, 2])
    s = multiclass_scores(y, proba=np.full((6, 3), 1 / 3))
    assert s["log_loss"] == pytest.approx(np.log(3))
    assert s["auc_ovr"] == pytest.approx(0.5)


def test_balanced_accuracy_averages_class_recall():
    s = multiclass_scores([0, 0, 1, 2], pred=[0, 1, 1, 2])
    assert s["balanced_acc"] == pytest.approx((0.5 + 1 + 1) / 3)
    assert s["accuracy"] == pytest.approx(0.75)
    assert np.isnan(s["log_loss"]) and np.isnan(s["auc_ovr"])  # hard predictions only


def test_a_missing_class_does_not_break_macro_f1():
    s = multiclass_scores([0, 0, 2, 2], pred=[0, 0, 2, 2])
    assert s["f1_stable"] == 0.0 and s["f1_down"] == 1.0


def test_binary_scores_perfect_and_lift():
    s = binary_scores([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.9])
    assert s["roc_auc"] == s["pr_auc"] == 1.0
    assert s["base_rate"] == 0.25 and s["pr_lift"] == pytest.approx(4.0)


def test_prior_proba_is_the_training_mix():
    p = prior_proba([0, 0, 1, 2], n=5)
    assert p.shape == (5, 3) and np.allclose(p.sum(axis=1), 1)
    assert p[0].tolist() == [0.5, 0.25, 0.25]


def test_direction_persistence_and_mean_reversion_are_mirrors():
    g1 = pd.Series([0.5, -0.5, 0.05, np.nan])
    assert direction_of(g1, 0.1).tolist() == [2, 0, 1, -1]
    assert persistence_pred(g1, 0.1, fallback=1).tolist() == [2, 0, 1, 1]
    assert mean_reversion_pred(g1, 0.1, fallback=1).tolist() == [0, 2, 1, 1]


def test_binned_baseline_learns_mean_reversion_from_training_data():
    train_g1 = pd.Series([0.5] * 30 + [-0.5] * 30)
    y_train = np.array([0] * 30 + [2] * 30)  # after a rise comes a fall, and vice versa
    p = binned_proba(train_g1, y_train, pd.Series([0.4, -0.4, np.nan]), band=0.1)
    assert np.allclose(p.sum(axis=1), 1)
    assert p[0].argmax() == 0 and p[1].argmax() == 2
    assert p[2].tolist() == pytest.approx([1 / 3] * 3)  # unseen bin falls back to a flat prior


def make_scores(n, accuracy, rng):
    y = rng.integers(0, 3, n)
    right = rng.random(n) < accuracy
    pred = np.where(right, y, (y + 1) % 3)
    return y, ONE_HOT[pred] * 0.94 + 0.02


def test_bootstrap_ci_brackets_the_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    y, proba = make_scores(600, 0.6, rng)
    groups = np.arange(600) // 3
    ci = bootstrap_ci(y, proba, macro_f1_from_proba, groups, n_boot=200, seed=1)
    point = macro_f1_from_proba(y, proba)
    assert ci[0] < point < ci[1]
    assert ci == bootstrap_ci(y, proba, macro_f1_from_proba, groups, n_boot=200, seed=1)


def test_more_data_gives_a_narrower_interval():
    rng = np.random.default_rng(0)
    widths = []
    for n in (150, 3000):
        y, proba = make_scores(n, 0.6, rng)
        lo, hi = bootstrap_ci(y, proba, macro_f1_from_proba, np.arange(n), n_boot=200)
        widths.append(hi - lo)
    assert widths[1] < widths[0]


def test_group_bootstrap_is_wider_than_row_bootstrap_when_errors_cluster():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(100), 10)
    group_correct = rng.random(100) < 0.6  # a whole group is right or wrong together
    y = rng.integers(0, 3, 1000)
    pred = np.where(np.repeat(group_correct, 10), y, (y + 1) % 3)
    proba = ONE_HOT[pred] * 0.94 + 0.02
    by_row = bootstrap_ci(y, proba, macro_f1_from_proba, np.arange(1000), n_boot=300)
    by_group = bootstrap_ci(y, proba, macro_f1_from_proba, groups, n_boot=300)
    assert (by_group[1] - by_group[0]) > 1.5 * (by_row[1] - by_row[0])


def test_paired_diff_of_identical_models_is_zero_and_of_a_better_model_is_positive():
    rng = np.random.default_rng(0)
    y, weak = make_scores(1500, 0.4, rng)
    strong = ONE_HOT[y] * 0.94 + 0.02  # perfect
    groups = np.arange(1500) // 5
    same = paired_bootstrap_diff(y, weak, weak, macro_f1_from_proba, groups, n_boot=200)
    assert same["diff"] == 0 and same["lo"] == 0 == same["hi"] and same["p_a_better"] == 0
    better = paired_bootstrap_diff(y, strong, weak, macro_f1_from_proba, groups, n_boot=200)
    assert better["lo"] > 0 and better["p_a_better"] == 1.0


def test_auc_from_proba_matches_the_score_dictionary_and_is_bootstrappable():
    rng = np.random.default_rng(0)
    y, proba = make_scores(300, 0.6, rng)
    assert auc_from_proba(y, proba) == pytest.approx(multiclass_scores(y, proba=proba)["auc_ovr"])
    lo, hi = bootstrap_ci(y, proba, auc_from_proba, np.arange(300) // 3, n_boot=100)
    assert lo < auc_from_proba(y, proba) < hi


def test_macro_f1_from_proba_handles_a_sample_missing_a_class_without_warning():
    import warnings

    y = np.array([0, 0, 2, 2])  # no "stable" rows and none predicted
    proba = ONE_HOT[y] * 0.94 + 0.02
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert macro_f1_from_proba(y, proba) == pytest.approx(2 / 3)


def test_auc_is_nan_without_warning_when_a_class_is_missing_and_intervals_degrade_to_nan():
    import warnings

    y = np.array([0, 0, 2, 2, 0, 2])
    proba = ONE_HOT[y] * 0.94 + 0.02
    groups = np.arange(6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(auc_from_proba(y, proba))
        lo, hi = bootstrap_ci(y, proba, auc_from_proba, groups, n_boot=20)
        diff = paired_bootstrap_diff(y, proba, proba, auc_from_proba, groups, n_boot=20)
    assert np.isnan(lo) and np.isnan(hi)
    assert np.isnan(diff["lo"]) and np.isnan(diff["p_a_better"]) and np.isnan(diff["diff"])


def test_classification_table_matches_a_hand_computed_example():
    from crimebase.evaluate import classification_table

    y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    pred = np.array([0, 0, 1, 1, 1, 2, 2, 2, 0, 1])
    t = classification_table(y, pred)
    assert list(t.index) == ["down", "stable", "up", "accuracy", "macro avg", "weighted avg"]
    assert list(t.columns) == ["precision", "recall", "f1-score", "support"]
    # down: predicted 3 times (2 right) out of 3 true -> precision 2/3, recall 2/3
    assert t.loc["down", "precision"] == pytest.approx(2 / 3)
    assert t.loc["down", "recall"] == pytest.approx(2 / 3)
    # stable: predicted 4 times (2 right), 2 true -> precision 1/2, recall 1
    assert t.loc["stable", "precision"] == pytest.approx(0.5)
    assert t.loc["stable", "recall"] == pytest.approx(1.0)
    assert t.loc["up", "support"] == 5 and t.loc["accuracy", "f1-score"] == pytest.approx(0.7)
    assert t.loc["macro avg", "recall"] == pytest.approx((2 / 3 + 1 + 3 / 5) / 3)
    assert t.loc["weighted avg", "support"] == 10


def test_classification_table_agrees_with_scikit_learn():
    from sklearn.metrics import classification_report

    from crimebase.evaluate import classification_table

    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 400)
    pred = np.where(rng.random(400) < 0.5, y, rng.integers(0, 3, 400))
    ref = pd.DataFrame(
        classification_report(y, pred, target_names=["down", "stable", "up"], output_dict=True)
    ).T
    mine = classification_table(y, pred)
    for row in ["down", "stable", "up", "macro avg", "weighted avg"]:
        for col in ["precision", "recall", "f1-score"]:
            assert mine.loc[row, col] == pytest.approx(ref.loc[row, col])
    assert mine.loc["accuracy", "f1-score"] == pytest.approx(ref.loc["accuracy", "precision"])


def test_scores_carry_macro_precision_and_recall():
    y = np.array([0, 0, 1, 2])
    s = multiclass_scores(y, pred=np.array([0, 1, 1, 2]))
    assert s["precision_macro"] == pytest.approx((1 + 0.5 + 1) / 3)
    assert s["recall_macro"] == pytest.approx((0.5 + 1 + 1) / 3)
    assert s["recall_macro"] == s["balanced_acc"]


def test_roc_points_give_one_curve_per_class_with_the_matching_auc():
    from crimebase.evaluate import roc_points

    rng = np.random.default_rng(0)
    y, proba = make_scores(500, 0.6, rng)
    pts = roc_points(y, proba)
    assert list(pts) == ["down", "stable", "up"]
    for fpr, tpr, auc in pts.values():
        assert fpr[0] == 0 and tpr[0] == 0 and fpr[-1] == 1 and tpr[-1] == 1
        assert 0.5 < auc < 1.0
    mean_auc = np.mean([auc for _, _, auc in pts.values()])
    assert mean_auc == pytest.approx(multiclass_scores(y, proba=proba)["auc_ovr"])
