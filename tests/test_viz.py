import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
from crimebase import viz  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    viz.plt.close("all")


def scores_frame():
    return pd.DataFrame(
        {
            "macro_f1": [0.15, 0.39, 0.42, 0.41],
            "f1_lo": [0.14, 0.37, 0.40, 0.39],
            "f1_hi": [0.16, 0.40, 0.44, 0.43],
        },
        index=["majority", "mean_reversion", "xgboost_tuned", "lightgbm_tuned"],
    )


def test_palette_matches_the_validated_reference_slots():
    assert viz.SERIES == ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    assert viz.SURFACE == "#fcfcfb" and viz.INK == "#0b0b0b"


def test_scores_chart_has_one_dot_per_entry_and_both_legend_entries():
    fig = viz.plot_scores(scores_frame())
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()][-1] == "XGBoost (tuned)"  # best on top
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Baseline (no model)", "Model"]
    texts = [t.get_text() for t in ax.texts]
    assert "0.420" in texts and "0.150" in texts  # direct value labels


def test_baselines_are_grey_and_models_blue():
    fig = viz.plot_scores(scores_frame())
    dots = [ln for ln in fig.axes[0].lines if ln.get_marker() == "o"]
    colors = {ln.get_color() for ln in dots}
    assert colors == {viz.MUTED, viz.SERIES[0]}


def test_confusion_rows_sum_to_one_hundred_percent():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    proba = np.eye(3)[rng.integers(0, 3, 300)] * 0.9 + 0.033
    fig = viz.plot_confusion(y, proba)
    img = fig.axes[0].images[0].get_array()
    assert np.allclose(img.sum(axis=1), 1)


def test_line_charts_refuse_more_series_than_the_palette_supports():
    wide = pd.DataFrame(np.random.rand(3, 5), index=[2019, 2020, 2021])
    with pytest.raises(ValueError, match="at most 4"):
        viz.plot_by_year(wide)
    with pytest.raises(ValueError, match="at most 4"):
        viz.plot_lines(wide, "t", "y")


def test_by_year_labels_each_year_with_its_target_year():
    table = pd.DataFrame(
        {"mean_reversion": [0.38, 0.36, 0.40], "xgboost_tuned": [0.41, 0.40, 0.43]},
        index=[2019, 2020, 2021],
    )
    fig = viz.plot_by_year(table)
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == [
        "2019 → 2020", "2020 → 2021", "2021 → 2022",
    ]  # fmt: skip


def test_tuning_chart_marks_the_default_and_a_best_so_far_line():
    trials = pd.DataFrame({"number": range(5), "value": [1.0, 0.98, 0.99, 0.95, 0.96]})
    fig = viz.plot_tuning(trials, default_value=1.0)
    ax = fig.axes[0]
    assert any("default parameters" in t.get_text() for t in ax.texts)
    best = [ln for ln in ax.lines if ln.get_color() == viz.SERIES[0]]
    assert best and np.all(np.diff(best[0].get_ydata()) <= 0)  # best-so-far never rises


def test_importance_chart_is_one_series_largest_on_top():
    imp = pd.DataFrame({"total": [0.5, 0.3, 0.1]}, index=["a", "b", "c"])
    fig = viz.plot_importance(imp)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()][-1] == "a"
    assert {p.get_facecolor()[:3] for p in ax.patches} == {matplotlib.colors.to_rgb(viz.SERIES[0])}


def test_counts_chart_highlights_one_bar_and_explains_the_colours():
    counts = pd.Series({"ok": 40000, "below_min_cases": 167000, "no_next_year": 105000})
    fig = viz.plot_counts(counts, "t", highlight="ok")
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["ok", "excluded"]


def test_remaining_charts_render_and_save(tmp_path):
    rng = np.random.default_rng(0)
    years = [2019, 2020]
    sensitivity = pd.DataFrame(
        {"min_cases": [20, 50, 100], "model_f1": [0.43, 0.42, 0.41], "rule_f1": [0.40, 0.39, 0.38]}
    )
    mix = pd.DataFrame({"down": [0.3, 0.4], "stable": [0.3, 0.3], "up": [0.4, 0.3]}, index=years)
    lines = pd.DataFrame({"AP": [3, 2], "TS": [0, 1]}, index=[2013, 2014])
    left, right = pd.Series({"a": 0.6, "b": 0.4}), pd.Series({"a": 0.3})
    figs = [
        viz.plot_heatmap(pd.DataFrame(rng.random((4, 3)), index=[2019, 2020, 2021, 2022]), "t"),
        viz.plot_hist(rng.normal(size=200), "t", "x"),
        viz.plot_sensitivity(sensitivity),
        viz.plot_share_compare(left, right, "L", "R", "t"),
        viz.plot_label_mix(mix, "t"),
        viz.plot_learning_curve([1.1, 1.0, 0.99, 0.995], best_round=3),
        viz.plot_lines(lines, "t", "y", dashed=("TS",), annotate={2013.5: "split"}),
    ]
    for i, fig in enumerate(figs):
        out = viz.save_figure(fig, tmp_path / "sub" / f"f{i}.png")
        assert out.exists() and out.stat().st_size > 2000
