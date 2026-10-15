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


def test_show_displays_a_png_and_closes_the_figure(monkeypatch):
    import IPython.display

    shown = []
    monkeypatch.setattr(IPython.display, "display", shown.append)
    fig = viz.plot_hist(np.arange(10), "t", "x")
    viz.show(fig)
    assert len(shown) == 1 and shown[0].data[:8] == b"\x89PNG\r\n\x1a\n"
    assert not viz.plt.fignum_exists(fig.number)


def test_line_chart_uses_thousands_separators_for_large_values():
    big = pd.DataFrame({"AP": [218015, 114604], "TS": [np.nan, 106830]}, index=[2013, 2014])
    fig = viz.plot_lines(big, "t", "y")
    fig.canvas.draw()
    ticks = [t.get_text() for t in fig.axes[0].get_yticklabels() if t.get_text()]
    assert any("," in t for t in ticks)


def test_tuning_chart_ignores_pruned_trials():
    trials = pd.DataFrame(
        {
            "number": range(5),
            "value": [
                1.04,
                1.03,
                1.02,
                0.90,
                1.035,
            ],  # trial 3 is pruned: a partial, too-good value
            "state": ["COMPLETE", "COMPLETE", "COMPLETE", "PRUNED", "COMPLETE"],
        }
    )
    fig = viz.plot_tuning(trials, default_value=1.04)
    ax = fig.axes[0]
    best = [ln for ln in ax.lines if ln.get_color() == viz.SERIES[0]][0]
    assert min(best.get_ydata()) == 1.02  # not the pruned 0.90
    dots = [ln for ln in ax.lines if ln.get_marker() == "o"][0]
    assert len(dots.get_xdata()) == 4
    assert any("1 weak trial(s) stopped early" in t.get_text() for t in ax.texts)


def test_spread_separates_equal_values_and_keeps_order():
    out = viz._spread([0.406, 0.406, 0.359], min_gap=0.01)
    assert out[2] == 0.359 and out[0] == 0.406 and out[1] >= 0.416 - 1e-12
    assert viz._spread([0.1, 0.5], min_gap=0.01) == [0.1, 0.5]  # well apart, untouched


def test_by_year_end_labels_do_not_overprint_when_lines_end_together():
    table = pd.DataFrame(
        {"a": [0.38, 0.406], "b": [0.381, 0.406], "c": [0.34, 0.359]}, index=[2019, 2020]
    )
    fig = viz.plot_by_year(table)
    labels = [t for t in fig.axes[0].texts if t.get_text().replace(".", "").isdigit()]
    ys = sorted(t.get_position()[1] for t in labels)
    assert len(ys) == 3 and min(b - a for a, b in zip(ys, ys[1:], strict=False)) > 0
