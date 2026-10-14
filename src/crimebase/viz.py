"""Charts for the notebooks and the report. Every function returns a matplotlib Figure.

Colour follows one palette (validated for colour-vision safety): categorical hues are used in
a fixed order, one hue marks a single series, a one-hue ramp marks magnitude, and text is always
ink, never a series colour. Where a colour sits below 3:1 contrast on the surface, the values are
also printed (direct labels) and the notebooks show the same numbers as a table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# categorical slots 1-4 in fixed order (blue, orange, aqua, yellow)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", SEQ_STEPS)

LABELS = {
    "majority": "Majority class",
    "persistence": "Persistence rule",
    "mean_reversion": "Mean-reversion rule",
    "binned_g1": "Last-year-direction table",
    "xgboost_default": "XGBoost (default)",
    "xgboost_tuned": "XGBoost (tuned)",
    "lightgbm_default": "LightGBM (default)",
    "lightgbm_tuned": "LightGBM (tuned)",
    "catboost_default": "CatBoost (default)",
    "catboost_tuned": "CatBoost (tuned)",
    "logreg_default": "Logistic regression (default)",
    "logreg_tuned": "Logistic regression (tuned)",
    "ensemble": "Ensemble (XGB + LGBM + Cat)",
}
BASELINES = ("majority", "persistence", "mean_reversion", "binned_g1")


def label(entry: str) -> str:
    return LABELS.get(entry, entry)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Segoe UI", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.solid_capstyle": "round",
        }
    )


def _head(fig, ax, title: str, subtitle: str | None = None) -> None:
    """Title above an optional subtitle; both offsets are in points so tall figures stay tidy."""
    ax.set_title(title, color=INK, pad=24 if subtitle else 10)
    if subtitle:
        ax.annotate(
            subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5),
            textcoords="offset points", color=INK2, fontsize=9, va="bottom",
        )  # fmt: skip


def save_figure(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


# --- model comparison ------------------------------------------------------------------


def plot_scores(
    scores: pd.DataFrame,
    metric: str = "macro_f1",
    lo: str = "f1_lo",
    hi: str = "f1_hi",
    xlabel: str = "Macro-F1 on the 2019+ holdout",
    title: str = "Models against baselines",
    subtitle: str | None = "Dot = score, line = 95% interval (resampling state-year groups)",
):
    """One dot and interval per entry, best at the top; baselines grey, models blue."""
    apply_style()
    df = scores.sort_values(metric)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(df) + 1.8), layout="constrained")
    for yi, (entry, row) in zip(y, df.iterrows(), strict=True):
        color = MUTED if entry in BASELINES else SERIES[0]
        ax.plot([row[lo], row[hi]], [yi, yi], color=color, lw=2, alpha=0.9)
        ax.plot(row[metric], yi, "o", ms=9, color=color, mec=SURFACE, mew=2)
        ax.text(row[hi] + 0.004, yi, f"{row[metric]:.3f}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, [label(e) for e in df.index])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_xlim(left=max(0, df[lo].min() - 0.03))
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=MUTED, ms=8, label="Baseline (no model)"),
            Line2D([], [], marker="o", ls="", color=SERIES[0], ms=8, label="Model"),
        ],
        loc="lower right",
    )
    _head(fig, ax, title, subtitle)
    return fig


def plot_confusion(y_true, proba, title: str = "Where the model is right and wrong"):
    """Row-normalised confusion matrix: of the rows that truly were X, the share predicted as Y."""
    apply_style()
    names = ["down", "stable", "up"]
    pred = np.asarray(proba).argmax(axis=1)
    counts = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(pred, name="pred"))
    counts = counts.reindex(index=range(3), columns=range(3), fill_value=0)
    share = counts.div(counts.sum(axis=1), axis=0).to_numpy()
    fig, ax = plt.subplots(figsize=(4.8, 4.4), layout="constrained")
    ax.imshow(share, cmap=SEQUENTIAL, vmin=0, vmax=max(share.max(), 0.5))
    for i in range(3):
        for j in range(3):
            dark = share[i, j] > 0.38
            ax.text(
                j, i, f"{share[i, j]:.0%}\n{counts.iloc[i, j]:,}", ha="center", va="center",
                color=SURFACE if dark else INK, fontsize=10,
            )  # fmt: skip
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _head(fig, ax, title, "Share of each actual class (count underneath)")
    return fig


def plot_by_year(table: pd.DataFrame, title: str = "Macro-F1 by holdout year"):
    """table: rows = feature year, columns = entry names. One line per entry (max 4)."""
    apply_style()
    if table.shape[1] > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} series; fold the rest or use small multiples")
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    for color, col in zip(SERIES, table.columns, strict=False):
        ax.plot(table.index, table[col], color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2)
        ax.text(
            table.index[-1] + 0.06, table[col].iloc[-1], f"{table[col].iloc[-1]:.3f}",
            va="center", color=INK2, fontsize=9,
        )  # fmt: skip
    ax.set_xticks(list(table.index), [f"{y} → {y + 1}" for y in table.index])
    ax.set_xlim(table.index.min() - 0.2, table.index.max() + 0.6)
    ax.set_ylabel("Macro-F1")
    ax.legend(
        handles=[
            Line2D([], [], color=c, lw=2, marker="o", ms=7, label=label(col))
            for c, col in zip(SERIES, table.columns, strict=False)
        ],
        loc="best",
    )
    _head(fig, ax, title, "Feature year → year being predicted")
    return fig


# --- tuning and explanation ------------------------------------------------------------


def plot_tuning(trials: pd.DataFrame, default_value: float, title: str = "Tuning history"):
    """Each trial's CV log loss, the best so far, and the default parameters as a reference."""
    apply_style()
    done = trials.dropna(subset=["value"]).sort_values("number")
    fig, ax = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
    ax.plot(done["number"], done["value"], "o", color=MUTED, ms=6, mec=SURFACE, mew=1.5)
    ax.step(done["number"], done["value"].cummin(), where="post", color=SERIES[0], lw=2)
    ax.axhline(default_value, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(
        done["number"].max(), default_value, " default parameters", color=INK2, fontsize=9,
        va="bottom", ha="right",
    )  # fmt: skip
    ax.set_xlabel("Trial")
    ax.set_ylabel("Cross-validated log loss (lower is better)")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=MUTED, ms=6, label="Trial"),
            Line2D([], [], color=SERIES[0], lw=2, label="Best so far"),
        ],
        loc="lower left",
    )
    _head(fig, ax, title, "Time-ordered folds; the 2019+ test years take no part")
    return fig


def plot_importance(imp: pd.DataFrame, top: int = 15, title: str = "What the model relies on"):
    """Mean |SHAP| over the holdout rows, averaged over the three classes."""
    apply_style()
    df = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(df) + 1.8), layout="constrained")
    ax.barh(df.index, df["total"], color=SERIES[0], height=0.62)
    for yi, v in enumerate(df["total"]):
        ax.text(v + df["total"].max() * 0.01, yi, f"{v:.3f}", va="center", color=INK2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP value| (change in log-odds)")
    _head(fig, ax, title, "XGBoost TreeSHAP on the held-out rows")
    return fig


# --- data understanding ----------------------------------------------------------------


def plot_counts(
    counts: pd.Series, title: str, subtitle: str | None = None, highlight: str | None = None
):
    """Horizontal bars with the count printed on each, largest on top.

    One hue because these are nominal categories (bar length already carries magnitude). With
    `highlight`, that one bar is blue and the rest grey, and a legend says which is which.
    """
    apply_style()
    s = counts.sort_values()
    fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(s) + 1.8), layout="constrained")
    colors = [SERIES[0] if (highlight is None or k == highlight) else MUTED for k in s.index]
    ax.barh([str(k) for k in s.index], s.to_numpy(), color=colors, height=0.62)
    for yi, v in enumerate(s):
        ax.text(v + s.max() * 0.01, yi, f"{v:,.0f}", va="center", color=INK2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=s.max() * 1.12)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    if highlight is not None:
        ax.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, color=SERIES[0], label=highlight),
                plt.Rectangle((0, 0), 1, 1, color=MUTED, label="excluded"),
            ],
            loc="lower right",
        )
    _head(fig, ax, title, subtitle)
    return fig


def plot_heatmap(
    table: pd.DataFrame, title: str, subtitle: str | None = None, cbar_label: str = "rows"
):
    """Magnitude on one hue ramp: light is few, dark is many."""
    apply_style()
    fig, ax = plt.subplots(
        figsize=(0.46 * table.shape[1] + 3.2, 0.3 * table.shape[0] + 2.2), layout="constrained"
    )
    im = ax.imshow(table.to_numpy(dtype=float), cmap=SEQUENTIAL, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns], rotation=40, ha="right")
    ax.set_yticks(range(table.shape[0]), [str(i) for i in table.index])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(cbar_label, color=INK2)
    cb.outline.set_visible(False)
    _head(fig, ax, title, subtitle)
    return fig


def plot_hist(values, title: str, xlabel: str, subtitle: str | None = None, bins: int = 40):
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    ax.hist(np.asarray(values), bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Rows")
    ax.grid(axis="x", visible=False)
    _head(fig, ax, title, subtitle)
    return fig


def plot_sensitivity(table: pd.DataFrame, title: str = "Does the threshold matter?"):
    """Macro-F1 of the model and of the mean-reversion rule as the minimum count rises."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    series = [
        ("model_f1", "XGBoost (default)", SERIES[0]),
        ("rule_f1", "Mean-reversion rule", SERIES[1]),
    ]
    for col, _, color in series:
        ax.plot(
            table["min_cases"], table[col], color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2
        )
        ax.text(
            table["min_cases"].iloc[-1] * 1.08, table[col].iloc[-1], f"{table[col].iloc[-1]:.3f}",
            va="center", color=INK2, fontsize=9,
        )  # fmt: skip
    ax.set_xscale("log")
    ax.set_xticks(table["min_cases"], [str(v) for v in table["min_cases"]])
    ax.minorticks_off()
    ax.set_xlabel("Minimum registered cases in the starting year")
    ax.set_ylabel("Macro-F1 on the 2019+ holdout")
    ax.legend(
        handles=[Line2D([], [], color=c, lw=2, marker="o", ms=7, label=n) for _, n, c in series],
        loc="best",
    )
    _head(fig, ax, title, "Same rows for both lines at each threshold")
    return fig


def plot_lines(
    table: pd.DataFrame,
    title: str,
    ylabel: str,
    subtitle: str | None = None,
    dashed: tuple[str, ...] = (),
    annotate: dict[float, str] | None = None,
):
    """One line per column of `table` (max 4) over its index; `dashed` columns are derived series.

    `annotate` maps an x position to a short note drawn as a vertical hairline.
    """
    apply_style()
    if table.shape[1] > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} series; fold the rest or use small multiples")
    fig, ax = plt.subplots(figsize=(7.6, 4.2), layout="constrained")
    for color, col in zip(SERIES, table.columns, strict=False):
        style = (0, (4, 3)) if col in dashed else "-"
        ax.plot(table.index, table[col], color=color, lw=2, ls=style, marker="o", ms=5,
                mec=SURFACE, mew=1.5)  # fmt: skip
    for x, note in (annotate or {}).items():
        ax.axvline(x, color=AXIS, lw=0.8)
        ax.text(x, ax.get_ylim()[1], f" {note}", color=INK2, fontsize=8, va="top")
    ax.set_ylabel(ylabel)
    if np.nanmax(table.to_numpy(dtype=float)) >= 10_000:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.legend(
        handles=[
            Line2D([], [], color=c, lw=2, ls=(0, (4, 3)) if col in dashed else "-", label=str(col))
            for c, col in zip(SERIES, table.columns, strict=False)
        ],
        loc="best",
    )
    _head(fig, ax, title, subtitle)
    return fig


def plot_share_compare(
    left: pd.Series, right: pd.Series, left_name: str, right_name: str, title: str,
    subtitle: str | None = None,
):  # fmt: skip
    """Two shares side by side per category (two series: blue and orange), sorted by the first."""
    apply_style()
    idx = left.sort_values().index
    y = np.arange(len(idx))
    fig, ax = plt.subplots(figsize=(7.8, 0.44 * len(idx) + 1.9), layout="constrained")
    ax.barh(y + 0.19, left.reindex(idx), height=0.34, color=SERIES[0])
    ax.barh(y - 0.19, right.reindex(idx).fillna(0), height=0.34, color=SERIES[1])
    ax.set_yticks(y, [str(i) for i in idx])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of the total")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=SERIES[0], label=left_name),
            plt.Rectangle((0, 0), 1, 1, color=SERIES[1], label=right_name),
        ],
        loc="lower right",
    )
    _head(fig, ax, title, subtitle)
    return fig


def plot_label_mix(table: pd.DataFrame, title: str, subtitle: str | None = None):
    """Stacked shares of down / stable / up per year: blue and red poles around a grey middle."""
    apply_style()
    colors = {"down": SERIES[0], "stable": AXIS, "up": "#e34948"}
    fig, ax = plt.subplots(figsize=(8.0, 4.0), layout="constrained")
    bottom = np.zeros(len(table))
    for name in ("down", "stable", "up"):
        ax.bar(table.index, table[name], bottom=bottom, color=colors[name], width=0.78,
               edgecolor=SURFACE, linewidth=1.5, label=name)  # fmt: skip
        bottom += table[name].to_numpy()
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xticks(list(table.index), [str(y) for y in table.index], rotation=45, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Share of labelled rows")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _head(fig, ax, title, subtitle)
    return fig


def plot_learning_curve(history: list[float], best_round: int, title: str = "Early stopping"):
    """Validation log loss against boosting rounds, with the round that was kept."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    rounds = np.arange(1, len(history) + 1)
    ax.plot(rounds, history, color=SERIES[0], lw=2)
    ax.axvline(best_round, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(best_round, max(history), f" kept round {best_round}", color=INK2, fontsize=9, va="top")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Validation log loss")
    _head(
        fig,
        ax,
        title,
        "Training stops when the 2017-18 validation loss has not improved for 100 rounds",
    )
    return fig


def show(fig, dpi: int = 130) -> None:
    """Display a figure in a notebook as a PNG and close it (the Agg backend has no inline view)."""
    import io

    from IPython.display import Image, display

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    display(Image(buf.getvalue()))
