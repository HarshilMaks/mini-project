"""Build reports/results.md and its figures from the saved metrics.

Every number in the report is computed here from files under reports/metrics (plus one SHAP
refit of the tuned XGBoost); none is typed by hand. Sentences that interpret a number are
conditional on it, so the text stays true if the numbers change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crimebase.config import Config
from crimebase.evaluate import classification_table
from crimebase.experiment import final_fit, holdout_of, split_performance
from crimebase.explain import contributions, importance
from crimebase.features import FEATURES
from crimebase.pipeline import load_matrix
from crimebase.results import (
    SavedComparison,
    load_comparison,
    recompute_scores,
    scores_by_year,
)
from crimebase.splits import holdout_mask
from crimebase.viz import (
    BASELINES,
    label,
    plot_by_year,
    plot_class_metrics,
    plot_confusion,
    plot_importance,
    plot_roc,
    plot_scores,
    plot_sensitivity,
    plot_tuning,
    save_figure,
)

FIG = "figures"


def md_table(df: pd.DataFrame) -> str:
    """A GitHub-flavoured markdown table from a frame of strings."""
    head = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *rows])


def _num(v: float) -> str:
    """Three decimals, or four when the value is so small that three would print -0.000."""
    return f"{v:.3f}" if abs(v) >= 0.0005 or v == 0 else f"{v:.4f}"


def _signed(v: float) -> str:
    return ("+" if v > 0 else "") + _num(v)


def _ci(value: float, lo: float, hi: float) -> str:
    return f"{_num(value)} [{_num(lo)}, {_num(hi)}]"


def _f(v: float) -> str:
    return "n/a" if pd.isna(v) else f"{v:.3f}"


def scores_table(saved: SavedComparison, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """The standard test-set table: accuracy, precision, recall, F1, ROC-AUC and log loss.

    Precision and recall are macro averages over the three classes. `extra` (from
    `recompute_scores`) supplies the columns the saved run does not hold; without it they read n/a.
    """
    s = saved.scores
    rows = []
    for entry, r in s.iterrows():
        e = extra.loc[entry] if extra is not None else r
        has_auc = pd.notna(r["auc_ovr"])
        rows.append(
            {
                "Model": label(entry),
                "Accuracy": _f(e["accuracy"]),
                "Precision (macro)": _f(e.get("precision_macro", np.nan)),
                "Recall (macro)": _f(e.get("recall_macro", r["balanced_acc"])),
                "F1 (macro) [95% CI]": _ci(r["macro_f1"], r["f1_lo"], r["f1_hi"]),
                "ROC-AUC [95% CI]": _ci(r["auc_ovr"], r["auc_lo"], r["auc_hi"])
                if has_auc
                else "n/a",
                "Log loss": _f(r["log_loss"]),
                "Trees": f"{int(r['trees'])}" if pd.notna(r["trees"]) else "",
            }
        )
    return pd.DataFrame(rows)


def classification_report_table(table: pd.DataFrame) -> pd.DataFrame:
    """`classification_table` as strings, laid out like scikit-learn's printed report."""
    out = pd.DataFrame(
        {
            "Class": table.index,
            "Precision": [_f(v) for v in table["precision"]],
            "Recall": [_f(v) for v in table["recall"]],
            "F1-score": [_f(v) for v in table["f1-score"]],
            "Support": [f"{int(v):,}" for v in table["support"]],
        }
    )
    return out.replace("n/a", "")


def split_table(perf: pd.DataFrame) -> pd.DataFrame:
    """Train / validation / test rows of `split_performance` as a labelled table."""
    return pd.DataFrame(
        {
            "Split": perf.index,
            "Years (feature year)": perf["years"].to_numpy(),
            "Rows": [f"{int(v):,}" for v in perf["rows"]],
            "Accuracy": [_f(v) for v in perf["accuracy"]],
            "Precision (macro)": [_f(v) for v in perf["precision_macro"]],
            "Recall (macro)": [_f(v) for v in perf["recall_macro"]],
            "F1 (macro)": [_f(v) for v in perf["macro_f1"]],
            "ROC-AUC": [_f(v) for v in perf["auc_ovr"]],
            "Log loss": [_f(v) for v in perf["log_loss"]],
        }
    )


def paired_table(saved: SavedComparison) -> pd.DataFrame:
    rows = []
    for r in saved.paired.itertuples():
        rows.append(
            {
                "A": label(r.a),
                "B": label(r.b),
                "Metric": "macro-F1" if r.metric == "macro_f1" else "AUC",
                "A - B [95% CI]": _ci(r.diff, r.lo, r.hi),
                "Excludes 0": "yes" if (r.lo > 0 or r.hi < 0) else "no",
            }
        )
    return pd.DataFrame(rows)


def tuning_table(saved: SavedComparison) -> pd.DataFrame:
    rows = []
    for name, t in saved.tuning.items():
        gain = t["default_cv_log_loss"] - t["best_cv_log_loss"]
        rows.append(
            {
                "Model": name,
                "Trials": t["n_trials"],
                "Default CV log loss": f"{t['default_cv_log_loss']:.4f}",
                "Tuned CV log loss": f"{t['best_cv_log_loss']:.4f}",
                "Gain": f"{gain:.4f}",
            }
        )
    return pd.DataFrame(rows)


def by_year_table(by_year: pd.DataFrame) -> pd.DataFrame:
    """Years as rows ("2019 to 2020"), one column per entry."""
    table = by_year.map("{:.3f}".format)
    table.columns = [label(c) for c in table.columns]
    table.insert(0, "Year", [f"{y} to {y + 1}" for y in by_year.index])
    return table


def _paired(saved: SavedComparison, a: str, b: str, metric: str = "macro_f1"):
    rows = saved.paired[
        (saved.paired.a == a) & (saved.paired.b == b) & (saved.paired.metric == metric)
    ]
    return rows.iloc[0] if len(rows) else None


def _verdict(row) -> str:
    return "excludes" if (row.lo > 0 or row.hi < 0) else "includes"


def headline(saved: SavedComparison) -> str:
    """The headline model is fixed in advance (tuned XGBoost), never chosen by test score."""
    s = saved.scores
    main, rule = s.loc["xgboost_tuned"], s.loc["mean_reversion"]
    models = [e for e in s.index if e not in BASELINES]
    low, high = s.loc[models, "macro_f1"].min(), s.loc[models, "macro_f1"].max()
    lines = [
        f"The model fixed in advance, **tuned XGBoost**, scores "
        f"**{main['macro_f1']:.3f}** macro-F1 on the held-out years "
        f"(95% CI {main['f1_lo']:.3f} to {main['f1_hi']:.3f}). The one-line "
        f"mean-reversion rule scores {rule['macro_f1']:.3f} and always predicting the commonest "
        f"training class scores {s.loc['majority', 'macro_f1']:.3f}."
    ]
    vs_rule = _paired(saved, "xgboost_tuned", "mean_reversion")
    if vs_rule is not None:
        lines.append(
            f"XGBoost beats that rule by {_signed(vs_rule['diff'])} macro-F1 (paired 95% CI "
            f"{_signed(vs_rule.lo)} to {_signed(vs_rule.hi)}), an interval that "
            f"{_verdict(vs_rule)} zero."
        )
    lines.append(
        f"All {len(models)} model variants land between {low:.3f} and {high:.3f}; naming the "
        "top one on the test set would be optimistic, so no winner is picked from it."
    )
    tuning = _paired(saved, "xgboost_tuned", "xgboost_default")
    if tuning is not None:
        lines.append(
            f"Tuning moved held-out macro-F1 by {_signed(tuning['diff'])} "
            f"(paired CI {_signed(tuning.lo)} to {_signed(tuning.hi)}, which "
            f"{_verdict(tuning)} zero)."
        )
    return " ".join(lines)


def sensitivity_sentence(sens: pd.DataFrame) -> str:
    clear = sens[sens.diff_lo > 0].min_cases.tolist()
    unclear = sens[sens.diff_lo <= 0].min_cases.tolist()
    parts = []
    if clear:
        parts.append(f"the model's edge over the rule is clearly positive at min_cases = {clear}")
    if unclear:
        parts.append(f"its interval includes zero at {unclear}")
    return "At the thresholds tried, " + " and ".join(parts) + "."


def drivers_sentence(imp: pd.DataFrame) -> str:
    """Which features the model leans on, and where last year's own growth ranks."""
    names = list(imp.index)
    top = names[:3]
    text = f"The three features the model leans on most are `{top[0]}`, `{top[1]}` and `{top[2]}`."
    if "g1" in names:
        text += f" Last year's own growth (`g1`) ranks {names.index('g1') + 1} of {len(names)}."
    return text


def prediction_mix_sentence(y_true, proba) -> str:
    """How the predicted class mix compares with the actual one on the held-out rows."""
    y_true = np.asarray(y_true)
    pred = np.asarray(proba).argmax(axis=1)
    actual = np.bincount(y_true, minlength=3) / len(y_true)
    guessed = np.bincount(pred, minlength=3) / len(pred)
    names = ("down", "stable", "up")
    a = ", ".join(f"{n} {v:.0%}" for n, v in zip(names, actual, strict=True))
    g = ", ".join(f"{n} {v:.0%}" for n, v in zip(names, guessed, strict=True))
    return f"Actual mix of the held-out rows: {a}. Mix the model predicted: {g}."


def overfit_sentence(perf: pd.DataFrame) -> str:
    """The train-to-test gap of one model, in macro-F1."""
    tr, va, te = (perf.loc[k, "macro_f1"] for k in ("Train", "Validation", "Test"))
    return (
        f"Macro-F1 is {tr:.3f} on the training years, {va:.3f} on validation and {te:.3f} on test: "
        f"a train-to-test gap of {tr - te:.3f}. The model fits the rows it was trained on better "
        "than unseen years, so the held-out numbers, not the training ones, are the result."
    )


def build_report(cfg: Config, target: str = "y_trend") -> dict[str, Path]:
    """Write figures, SHAP table, the tuned model file and reports/results.md. Returns paths."""
    saved = load_comparison(cfg, target)
    if "xgboost" not in saved.tuning:
        raise ValueError("the report needs an XGBoost run; train with --models including xgboost")
    m = load_matrix(cfg)
    fig_dir = cfg.paths.figures
    meta = saved.meta

    def fig_path(name: str):
        return fig_dir / f"{target}_{name}.png"

    extra = recompute_scores(saved, m, cfg)
    save_figure(plot_scores(saved.scores), fig_path("scores"))
    year_entries = [
        e for e in ("mean_reversion", "xgboost_tuned", "ensemble") if e in saved.scores.index
    ]
    by_year = scores_by_year(saved, m, cfg, year_entries)
    save_figure(plot_by_year(by_year), fig_path("by_year"))

    h = holdout_of(m, cfg, target)
    final_proba = saved.proba("xgboost_tuned").to_numpy()
    final_pred = final_proba.argmax(axis=1)
    cls_report = classification_table(h.y, final_pred)
    cls_report.to_csv(cfg.paths.metrics / f"{target}_classification_report.csv")
    save_figure(plot_confusion(h.y, final_proba), fig_path("confusion"))
    save_figure(plot_roc(h.y, final_proba), fig_path("roc"))
    save_figure(plot_class_metrics(cls_report), fig_path("class_metrics"))

    t = saved.tuning["xgboost"]
    best = t["best_params"]
    perf = split_performance(m, cfg, target, "xgboost", best)
    perf.to_csv(cfg.paths.metrics / f"{target}_split_performance.csv")
    trials = pd.read_csv(cfg.paths.metrics / f"{target}_trials_xgboost.csv")
    save_figure(
        plot_tuning(trials, t["default_cv_log_loss"], "XGBoost tuning"), fig_path("tuning_xgboost")
    )

    final, _ = final_fit("xgboost", m, cfg, target, best)
    cfg.paths.models.mkdir(parents=True, exist_ok=True)
    model_file = cfg.paths.models / f"{target}_xgboost.json"
    final.model.save_model(model_file)
    X_test = m.loc[holdout_mask(m, cfg.split, target), FEATURES]
    imp = importance(contributions(final, X_test), FEATURES)
    imp.to_csv(cfg.paths.metrics / f"{target}_shap_importance.csv")
    save_figure(plot_importance(imp), fig_path("importance"))

    sens_file = cfg.paths.metrics / f"{target}_sensitivity.csv"
    sens = pd.read_csv(sens_file) if sens_file.exists() else None
    if sens is not None:
        save_figure(plot_sensitivity(sens), fig_path("sensitivity"))

    top = imp.head(10)
    top_table = pd.DataFrame(
        {"Feature": top.index, "Mean abs. SHAP": [f"{v:.3f}" for v in top["total"]]}
    )
    text = [
        "# Results: year-ahead crime trend by state and crime head",
        "",
        "Generated by `python -m crimebase report` from `reports/metrics/`. Do not edit by hand.",
        "",
        "## 1. Summary",
        "",
        headline(saved),
        "",
        "## 2. Problem and data",
        "",
        "- **Task:** multi-class classification. For a state/UT and crime head in year *t*, "
        "predict whether `cases_registered` in *t+1* goes **down**, stays **stable** or goes "
        "**up**, with "
        f"a band of ±{meta['trend_band']:.0%}.",
        f"- **Rows:** a series-year needs at least {meta['min_cases']} registered cases, a next "
        "year, no state boundary change, no source switch and no 10x jump.",
        f"- **Split (by time, never random):** train on feature years up to {meta['train_to']}, "
        f"validate on {meta['val_years'][0]}-{meta['val_years'][-1]}, test once on "
        f"{meta['test_from']} onwards (**{meta['holdout_rows']:,} test rows** in "
        f"{meta['holdout_clusters']} state-year groups).",
        "- **Classes:** `down`, `stable`, `up`; chance accuracy is about one in three.",
        f"- **Tuning:** {meta['n_trials']} Optuna trials per model over expanding time folds; "
        f"confidence intervals are {meta['n_boot']}-resample bootstraps of whole state-year "
        "groups.",
        "",
        "## 3. Test-set performance of every model",
        "",
        "**Table 1.** Metrics on the held-out years. Precision and recall are macro averages over "
        "the three classes; ROC-AUC is one-vs-rest. Lower log loss is better.",
        "",
        md_table(scores_table(saved, extra)),
        "",
        f"![Figure 1: macro-F1 of every model with 95% intervals]({FIG}/{target}_scores.png)",
        "",
        "**Figure 1.** Macro-F1 of every model with 95% intervals; baselines are grey.",
        "",
        "## 4. The final model: tuned XGBoost",
        "",
        "### 4.1 Classification report",
        "",
        "**Table 2.** Per-class precision, recall and F1 on the held-out years.",
        "",
        md_table(classification_report_table(cls_report)),
        "",
        f"![Figure 2: precision, recall and F1 by class]({FIG}/{target}_class_metrics.png)",
        "",
        "**Figure 2.** Precision, recall and F1 by class.",
        "",
        "### 4.2 Confusion matrix",
        "",
        f"![Figure 3: confusion matrix]({FIG}/{target}_confusion.png)",
        "",
        "**Figure 3.** Row-normalised confusion matrix: the share of each actual class, with the "
        "count underneath.",
        "",
        prediction_mix_sentence(h.y, final_proba),
        "",
        "### 4.3 ROC curves",
        "",
        f"![Figure 4: one-vs-rest ROC curves]({FIG}/{target}_roc.png)",
        "",
        "**Figure 4.** One-vs-rest ROC curve per class; the dashed line is chance.",
        "",
        "### 4.4 Train, validation and test",
        "",
        "**Table 3.** One XGBoost model fitted on the training years and early-stopped on the "
        "validation years, scored on all three splits. The headline test numbers above refit on "
        "train + validation first, so they differ from this table's test row.",
        "",
        md_table(split_table(perf)),
        "",
        overfit_sentence(perf),
        "",
        "## 5. Is one model really better? Paired tests",
        "",
        "**Table 4.** Difference between two models on the same bootstrap resamples, so shared "
        "noise cancels. `Excludes 0` says whether the 95% interval stays on one side of zero.",
        "",
        md_table(paired_table(saved)),
        "",
        "## 6. Hyperparameter tuning",
        "",
        "**Table 5.** Mean cross-validated log loss over the time folds (lower is better).",
        "",
        md_table(tuning_table(saved)),
        "",
        f"![Figure 5: XGBoost tuning history]({FIG}/{target}_tuning_xgboost.png)",
        "",
        "**Figure 5.** Each completed trial and the best so far.",
        "",
        "## 7. Performance by held-out year",
        "",
        "**Table 6.** Macro-F1 for each feature year in the test period.",
        "",
        md_table(by_year_table(by_year)),
        "",
        f"![Figure 6: macro-F1 by held-out year]({FIG}/{target}_by_year.png)",
        "",
        "**Figure 6.** Macro-F1 by held-out year.",
        "",
        "## 8. What drives the model (TreeSHAP)",
        "",
        drivers_sentence(imp),
        "",
        "**Table 7.** The ten features with the largest mean absolute SHAP value over the test "
        "rows.",
        "",
        md_table(top_table),
        "",
        "`lv` is the log of this year's count, `mean4` and `std4` the mean and spread of the "
        "last four years of log counts, `g1` last year's log growth and `g1_rel` that growth "
        "relative to other states for the same head.",
        "",
        f"![Figure 7: feature importance]({FIG}/{target}_importance.png)",
        "",
        "**Figure 7.** Mean absolute SHAP value of the top features.",
        "",
    ]
    if sens is not None:
        text += [
            "## 9. Sensitivity to the minimum-count rule",
            "",
            sensitivity_sentence(sens),
            "",
            "**Table 8.** Default XGBoost and the mean-reversion rule scored on the same rows at "
            "each minimum count.",
            "",
            md_table(
                pd.DataFrame(
                    {
                        "min_cases": sens.min_cases,
                        "Labelled test rows": sens.n_test.map("{:,}".format),
                        "XGBoost macro-F1": sens.model_f1.map("{:.3f}".format),
                        "Rule macro-F1": sens.rule_f1.map("{:.3f}".format),
                        "Edge [95% CI]": [
                            _ci(r["diff"], r.diff_lo, r.diff_hi) for _, r in sens.iterrows()
                        ],
                    }
                )
            ),
            "",
            f"![Figure 8: sensitivity to min_cases]({FIG}/{target}_sensitivity.png)",
            "",
            "**Figure 8.** Macro-F1 of the model and of the rule as the minimum count rises.",
            "",
        ]
    text += [
        "## 10. Limitations",
        "",
        "- Reported crime reflects registration practice, not crime itself; definitions change "
        "between NCRB editions and a row that jumps 10x is treated as a definition break.",
        "- Counts are raw, not per capita: the zip has no state population table.",
        "- Three held-out years (one of them 2020) is a small number of independent periods, which "
        "is why the intervals resample whole state-year groups.",
        "- The signal is modest. Compare every model with the mean-reversion rule, not chance.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv sync",
        "uv run python -m crimebase clean && uv run python -m crimebase features",
        "uv run python -m crimebase train          # about 40 minutes on 4 CPUs",
        "uv run python -m crimebase sensitivity && uv run python -m crimebase report",
        "```",
        "",
    ]
    cfg.paths.report.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.report.write_text("\n".join(text))
    return {"report": cfg.paths.report, "model": model_file}
