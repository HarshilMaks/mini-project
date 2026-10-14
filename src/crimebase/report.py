"""Build reports/results.md and its figures from the saved metrics.

Every number in the report is computed here from files under reports/metrics (plus one SHAP
refit of the tuned XGBoost); none is typed by hand. Sentences that interpret a number are
conditional on it, so the text stays true if the numbers change.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crimebase.config import Config
from crimebase.experiment import final_fit, holdout_of
from crimebase.explain import contributions, importance
from crimebase.features import FEATURES
from crimebase.pipeline import load_matrix
from crimebase.results import SavedComparison, load_comparison, scores_by_year
from crimebase.splits import holdout_mask
from crimebase.viz import (
    BASELINES,
    label,
    plot_by_year,
    plot_confusion,
    plot_importance,
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


def _ci(value: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def scores_table(saved: SavedComparison) -> pd.DataFrame:
    s = saved.scores
    rows = []
    for entry, r in s.iterrows():
        has_auc = pd.notna(r["auc_ovr"])
        rows.append(
            {
                "Model": label(entry),
                "Macro-F1 [95% CI]": _ci(r["macro_f1"], r["f1_lo"], r["f1_hi"]),
                "Balanced acc.": f"{r['balanced_acc']:.3f}",
                "AUC [95% CI]": _ci(r["auc_ovr"], r["auc_lo"], r["auc_hi"]) if has_auc else "n/a",
                "Log loss": f"{r['log_loss']:.3f}" if pd.notna(r["log_loss"]) else "n/a",
                "Trees": f"{int(r['trees'])}" if pd.notna(r["trees"]) else "",
            }
        )
    return pd.DataFrame(rows)


def paired_table(saved: SavedComparison) -> pd.DataFrame:
    rows = []
    for r in saved.paired.itertuples():
        rows.append(
            {
                "A": label(r.a),
                "B": label(r.b),
                "Metric": "macro-F1" if r.metric == "macro_f1" else "AUC",
                "A - B [95% CI]": _ci(r.diff, r.lo, r.hi, 3),
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
            f"XGBoost beats that rule by {vs_rule['diff']:+.3f} macro-F1 (paired 95% CI "
            f"{vs_rule.lo:+.3f} to {vs_rule.hi:+.3f}), an interval that {_verdict(vs_rule)} zero."
        )
    lines.append(
        f"All {len(models)} model variants land between {low:.3f} and {high:.3f}; naming the "
        "top one on the test set would be optimistic, so no winner is picked from it."
    )
    tuning = _paired(saved, "xgboost_tuned", "xgboost_default")
    if tuning is not None:
        lines.append(
            f"Tuning moved held-out macro-F1 by {tuning['diff']:+.3f} "
            f"(paired CI {tuning.lo:+.3f} to {tuning.hi:+.3f}, which {_verdict(tuning)} zero)."
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

    save_figure(plot_scores(saved.scores), fig_path("scores"))
    year_entries = [
        e for e in ("mean_reversion", "xgboost_tuned", "ensemble") if e in saved.scores.index
    ]
    by_year = scores_by_year(saved, m, cfg, year_entries)
    save_figure(plot_by_year(by_year), fig_path("by_year"))

    h = holdout_of(m, cfg, target)
    save_figure(plot_confusion(h.y, saved.proba("xgboost_tuned").to_numpy()), fig_path("confusion"))
    if "xgboost" in saved.tuning:
        t = saved.tuning["xgboost"]
        trials = pd.read_csv(cfg.paths.metrics / f"{target}_trials_xgboost.csv")
        save_figure(
            plot_tuning(trials, t["default_cv_log_loss"], "XGBoost tuning"),
            fig_path("tuning_xgboost"),
        )

    final, _ = final_fit("xgboost", m, cfg, target, saved.tuning["xgboost"]["best_params"])
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
        {"Feature": top.index, "Mean |SHAP|": [f"{v:.3f}" for v in top["total"]]}
    )
    text = [
        "# Results: year-ahead crime trend by state and crime head",
        "",
        "Generated by `python -m crimebase report` from `reports/metrics/`. Do not edit by hand.",
        "",
        "## Headline",
        "",
        headline(saved),
        "",
        "## Setup",
        "",
        f"- Task: for a state/UT and crime head in year *t*, predict whether `cases_registered` in "
        f"*t+1* goes **down**, stays **stable** or goes **up**, with a band of "
        f"±{meta['trend_band']:.0%}.",
        f"- Rows: a series-year needs at least {meta['min_cases']} registered cases, a next year, "
        "no state boundary change, no source switch and no 10x jump.",
        f"- Time split: train on feature years up to {meta['train_to']}, validate on "
        f"{meta['val_years'][0]}-{meta['val_years'][-1]}, test once on {meta['test_from']} onwards "
        f"(**{meta['holdout_rows']:,} rows** in {meta['holdout_clusters']} state-year groups).",
        f"- Tuning: {meta['n_trials']} Optuna trials per model over expanding time folds; "
        f"intervals are {meta['n_boot']}-resample bootstraps of whole state-year groups.",
        "",
        "## All models on the held-out years",
        "",
        md_table(scores_table(saved)),
        "",
        f"![Models against baselines]({FIG}/{target}_scores.png)",
        "",
        "## Is a model really better? Paired comparisons",
        "",
        "Both sides are scored on the same bootstrap resamples, so shared noise cancels.",
        "",
        md_table(paired_table(saved)),
        "",
        "## Tuning",
        "",
        md_table(tuning_table(saved)),
        "",
        f"![XGBoost tuning history]({FIG}/{target}_tuning_xgboost.png)",
        "",
        "## By held-out year",
        "",
        md_table(by_year_table(by_year)),
        "",
        f"![Macro-F1 by year]({FIG}/{target}_by_year.png)",
        "",
        "## Where the tuned XGBoost is right and wrong",
        "",
        f"![Confusion matrix]({FIG}/{target}_confusion.png)",
        "",
        "## What drives it (TreeSHAP)",
        "",
        md_table(top_table),
        "",
        f"![Feature importance]({FIG}/{target}_importance.png)",
        "",
    ]
    if sens is not None:
        text += [
            "## Sensitivity to the minimum-count rule",
            "",
            sensitivity_sentence(sens),
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
            f"![Sensitivity]({FIG}/{target}_sensitivity.png)",
            "",
        ]
    text += [
        "## Limits",
        "",
        "- Reported crime reflects registration practice, not crime itself; definitions change "
        "between NCRB editions and a row that jumps 10x is treated as a definition break.",
        "- Counts are raw, not per capita: the zip has no state population table.",
        "- Three held-out years (one of them 2020) is a small number of independent periods, which "
        "is why the intervals resample whole state-year groups.",
        "- The signal is modest. Compare every model with the mean-reversion rule, not chance.",
        "",
    ]
    cfg.paths.report.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.report.write_text("\n".join(text))
    return {"report": cfg.paths.report, "model": model_file}
