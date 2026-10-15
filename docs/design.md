# Design: CrimeBase classification pipeline

## 1. Problem and what the data allows

The dataset (`data/raw/india_crime_data.zip`) is NCRB's published crime statistics for India in long
format, not incident records. The tables that matter:

| File in zip | Grain | Rows | Use |
|---|---|---|---|
| `crime_state_year.csv` | state/UT x year x scope x crime head x metric | 668k | **modelling table** |
| `crime_india_year.csv` | all-India x year x scope x crime head x metric | 22k | cross-check of the state table |
| `news.parquet` | one crime news article | 1.5k | context only (see below) |
| `facts*.parquet`, `documents.csv` | raw parsed cells and source catalogue | 8.8M / 6.6k | provenance, not used |

Profiling found that most "obvious" classification targets are unusable:

* **Crime category of a head or headline** is assigned by a keyword rule (`labeller == rules` for every
  news row). A model would only re-learn the rule. Titles are also null for exactly the sexual-offence
  (149 articles) and child-offence (3) classes, so missingness alone would leak the label.
* **Incident-level classification** is impossible: only news has incident rows, and it is a small, rule-labelled sample.

**Chosen task: year-ahead trend of registered cases.** For a state/UT and crime head in year *t*,
predict how `cases_registered` moves in *t+1*: `down` (< -10%), `stable` (within +-10%) or `up` (> +10%).
It is non-circular (the label comes from future counts, not from a rule), multi-class, time-ordered and
hard enough to need real validation. The matrix also carries two binary labels, `y_drop` (< -25%) and
`y_surge` (> +25%), built from the same features; no model is trained on them yet (an early-warning
framing is the obvious next step).

**What it turned out to support** (measured, see `reports/results.md`): tuned XGBoost scores macro-F1 0.409
on the 2019+ holdout against 0.387 for a one-line mean-reversion rule, a paired difference of +0.022 (95% CI
+0.003 to +0.042) with AUC about 0.61. None of the alternatives beats it in a paired test: LightGBM and a
soft-voting ensemble are indistinguishable from it, CatBoost and the tuned logistic regression are slightly
worse, and tuning did not improve the held-out score. Reported crime counts
are noisy, so every model is compared with the rule and with confidence intervals, not with chance.

## 2. Requirements

Functional
* One command per stage; notebooks call the same library code, so a number in a notebook can be reproduced from the CLI.
* Read the raw zip in place; never modify `data/raw/`.
* Compare XGBoost against LightGBM, CatBoost, a linear model, simple rules and an ensemble.

Non-functional
* Reproducible: `uv sync` then one command per stage; fixed seeds; locked dependencies.
* No leakage: features use only years <= *t*; evaluation is time-ordered.
* Runs on 4 CPUs / 15 GB RAM with no GPU: `clean` and `features` take seconds, the full tuned comparison about 40 minutes.

Constraints: a Git repo with a 100 MB file limit (the zip is 52 MB, the extracted CSV is 113 MB, so data stays zipped).

## 3. Layout

```
configs/crimebase.yaml        all paths, thresholds, split years, seeds (the only place numbers live)
data/raw/                     immutable input zip (tracked)
data/interim/                 cleaned panel (git-ignored, rebuilt)
data/processed/               model matrix with labels (git-ignored, rebuilt)
src/crimebase/                importable package, one module per concern:
                                data, clean, features, splits  (inputs to a model matrix)
                                models, tuning, evaluate, experiment, results  (fit, score, persist)
                                explain, viz, report  (interpretation and output)  pipeline, cli, config
notebooks/                    01 understanding, 02 cleaning + dataset comparison, 03 XGBoost, 04 model comparison
models/                       fitted model files, written by the report stage (git-ignored)
reports/metrics/              metric JSON/CSV written by the pipeline (tracked)
reports/figures/              PNGs used by the report (tracked)
reports/results.md            generated summary, never hand-edited
tests/                        unit tests on a small synthetic panel
docs/design.md                this file
```

Rules: code lives in `src/`, notebooks only orchestrate and explain; every output directory has exactly one writer stage.

## 4. Pipeline and contracts

```
raw zip --ingest--> state_year table --clean--> panel.parquet --features--> model_matrix.parquet
                                                                                 |
              reports/results.md <--report-- reports/metrics <--train, sensitivity--+
```

| Stage | Input | Output | Contract |
|---|---|---|---|
| ingest | zip + config | DataFrame | columns match the data dictionary; failures raise, never fill |
| clean | state-year table | `data/interim/panel.parquet` | unique key `(state_code, scope, head_key, year)`; metrics become columns |
| features | panel | `data/processed/model_matrix.parquet` | one row per key; features from years <= t; labels from t+1; `eligible` + `reason` columns, nothing silently dropped |
| train | matrix + config | `reports/metrics/*` | fit on train only; tune with time-ordered CV; test years touched once |
| sensitivity | panel + config | `reports/metrics/*_sensitivity.csv` | rebuilds the matrix at each `min_cases`; same rows for model and rule |
| report | metrics + matrix | `reports/results.md`, `reports/figures/*`, `models/*` | every number computed from the metric files (plus one SHAP refit); nothing typed by hand |

### Cleaning rules (each backed by a measurement and a unit test)

1. **Normalise head text** to `head_key` (lower-case, strip punctuation): 2,490 raw spellings become 2,144.
2. **De-duplicate by max, never sum.** 97.5% of duplicate groups hold identical values; in the 141 that differ
   the larger value is never flagged `disputed` while the smaller one is in 68% of them (for example 695 vs 1),
   so the larger is the consensus value. Summing would double-count.
3. **Never aggregate across `scope`.** The scopes overlap (the data dictionary says so).
4. **Pivot** `cases_registered`, `persons_arrested`, `cases_chargesheeted`, `persons_convicted`, `victims` to columns for the same key.
5. **Eligibility for a label** (flagged, not deleted): next year present; `cases_registered >= min_cases`
   (Poisson noise is 1/sqrt(n), so below 100 cases it is already as large as the 10% band; the value is a
   config setting and notebook 04 reports the sensitivity); not a police complaint-handling table (10 heads,
   486 rows, 2013-2015, counts up to 8.9 million); growth ratio within [0.1, 10]
   (2.65% of pairs break this and the 99.9th percentile is 266x, which is a changed definition, not crime);
   no state boundary change (AP 2013->2014: the IPC total falls 218k -> 115k and the median head ratio is
   0.53 when Telangana is carved out; JK shows no break around Ladakh); and no `source` switch between the two years.

### Features (all computed from years <= t)

Log level, 1-3 year log growth, 4-year mean/std/deviation and z-score, deviation from the expanding mean,
history length; cross-state context for the same head and year (median growth, own growth relative to it,
level relative to median, share of national count, number of reporting states); arrest, chargesheet,
conviction and victim ratios; head flags (total / other / child) and the `disputed` flag; categoricals `scope`, `crime_category`, `state_code`, `source`.

## 5. Evaluation protocol

* **Split by time**, never at random: train t <= 2016, validation 2017-2018, test t >= 2019 (targets 2020-2022, which includes the COVID shift).
* **Tuning** uses expanding-window CV over earlier years; the test years are not used for any choice.
* **Primary metric** macro-F1 (the stable class is the smallest); also balanced accuracy, accuracy, log loss, OvR ROC-AUC, per-class F1, per-test-year scores.
* **Baselines**: majority class, persistence, mean reversion, a one-feature table learned from last year's direction, and logistic regression.
* **Uncertainty**: 95% percentile bootstrap that resamples whole state-year groups (rows of one state and year share shocks, so row-level resampling would be too optimistic), plus paired bootstraps on the same resamples for model-versus-model differences.

## 6. Model strategy

XGBoost first (hist trees, native categoricals, early stopping, Optuna tuning). LightGBM and CatBoost are
trained on the same folds, plus a soft-voting ensemble of the three. "Better than XGBoost" is decided by
the test table and its paired intervals, not assumed. Attributions come from XGBoost's built-in TreeSHAP
(`pred_contribs`), which handles categorical features natively and is checked to add up to the raw margins.

## 7. Trade-offs and risks

| Decision | Alternative | Why this one | Revisit when |
|---|---|---|---|
| Panel forecasting target | news headline category | label is a keyword rule, titles withheld for two classes | an LLM-labelled news subset (`labeller == llm`) exists |
| Keep data zipped, read in place | extract CSVs into the repo | the CSV exceeds GitHub's 100 MB file limit | the data moves to LFS or object storage |
| Flag ineligible rows | delete them | the cleaning notebook can quantify each rule | never |
| Raw counts, no per-capita rate | join population | no state population table in the zip | a census table is added |
| Single holdout of 3 years | rolling-origin test | test is touched once; rolling would use more years | more years of data arrive |

Risks: definition changes between NCRB editions add label noise that no model can remove; 2016 is sparsely
covered; the COVID year 2020 shifts the base rates; reported crime reflects registration practice, not actual crime.
