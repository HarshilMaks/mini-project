import pytest
import yaml

from crimebase.cli import main
from crimebase.config import DEFAULT_CONFIG
from test_pipeline import cfg  # noqa: F401  (fixture: synthetic zip + tmp output dirs)


@pytest.fixture
def config_file(cfg, tmp_path):  # noqa: F811
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["paths"].update(
        raw_zip=str(cfg.paths.raw_zip),
        interim=str(cfg.paths.interim),
        processed=str(cfg.paths.processed),
        metrics=str(cfg.paths.metrics),
        figures=str(cfg.paths.figures),
        models=str(cfg.paths.models),
        report=str(cfg.paths.report),
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_cli_runs_stages_in_order(config_file, capsys):
    assert main(["--config", str(config_file), "clean"]) == 0
    assert main(["--config", str(config_file), "features"]) == 0
    out = capsys.readouterr().out
    assert "clean: wrote" in out and "panel.parquet" in out
    assert "features: wrote" in out and "model_matrix.parquet" in out


def test_cli_rejects_unknown_stage():
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_cli_train_writes_metrics(config_file, cfg, capsys):  # noqa: F811
    main(["--config", str(config_file), "clean"])
    main(["--config", str(config_file), "features"])
    code = main(
        ["--config", str(config_file), "train", "--trials", "2", "--boot", "30",
         "--models", "xgboost", "lightgbm"]
    )  # fmt: skip
    assert code == 0
    out = capsys.readouterr().out
    assert "xgboost: tuning 2 trials" in out and "train: wrote" in out
    written = {p.name for p in cfg.paths.metrics.iterdir()}
    assert {"y_trend_scores.csv", "y_trend_tuning.json", "y_trend_meta.json"} <= written


def test_cli_train_rejects_an_unknown_model():
    with pytest.raises(SystemExit):
        main(["train", "--models", "random_forest"])


def test_cli_sensitivity_then_report_write_the_deliverables(config_file, cfg, capsys):  # noqa: F811
    run = lambda *a: main(["--config", str(config_file), *a])  # noqa: E731
    run("clean")
    run("features")
    run("train", "--trials", "2", "--boot", "30", "--models", "xgboost", "lightgbm")
    run("sensitivity", "--thresholds", "50", "300", "--boot", "30")
    assert run("report") == 0
    out = capsys.readouterr().out
    assert "sensitivity: wrote" in out and "report: wrote" in out
    text = cfg.paths.report.read_text()
    assert text.startswith("# Results") and "Sensitivity to the minimum-count rule" in text
    assert (cfg.paths.models / "y_trend_xgboost.json").exists()
    figures = {p.name for p in cfg.paths.figures.iterdir()}
    assert {
        "y_trend_scores.png",
        "y_trend_by_year.png",
        "y_trend_importance.png",
        "y_trend_roc.png",
        "y_trend_class_metrics.png",
    } <= figures
    for heading in (
        "## 1. Summary",
        "**Table 1.**",
        "### 4.1 Classification report",
        "### 4.4 Train, validation and test",
        "**Figure 4.**",
    ):
        assert heading in text, heading
    assert {"y_trend_classification_report.csv", "y_trend_split_performance.csv"} <= {
        p.name for p in cfg.paths.metrics.iterdir()
    }
    assert (cfg.paths.metrics / "y_trend_sensitivity.csv").exists()


def test_cli_sensitivity_without_a_panel_names_the_missing_stage(config_file):
    with pytest.raises(FileNotFoundError, match="python -m crimebase clean"):
        main(["--config", str(config_file), "sensitivity"])
