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
