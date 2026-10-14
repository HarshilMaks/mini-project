from crimebase.config import ROOT, load_config


def test_paths_resolve_against_repo_root():
    cfg = load_config()
    assert cfg.paths.raw_zip == ROOT / "data" / "raw" / "india_crime_data.zip"
    assert cfg.paths.raw_zip.exists()


def test_values_are_typed():
    cfg = load_config()
    assert cfg.task.min_cases == 50
    assert cfg.task.boundary_breaks == (("AP", 2013),)
    assert cfg.split.cv_windows[-1] == (2017, 2018)
    assert cfg.split.val_years == (2017, 2018)


def test_split_years_do_not_overlap():
    s = load_config().split
    assert s.train_to < min(s.val_years)
    assert max(s.val_years) < s.test_from
    assert all(b < s.test_from for _, b in s.cv_windows)
