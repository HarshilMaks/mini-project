from dataclasses import replace

import numpy as np
import pytest

from crimebase.config import load_config
from crimebase.features import build_matrix
from crimebase.splits import cv_folds, dev_mask, holdout_mask, labelled, train_mask, val_mask


@pytest.fixture
def matrix(panel):
    return build_matrix(panel, load_config().task)


@pytest.fixture
def split():
    # synthetic panel spans 2005-2020, so shift the real windows to fit it
    base = load_config().split
    return replace(
        base,
        train_to=2013,
        val_years=(2014, 2015),
        test_from=2016,
        cv_windows=((2010, 2011), (2012, 2013), (2014, 2015)),
    )


def test_train_val_test_are_disjoint_and_ordered(matrix, split):
    t = "y_trend"
    tr, va, te = (f(matrix, split, t) for f in (train_mask, val_mask, holdout_mask))
    assert tr.any() and va.any() and te.any()
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert (
        matrix.year[tr].max()
        < matrix.year[va].min()
        <= matrix.year[va].max()
        < matrix.year[te].min()
    )


def test_dev_is_train_plus_val_and_never_touches_test_years(matrix, split):
    t = "y_trend"
    dev = dev_mask(matrix, split, t)
    assert (dev == (train_mask(matrix, split, t) | val_mask(matrix, split, t))).all()
    assert matrix.year[dev].max() < split.test_from


def test_only_labelled_eligible_rows_are_ever_selected(matrix, split):
    t = "y_surge"
    assert labelled(matrix, t).sum() == matrix.eligible.sum()
    for f in (train_mask, val_mask, holdout_mask, dev_mask):
        assert matrix.eligible[f(matrix, split, t)].all()


def test_cv_folds_train_strictly_before_validation(matrix, split):
    folds = cv_folds(matrix, split, "y_trend")
    assert len(folds) == 3
    years = matrix.year.to_numpy()
    for (train, val), (first, last) in zip(folds, split.cv_windows, strict=True):
        assert years[train].max() < first
        assert first <= years[val].min() and years[val].max() <= last
        assert not set(train) & set(val)


def test_cv_folds_expand_and_never_reach_test_years(matrix, split):
    folds = cv_folds(matrix, split, "y_trend")
    sizes = [len(tr) for tr, _ in folds]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]
    used = np.concatenate([np.concatenate(f) for f in folds])
    assert matrix.year.to_numpy()[used].max() < split.test_from


def test_cv_window_into_test_years_is_refused(matrix, split):
    bad = replace(split, cv_windows=((2014, 2016),))
    with pytest.raises(ValueError, match="test years"):
        cv_folds(matrix, bad, "y_trend")


def test_real_config_windows_stay_before_the_test_years(matrix):
    real = load_config().split
    assert all(last < real.test_from for _, last in real.cv_windows)
