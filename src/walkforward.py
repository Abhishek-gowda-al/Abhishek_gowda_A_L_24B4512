"""
walkforward.py — Expanding-window walk-forward split generator.

Same idea as the "Try it" exercise in the guide notebook, promoted to a
reusable utility: each fold's training window grows (expanding), the test
window slides forward, and folds never overlap in time in a way that lets
a test day influence its own fold's training data.
"""

import numpy as np


def expanding_walk_forward_splits(n_obs, n_splits=8, min_train_size=756, test_size=63):
    """
    Yields (train_idx, test_idx) integer-position arrays.

    n_obs:          total number of rows in the feature matrix
    n_splits:       number of folds to produce
    min_train_size: size of the very first training window (default ~3y of trading days)
    test_size:      size of each test window (default ~1 quarter)
    """
    splits = []
    start_test = min_train_size
    for i in range(n_splits):
        end_test = start_test + test_size
        if end_test > n_obs:
            break
        train_idx = np.arange(0, start_test)
        test_idx = np.arange(start_test, end_test)
        splits.append((train_idx, test_idx))
        start_test = end_test
    return splits


def auto_walk_forward_config(n_obs, target_folds=8, min_train_frac=0.35,
                              test_frac=0.08, min_train_size_floor=252,
                              min_test_size_floor=21):
    """
    Derive (n_splits, min_train_size, test_size) from the actual length of
    the dataset, instead of hard-coding numbers tuned to one specific date
    range. This is what keeps the walk-forward setup from silently
    breaking (or silently overfitting its own split sizes) when pointed at
    a shorter or longer "unseen" dataset than the one used to develop it.

    - min_train_size: at least min_train_frac of the data, but never less
      than ~1 trading year (min_train_size_floor), so the first HMM fit
      always has a reasonable amount of history.
    - test_size: a fraction of the data, but never less than ~1 trading
      month (min_test_size_floor), so folds aren't so tiny that regime
      counts per fold become meaningless.
    - n_splits: as many folds of that size as actually fit in the
      remaining data, capped at target_folds.
    """
    min_train_size = max(min_train_size_floor, int(n_obs * min_train_frac))
    test_size = max(min_test_size_floor, int(n_obs * test_frac))

    remaining = n_obs - min_train_size
    max_possible_splits = max(0, remaining // test_size)
    n_splits = int(min(target_folds, max_possible_splits))

    if n_splits < 1:
        raise ValueError(
            f"Not enough data ({n_obs} rows) for walk-forward validation with "
            f"min_train_size={min_train_size}, test_size={test_size}. "
            f"Need at least {min_train_size + test_size} rows — either supply "
            f"more history or lower min_train_frac/test_frac."
        )
    return n_splits, min_train_size, test_size
