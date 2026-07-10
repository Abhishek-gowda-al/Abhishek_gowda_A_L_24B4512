"""
backtest.py — Ties data -> features -> HMM -> optimizer into a single
walk-forward-validated, transaction-cost-aware backtest.

The one rule every function here obeys:
    A decision made "as of" day t may only use asset/feature data with
    index <= t-1 (i.e., data known by the start of day t).
"""

import numpy as np
import pandas as pd

from src.features import HMM_FEATURE_COLS, fit_zscore_params, apply_zscore
from src.regime import fit_hmm, predict_regimes
from src.optimize import optimize_weights, estimate_mu_sigma
from src.walkforward import expanding_walk_forward_splits, auto_walk_forward_config


def run_walk_forward_regimes(feature_df: pd.DataFrame, n_states=3, n_splits=None,
                              min_train_size=None, test_size=None):
    """
    Walk-forward regime detection.

    If n_splits/min_train_size/test_size are left as None (the default),
    they're auto-derived from the actual length of feature_df via
    `auto_walk_forward_config` — this is what lets the same notebook run
    correctly on a dataset with a different date range/length (e.g. an
    evaluator's "unseen" holdout period) without hand-tuning fold sizes
    for that specific dataset.

    For every fold:
      1. z-score params fit on TRAIN ONLY, applied to both train and test
      2. HMM fit on TRAIN ONLY
      3. states decoded on TEST ONLY, using that fold's own model+scaler
      4. state->label map built from TRAIN state/volatility relationship,
         reused (not refit) on the test fold, so labels are meaningful
         and not just relabeled per-fold noise

    Returns a single Series of regime labels covering every day that fell
    into some fold's test window (i.e. everything after the first
    min_train_size warm-up).
    """
    feat = feature_df.dropna(subset=HMM_FEATURE_COLS + ["vol_21d"]).copy()
    n = len(feat)

    if n_splits is None or min_train_size is None or test_size is None:
        auto_splits, auto_train, auto_test = auto_walk_forward_config(n)
        n_splits = n_splits or auto_splits
        min_train_size = min_train_size or auto_train
        test_size = test_size or auto_test
        print(f"[walk-forward] auto-derived config from {n} rows: "
              f"n_splits={n_splits}, min_train_size={min_train_size}, test_size={test_size}")

    splits = expanding_walk_forward_splits(n, n_splits=n_splits,
                                            min_train_size=min_train_size,
                                            test_size=test_size)
    if not splits:
        raise ValueError("Not enough data for the requested walk-forward configuration.")

    all_labels = pd.Series(index=feat.index, dtype=object)
    all_transition_mats = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        train_df = feat.iloc[train_idx]
        test_df = feat.iloc[test_idx]

        mu, sigma = fit_zscore_params(train_df[HMM_FEATURE_COLS])
        train_scaled = apply_zscore(train_df[HMM_FEATURE_COLS], mu, sigma).values
        test_scaled = apply_zscore(test_df[HMM_FEATURE_COLS], mu, sigma).values

        model = fit_hmm(train_scaled, n_states=n_states)

        # Build the state->label map from TRAIN decoding (train vol is
        # what defines "what does high vol look like" for this fold).
        train_states = model.predict(train_scaled)
        from src.regime import label_states_by_volatility
        label_map = label_states_by_volatility(train_states, train_df["vol_21d"])

        test_states = model.predict(test_scaled)
        test_labels = pd.Series(test_states, index=test_df.index).map(label_map)

        all_labels.loc[test_df.index] = test_labels
        all_transition_mats.append(model.transmat_)

        print(f"[walk-forward] fold {fold_i+1}/{len(splits)}: "
              f"train={train_df.index[0].date()}..{train_df.index[-1].date()} "
              f"test={test_df.index[0].date()}..{test_df.index[-1].date()} "
              f"test regime counts={test_labels.value_counts().to_dict()}")

    return all_labels.dropna(), all_transition_mats


def run_backtest(asset_returns: pd.DataFrame, regime_labels: pd.Series,
                  lookback=126, cost_bps=7.5, rebalance_every=21):
    """
    Walk forward day by day over the period covered by `regime_labels`,
    rebalancing (a) whenever the detected regime changes, or (b) at least
    every `rebalance_every` trading days regardless, using mu/sigma
    estimated from a trailing `lookback`-day window of REALIZED returns
    ending the day before the rebalance (strictly historical -> no
    leakage, even though this window spans back before the walk-forward
    test period; it's actual past data, not information from the future
    relative to the decision day).

    cost_bps: transaction cost in basis points, applied to one-way
    turnover on rebalance days.

    Returns
    -------
    result_df : DataFrame with columns
        ['gross_return', 'cost', 'net_return', 'regime'] plus one column
        per asset weight, indexed by date.
    """
    assets = asset_returns.columns.tolist()
    dates = regime_labels.index
    cost_frac = cost_bps / 10000.0

    weights = np.ones(len(assets)) / len(assets)
    weights_history = []
    gross_returns = []
    costs = []
    last_regime = None
    days_since_rebalance = 0

    for i, t in enumerate(dates):
        regime = regime_labels.loc[t]
        need_rebalance = (
            last_regime is None
            or regime != last_regime
            or days_since_rebalance >= rebalance_every
        )

        if need_rebalance:
            window = asset_returns.loc[:t].iloc[-(lookback + 1):-1]  # strictly before t
            if len(window) >= max(30, lookback // 4):
                mu, sigma = estimate_mu_sigma(window)
                new_weights = optimize_weights(mu, sigma, regime)
            else:
                new_weights = weights  # not enough history yet, hold

            turnover = np.abs(new_weights - weights).sum() / 2
            cost = turnover * cost_frac
            weights = new_weights
            days_since_rebalance = 0
        else:
            cost = 0.0
            days_since_rebalance += 1

        # Apply today's asset returns to the (start-of-day) weights.
        day_ret = asset_returns.loc[t].values
        port_gross_ret = float(np.dot(weights, day_ret))

        weights_history.append(weights.copy())
        gross_returns.append(port_gross_ret)
        costs.append(cost)
        last_regime = regime

        # Weights drift with returns until the next rebalance.
        drifted = weights * (1 + day_ret)
        weights = drifted / drifted.sum()

    weights_df = pd.DataFrame(weights_history, index=dates, columns=assets)
    result = pd.DataFrame({
        "gross_return": gross_returns,
        "cost": costs,
        "regime": regime_labels.values,
    }, index=dates)
    result["net_return"] = result["gross_return"] - result["cost"]
    result = pd.concat([result, weights_df.add_prefix("w_")], axis=1)
    return result


def static_benchmark_returns(asset_returns: pd.DataFrame, weights: dict, index=None):
    """Fixed-weight benchmark (e.g. 60/40 or equal-weight), rebalanced daily
    back to target for simplicity (no drift / no transaction costs modeled
    on the benchmark, which is standard practice: benchmarks are meant to
    be a frictionless reference point, not a competing strategy)."""
    w = pd.Series(weights)
    w = w[asset_returns.columns]  # align order
    rets = asset_returns.loc[index] if index is not None else asset_returns
    return rets.dot(w.values)
