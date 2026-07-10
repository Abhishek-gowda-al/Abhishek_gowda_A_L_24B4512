"""
features.py — Leakage-safe feature engineering.

Every function here only uses `.rolling()` (fixed backward-looking window)
or `.expanding()` (backward-looking, growing window). Nothing here ever
uses centered windows, `.shift(-k)` as a feature, or full-sample statistics.
That discipline is what keeps the eventual HMM inputs free of lookahead bias.
"""

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff()


def momentum_features(prices: pd.DataFrame, windows=(5, 21, 63, 126), asset="equity"):
    """Total return over each lookback window, for one reference asset."""
    out = pd.DataFrame(index=prices.index)
    for w in windows:
        out[f"mom_{w}d"] = prices[asset].pct_change(w)
    return out


def volatility_features(log_rets: pd.Series, windows=(5, 21, 63)):
    """Annualized rolling realized volatility of one return series."""
    out = pd.DataFrame(index=log_rets.index)
    for w in windows:
        out[f"vol_{w}d"] = log_rets.rolling(w).std() * np.sqrt(252)
    return out


def expanding_zscore(series: pd.Series, min_periods=63) -> pd.Series:
    """
    Leakage-safe z-score: at time t, uses only data up to and including t
    (expanding window). This is what you use OUTSIDE a walk-forward loop
    for quick exploration; INSIDE a walk-forward loop use
    `fit_zscore_params` / `apply_zscore` below instead, so the test fold
    strictly reuses only the training fold's statistics.
    """
    mu = series.expanding(min_periods=min_periods).mean()
    sigma = series.expanding(min_periods=min_periods).std()
    return (series - mu) / sigma


def fit_zscore_params(train_df: pd.DataFrame):
    """Compute mean/std on a TRAINING slice only."""
    return train_df.mean(), train_df.std()


def apply_zscore(df: pd.DataFrame, mu: pd.Series, sigma: pd.Series) -> pd.DataFrame:
    """Apply previously-fit train mu/sigma to any slice (train or test)."""
    return (df - mu) / sigma


def build_feature_matrix(prices: pd.DataFrame, vix: pd.Series, equity_col="equity"):
    """
    Assemble the full raw (pre-scaling) feature set that will be fed to the
    HMM: momentum + volatility of the equity leg, plus the VIX level itself
    (already a well-known forward-looking-free fear gauge — its value on
    day t only reflects information available on day t).

    Returns a single DataFrame, NaN rows (from window warm-up) not yet
    dropped, so callers can align/trim explicitly and avoid accidentally
    dropping rows inconsistently between train/test folds.
    """
    log_rets = compute_log_returns(prices)[equity_col].rename("log_ret")

    mom = momentum_features(prices, windows=(5, 21, 63, 126), asset=equity_col)
    vol = volatility_features(log_rets, windows=(5, 21, 63))

    feat = pd.concat([log_rets, mom, vol], axis=1)
    feat = feat.join(vix.rename("vix_level"), how="left")
    feat["vix_level"] = feat["vix_level"].ffill()

    return feat


# The subset of columns actually fed into the HMM (kept small and
# deliberately: momentum for direction, volatility + VIX for the
# "stress" dimension — this is the pair of ideas the HMM is meant to
# separate into hidden states).
HMM_FEATURE_COLS = ["mom_21d", "vol_21d", "vix_level"]
