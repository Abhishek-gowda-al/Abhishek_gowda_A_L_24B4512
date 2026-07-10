"""
metrics.py — Standard backtest performance metrics.
"""

import numpy as np
import pandas as pd


def annualized_return(daily_returns: pd.Series) -> float:
    equity = (1 + daily_returns).cumprod()
    n_years = len(daily_returns) / 252
    return equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan


def annualized_vol(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(252)


def sharpe_ratio(daily_returns: pd.Series, rf=0.0) -> float:
    excess = daily_returns - rf / 252
    vol = excess.std()
    return (excess.mean() / vol) * np.sqrt(252) if vol > 0 else np.nan


def sortino_ratio(daily_returns: pd.Series, rf=0.0) -> float:
    excess = daily_returns - rf / 252
    downside = excess[excess < 0]
    dd = downside.std()
    return (excess.mean() / dd) * np.sqrt(252) if dd and dd > 0 else np.nan


def max_drawdown(daily_returns: pd.Series) -> float:
    equity = (1 + daily_returns).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return dd.min()


def calmar_ratio(daily_returns: pd.Series) -> float:
    mdd = max_drawdown(daily_returns)
    ann_ret = annualized_return(daily_returns)
    return ann_ret / abs(mdd) if mdd != 0 else np.nan


def turnover(weights_history: pd.DataFrame) -> float:
    """Average daily one-way turnover (sum of |weight changes| / 2)."""
    diffs = weights_history.diff().abs().sum(axis=1) / 2
    return diffs.mean()


def performance_summary(daily_returns: pd.Series, weights_history: pd.DataFrame = None, name="Strategy"):
    row = {
        "Strategy": name,
        "AnnReturn": annualized_return(daily_returns),
        "AnnVol": annualized_vol(daily_returns),
        "Sharpe": sharpe_ratio(daily_returns),
        "Sortino": sortino_ratio(daily_returns),
        "MaxDrawdown": max_drawdown(daily_returns),
        "Calmar": calmar_ratio(daily_returns),
    }
    if weights_history is not None:
        row["AvgDailyTurnover"] = turnover(weights_history)
    return row
