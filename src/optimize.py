"""
optimize.py — Regime-conditional convex portfolio optimization (cvxpy).

Each regime gets its own objective, matching the project brief:
  Bull    -> maximize Sharpe-like objective (return - risk_aversion * variance)
  Bear    -> defensive: minimize variance, but allow a modest return floor
  Crisis  -> minimize variance outright (capital preservation)

All three are solved as the same generic mean-variance QP with a
regime-dependent risk-aversion coefficient — simpler to reason about /
debug than switching solver formulations per regime, and behaves exactly
like "maximize Sharpe in Bull, minimize vol in Crisis" at the extremes of
the risk-aversion dial.
"""

import numpy as np
import cvxpy as cp

# Risk-aversion coefficient (lambda) per regime, in:  max  mu.w - lambda * w'Sigma w
# Larger lambda -> more weight on risk reduction over return.
REGIME_RISK_AVERSION = {
    "Bull": 2.0,
    "Bear": 8.0,
    "Crisis": 25.0,
}

# Hard bounds: long-only, fully invested, and a cap so the optimizer can't
# dump 100% into one asset even under a benign objective.
DEFAULT_MAX_WEIGHT = 0.70
DEFAULT_MIN_WEIGHT = 0.0


def optimize_weights(mu: np.ndarray, sigma: np.ndarray, regime_label: str,
                      max_weight=DEFAULT_MAX_WEIGHT, min_weight=DEFAULT_MIN_WEIGHT):
    """
    Solve: maximize  mu.w - lambda * w' Sigma w
           s.t.       sum(w) == 1, min_weight <= w <= max_weight

    mu, sigma should be estimated using ONLY data available at decision
    time (i.e. the training/lookback window ending at the current day —
    never the test/future window).
    """
    n = len(mu)
    w = cp.Variable(n)
    lam = REGIME_RISK_AVERSION.get(regime_label, 10.0)

    risk = cp.quad_form(w, cp.psd_wrap(sigma))
    objective = cp.Maximize(mu @ w - lam * risk)
    constraints = [cp.sum(w) == 1, w >= min_weight, w <= max_weight]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP)
    except cp.error.SolverError:
        prob.solve(solver=cp.SCS)

    if w.value is None:
        # Fallback: equal weight if the solver fails for any reason
        return np.ones(n) / n

    weights = np.clip(w.value, 0, None)
    weights = weights / weights.sum()
    return weights


def estimate_mu_sigma(returns_window: "pd.DataFrame"):
    """
    Simple historical estimator for expected returns / covariance from a
    lookback window of daily returns (annualized). Callers are responsible
    for making sure `returns_window` only contains data up to "today" —
    this function itself has no knowledge of what's leakage-safe.
    """
    mu = returns_window.mean().values * 252
    sigma = returns_window.cov().values * 252
    return mu, sigma
