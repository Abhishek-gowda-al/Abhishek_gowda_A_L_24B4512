"""
data.py — Data ingestion for the RegimeShift project.

Pulls daily prices for the NSE equity leg, a gold ETF, a bond/cash-duration
leg, and India VIX. Falls back to a synthetic-but-realistic dataset if
yfinance/network access isn't available, so the rest of the pipeline can
always be developed and unit-tested offline.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ticker choices (NSE, via Yahoo Finance)
#   Equity : NIFTYBEES.NS  -> Nifty 50 index ETF
#   Gold   : GOLDBEES.NS   -> Gold ETF
#   Bond   : LIQUIDBEES.NS -> money-market / near-cash ETF, used as the
#            low-volatility "ballast" leg. If you have access to a longer
#            duration G-Sec ETF (e.g. a 10yr gilt ETF) with a long enough
#            history on Yahoo Finance, swap it in — it will behave more
#            like a "real" bond (rallies when equities crash) than a
#            money-market fund does.
#   Vol    : ^INDIAVIX     -> India VIX
# ---------------------------------------------------------------------------
DEFAULT_TICKERS = {
    "equity": "NIFTYBEES.NS",
    "bond": "LIQUIDBEES.NS",
    "gold": "GOLDBEES.NS",
}
VIX_TICKER = "^INDIAVIX"


def download_price_data(start="2015-01-01", end=None, tickers=None, vix_ticker=VIX_TICKER):
    """
    Download daily adjusted close prices for the asset legs + VIX.

    Returns
    -------
    prices : DataFrame indexed by date, columns = asset names (equity/bond/gold)
    vix    : Series indexed by date, India VIX level
    """
    import yfinance as yf

    tickers = tickers or DEFAULT_TICKERS
    all_symbols = list(tickers.values()) + [vix_ticker]

    raw = yf.download(all_symbols, start=start, end=end, progress=False, auto_adjust=True)
    close = raw["Close"].copy()
    close.columns.name = None

    inv_map = {v: k for k, v in tickers.items()}
    prices = close[[t for t in tickers.values()]].rename(columns=inv_map)
    vix = close[vix_ticker].rename("VIX")

    prices = prices.dropna(how="all")
    vix = vix.dropna()

    # Sanity check on the bond/ballast leg: a near-cash instrument like
    # LIQUIDBEES.NS can show near-zero price volatility if yfinance isn't
    # capturing its daily dividend accrual correctly. This won't crash
    # anything, but a near-flat "bond" leg will make the optimizer treat
    # it as risk-free ballast rather than a real diversifier — worth a
    # loud warning rather than a silently wrong backtest.
    if "bond" in prices.columns:
        bond_daily_vol = prices["bond"].pct_change().std()
        if pd.notna(bond_daily_vol) and bond_daily_vol < 1e-4:
            print(f"[data] WARNING: bond leg ({tickers.get('bond')}) daily return "
                  f"std is only {bond_daily_vol:.6f} — this looks close to flat. "
                  f"If this is a cash-management ETF (e.g. LIQUIDBEES), its price "
                  f"series may not reflect real duration/rate risk. Check the "
                  f"downloaded series before trusting backtest results, or swap in "
                  f"a longer-duration bond ETF in DEFAULT_TICKERS.")

    return prices, vix


def synthetic_price_data(start="2015-01-01", end="2024-01-01", seed=7):
    """
    Generate a synthetic multi-asset dataset that mimics the statistical
    shape of NSE-equity / gold / bond-like data, INCLUDING a couple of
    injected crisis regimes (so an HMM has something real to find).

    Used only for offline development/testing of the pipeline when live
    market data can't be reached (e.g. no network access). Do not use this
    to report "results" — it exists purely to validate that the code runs
    and behaves sensibly end to end.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # Regime schedule: mostly bull, with two crisis windows and one bear window
    regime = np.zeros(n, dtype=int)  # 0 = bull, 1 = bear, 2 = crisis
    crisis_1 = slice(int(n * 0.28), int(n * 0.28) + 35)   # ~ "2020-style" shock
    bear_1 = slice(int(n * 0.55), int(n * 0.55) + 90)
    crisis_2 = slice(int(n * 0.80), int(n * 0.80) + 45)   # ~ "2022-style" shock
    regime[crisis_1] = 2
    regime[bear_1] = 1
    regime[crisis_2] = 2

    # Per-regime daily mean/vol for each asset (equity, bond, gold)
    params = {
        0: dict(mu=[0.0007, 0.00015, 0.0002], sig=[0.008, 0.0015, 0.006]),   # bull
        1: dict(mu=[-0.0004, 0.0002, 0.0003], sig=[0.014, 0.0018, 0.008]),   # bear
        2: dict(mu=[-0.0025, 0.0004, 0.0009], sig=[0.032, 0.0035, 0.014]),   # crisis
    }
    corr = np.array([
        [1.0, -0.25, 0.05],
        [-0.25, 1.0, 0.05],
        [0.05, 0.05, 1.0],
    ])

    rets = np.zeros((n, 3))
    for r in (0, 1, 2):
        idx = np.where(regime == r)[0]
        if len(idx) == 0:
            continue
        mu = np.array(params[r]["mu"])
        sig = np.array(params[r]["sig"])
        cov = np.outer(sig, sig) * corr
        rets[idx] = rng.multivariate_normal(mu, cov, size=len(idx))

    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    prices = pd.DataFrame(prices, index=dates, columns=["equity", "bond", "gold"])

    # Synthetic VIX-like series: rises with realized equity vol
    realized_vol = pd.Series(rets[:, 0], index=dates).rolling(10).std() * np.sqrt(252)
    vix = (realized_vol * 100).bfill() + rng.normal(0, 1.0, n)
    vix = vix.clip(lower=8)
    vix.name = "VIX"

    return prices, vix


def load_data(start="2015-01-01", end=None, use_synthetic=False, tickers=None):
    """
    Single entry point: tries live data first, falls back to synthetic data
    with a loud warning if that fails (e.g. no network / yfinance error).
    """
    if use_synthetic:
        print("[data] use_synthetic=True -> generating synthetic dataset.")
        return synthetic_price_data(start=start, end=end or "2024-01-01")

    try:
        prices, vix = download_price_data(start=start, end=end, tickers=tickers)
        if prices.empty or len(prices) < 500:
            raise ValueError("Downloaded data looks too small/empty.")
        print(f"[data] Loaded live data: {len(prices)} rows, {prices.index[0].date()} -> {prices.index[-1].date()}")
        return prices, vix
    except Exception as e:
        print(f"[data] Live download failed ({e!r}). Falling back to SYNTHETIC data. "
              f"Re-run with network access for real results.")
        return synthetic_price_data(start=start, end=end or "2024-01-01")
