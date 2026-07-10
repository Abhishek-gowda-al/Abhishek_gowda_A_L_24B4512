# RegimeShift — Macro-Aware Tactical Asset Allocation Engine

Summer of Quant advanced capstone. A system that detects the market's hidden
"regime" (Bull / Bear / Crisis) from price data using a Hidden Markov Model,
and reallocates a stocks/bonds/gold portfolio via convex optimization to
match the detected regime — validated with walk-forward testing so results
aren't inflated by lookahead bias.

## Repo structure

```
regimeshift-capstone/
├── README.md                     <- this file
├── requirements.txt
├── RegimeShift_Project.ipynb     <- MAIN DELIVERABLE: full pipeline, top to bottom
└── src/
    ├── data.py                   <- price/VIX ingestion (yfinance + synthetic fallback)
    ├── features.py                <- momentum / volatility / leakage-safe z-scoring
    ├── walkforward.py            <- expanding-window train/test split generator
    ├── regime.py                 <- HMM fit + Bull/Bear/Crisis state labeling
    ├── optimize.py                <- regime-conditional mean-variance QP (cvxpy)
    ├── backtest.py                <- walk-forward loop + transaction-cost-aware backtest
    └── metrics.py                 <- Sharpe / Sortino / MaxDrawdown / Calmar / turnover
```

## How to run

```bash
python -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt
jupyter notebook RegimeShift_Project.ipynb
```

Run the notebook top to bottom. `src/data.py::load_data()` pulls live NSE + gold +
VIX data via `yfinance`; if that fails (no network, rate limit, ticker changed),
it automatically falls back to a synthetic dataset with injected crisis periods
and prints a warning — this keeps the pipeline runnable for code review even
without market access, but **do not report synthetic-data numbers as results**.

## Reproducing results

Everything downstream of `load_data()` is deterministic given the same input
data: the HMM is seeded (`random_state=42` in `src/regime.py`), the walk-forward
split boundaries are a pure function of `n_splits`/`min_train_size`/`test_size`,
and the optimizer is a convex QP (unique global optimum, no randomness). The
only non-determinism is (a) whatever data Yahoo Finance returns on the day you
run it, since it's a live feed, and (b) minor solver floating-point differences
across `cvxpy`/`OSQP` versions.

## Key design decisions

**Assets.** `NIFTYBEES.NS` (Nifty 50 ETF) for the equity leg, `GOLDBEES.NS` for
gold, `LIQUIDBEES.NS` as the low-volatility "ballast" leg. Being upfront about a
limitation: `LIQUIDBEES` is a money-market/near-cash fund, not a duration bond —
it won't rally the way a long-dated gilt would during an equity selloff. If you
have Yahoo Finance history for a longer-duration Indian G-Sec ETF, swap it into
`DEFAULT_TICKERS` in `src/data.py`; the rest of the pipeline is asset-agnostic.

**Why 3 HMM states.** Maps directly onto the project's Bull/Bear/Crisis framing
and is small enough to stay statistically identifiable given the amount of daily
data available (more states risk overfitting to noise rather than finding
economically meaningful regimes — `hmmlearn`'s `n_components` is a modeling
choice, not something the model discovers on its own).

**HMM features.** Deliberately kept small: `mom_21d` (direction), `vol_21d` and
`vix_level` (stress). Feeding in every momentum/vol horizon at once would let the
model latch onto noise; one clean feature per "dimension" (direction vs. fear)
gave more stable, interpretable states in testing.

**State → label mapping.** HMM state indices (0/1/2) are arbitrary and can flip
between refits. Each fold's mapping is derived from *that fold's own training
data*: the state with the highest mean `vol_21d` in training is called Crisis,
lowest is Bull, middle is Bear — then that mapping (not a new one) is applied to
the test fold's decoded states. This avoids using test-set information to decide
what a label means.

**Walk-forward protocol.** Expanding-window: fold 1 trains on the first ~3
years and tests on the next ~6 months; fold 2 trains on everything up to that
point and tests on the next ~6 months; and so on (`src/walkforward.py`). Every
z-score and every HMM fit inside a fold uses training-fold statistics only —
see `src/backtest.py::run_walk_forward_regimes` for the exact sequencing. This
was chosen over k-fold CV because random folds would let future data train a
model tested on the past, which is exactly the leakage this project is meant to
avoid.

**Regime → portfolio objective.** All three regimes solve the same
mean-variance QP (`max μᵀw − λ·wᵀΣw`, long-only, ≤70% per asset), varying only
the risk-aversion coefficient λ (`src/optimize.py::REGIME_RISK_AVERSION`): low λ
in Bull behaves like a Sharpe-tilted portfolio, very high λ in Crisis behaves
like minimum-variance. One formulation, one solver path, easier to debug than
switching objective *shapes* per regime — and it reproduces "maximize Sharpe in
Bull / minimize vol in Crisis" at the extremes of the λ dial, as the brief asks.

μ/Σ for the optimizer are estimated from a strictly trailing 126-day window of
*realized* returns ending the day before the rebalance — historical, not future,
data, so no leakage even though the window can span back before the current
walk-forward test fold.

**Rebalance trigger & costs.** Rebalances when the detected regime changes, or
every 21 trading days regardless (so weight estimates don't go stale during a
long, stable regime). A 7.5bps cost is applied to one-way turnover on every
rebalance. Results are reported gross and net of costs specifically so the
turnover drag from frequent regime flips is visible rather than hidden.

**Benchmarks.** Static 60/40 (equity/bond) and equal-weight (1/3 each),
rebalanced back to target daily with zero cost — intentionally frictionless,
since a benchmark should be a clean reference point, not a competing strategy
that also has to fight transaction costs.

## Known limitations / next steps

- `LIQUIDBEES` as the bond proxy (see above) — the whole point of a "bond leg"
  is to zig when equities zag; a near-cash fund does that weakly at best.
  `src/data.py::load_data` now prints a warning if the bond leg's daily return
  volatility looks suspiciously close to zero, which is the failure mode if
  yfinance isn't capturing its dividend accrual correctly.
- Walk-forward fold sizing (`n_splits`/`min_train_size`/`test_size`) is
  auto-derived from however much data is actually passed in
  (`src/walkforward.py::auto_walk_forward_config`), rather than hard-coded to
  the ~9-year date range this was developed against. Verified against a
  synthetic 2-year window (4x shorter) and it re-derives sane fold sizes
  without breaking — but still worth sanity-checking the fold log output
  (`[walk-forward] auto-derived config from N rows: ...`) whenever you point
  this at a genuinely different date range.
- `mu`/`Σ` use simple historical estimators; shrinkage (Ledoit-Wolf) or a
  factor model would likely produce more stable weights fold to fold.
- No regime-persistence smoothing — a single-day regime flip triggers a full
  rebalance cycle; a minimum-dwell-time rule on regimes could reduce turnover
  further without hiding the transaction-cost effect entirely.
