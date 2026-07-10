"""
regime.py — HMM-based regime classification.

Key discipline: `fit_hmm` is always called on a TRAINING slice only. The
walk-forward loop (see backtest.py) is responsible for re-fitting a fresh
HMM inside every fold and only ever calling `.predict()` on the
corresponding test slice with that fold's own model + scaler.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm


def fit_hmm(X_train: np.ndarray, n_states=3, covariance_type="diag", random_state=42, n_iter=200):
    """Fit a Gaussian HMM on already-scaled training features."""
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X_train)
    return model


def label_states_by_volatility(states: np.ndarray, vol_series: pd.Series):
    """
    HMM state indices (0..k-1) are arbitrary. Map them to human-readable
    labels using average volatility within each state: highest avg vol ->
    Crisis, lowest -> Bull, everything else -> Bear (works cleanly for
    n_states=3; for more states this becomes a simple ordinal ranking).
    """
    tmp = pd.DataFrame({"state": states, "vol": vol_series.values})
    order = tmp.groupby("state")["vol"].mean().sort_values().index.tolist()

    if len(order) == 3:
        names = ["Bull", "Bear", "Crisis"]
    else:
        # generic fallback for n_states != 3: rank by vol, low->high
        names = [f"Regime_{i}" for i in range(len(order))]

    label_map = {state: names[i] for i, state in enumerate(order)}
    return label_map


def predict_regimes(model: hmm.GaussianHMM, X: np.ndarray, vol_ref: pd.Series):
    """
    Run Viterbi decoding (model.predict uses Viterbi by default in
    hmmlearn) on X, then map arbitrary state indices to Bull/Bear/Crisis
    using volatility ranking computed on the SAME slice being predicted.

    NOTE: when used inside a walk-forward test fold, `vol_ref` should be
    the vol feature for that fold only — the label mapping is a relabeling
    convenience, not a source of leakage, since it doesn't change which
    day gets which raw state, only what we call that state.
    """
    states = model.predict(X)
    label_map = label_states_by_volatility(states, vol_ref)
    labels = pd.Series(states, index=vol_ref.index).map(label_map)
    return states, labels, label_map
