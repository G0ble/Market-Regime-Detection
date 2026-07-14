"""
Shared utilities for market regime detection.

Core components:
  - Data loading and feature engineering
  - Gaussian HMM with causal (filtered) probabilities
  - Wasserstein k-means clustering
  - Walk-forward validation for both methods
  - Backtest metrics
"""

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(ticker="SPY", start="2006-01-01", end="2025-01-01"):
    """Download prices and build the feature set.

    Features:
      returns    -- daily log return, x100
      volatility -- 21-day rolling std of log returns, annualised (x sqrt(252))
      momentum   -- 21-day cumulative log return, x100
    """
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    log_ret = np.log(data["Close"]).diff()

    df = pd.DataFrame(index=data.index)
    df["returns"] = log_ret * 100
    df["volatility"] = log_ret.rolling(21).std() * 100 * np.sqrt(252)
    df["momentum"] = log_ret.rolling(21).sum() * 100
    df = df.dropna()

    return df, data


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------

def fit_hmm(X, n_states=2, n_inits=10, n_iter=200):
    """Fit a Gaussian HMM with multiple random initialisations.

    EM converges to local optima; running several starts and keeping the
    highest log-likelihood reduces that risk. Matters most on short windows.
    """
    best_ll, best_model = -np.inf, None
    for seed in range(n_inits):
        m = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=seed,
            tol=1e-4,
        )
        try:
            m.fit(X)
            ll = m.score(X)
            if ll > best_ll:
                best_ll, best_model = ll, m
        except Exception:
            continue
    return best_model, best_ll


def sort_states_by_variance(model):
    """Return state ordering by ascending variance (calm first).

    HMM state labels are arbitrary: "state 0" may be high-vol in one fit and
    low-vol in the next. Anchoring labels to an intrinsic property (variance)
    keeps them consistent across refits. Essential for walk-forward.
    """
    variances = np.array(
        [np.trace(model.covars_[k]) for k in range(model.n_components)]
    )
    return np.argsort(variances)


def compute_filtered_probs(model, X):
    """Filtered probabilities P(state_t | obs_1..t) via the forward algorithm.

    hmmlearn's predict_proba() returns SMOOTHED probabilities, which condition
    on the full sample including the future -- look-ahead bias. predict() uses
    Viterbi, which is also a global decode. Neither is usable as a trading
    signal. The forward recursion below only ever touches t-1 and t.

    Runs in log-space to avoid numerical underflow.
    """
    framelogprob = model._compute_log_likelihood(X)  # private API, see README

    n_samples = X.shape[0]
    n_states = model.n_components

    log_startprob = np.log(model.startprob_ + 1e-300)
    log_transmat = np.log(model.transmat_ + 1e-300)

    fwd = np.zeros((n_samples, n_states))
    fwd[0] = log_startprob + framelogprob[0]

    for t in range(1, n_samples):
        for j in range(n_states):
            fwd[t, j] = framelogprob[t, j] + np.logaddexp.reduce(fwd[t - 1] + log_transmat[:, j])

    log_norm = np.logaddexp.reduce(fwd, axis=1, keepdims=True)
    return np.exp(fwd - log_norm)


def walk_forward_hmm(df, features, train_years=4, refit_freq="ME",
                     n_states=2, n_inits=5):
    """Walk-forward HMM: refit monthly on a rolling window, infer causally.

    Filtered probabilities alone are not enough -- the MODEL must also be
    trained only on past data. A model fitted on the full sample "knows" what
    a crisis looks like because it has seen COVID.

    Rolling (not expanding) window because markets are non-stationary. The
    cost: fewer observations per fit, so states can be poorly separated in
    windows containing no real stress. `diagnostics` tracks that.
    """
    X_all = df[features].values
    dates = df.index
    probs = np.full((len(df), n_states), np.nan)
    diagnostics = []

    refit_dates = pd.date_range(
        start=dates[0] + pd.DateOffset(years=train_years),
        end=dates[-1],
        freq=refit_freq,
    )

    for i, refit_date in enumerate(refit_dates):
        train_start = refit_date - pd.DateOffset(years=train_years)
        train_mask = (dates >= train_start) & (dates < refit_date)
        X_train = X_all[train_mask]

        if len(X_train) < 250:
            continue

        model, _ = fit_hmm(X_train, n_states=n_states, n_inits=n_inits)
        if model is None:
            continue

        order = sort_states_by_variance(model)

        vol_idx = features.index("volatility")
        vols = model.means_[order, vol_idx]
        diagnostics.append({
            "date": refit_date,
            "vol_low": vols[0],
            "vol_high": vols[-1],
            "separation": vols[-1] / vols[0],
        })

        next_refit = (refit_dates[i + 1] if i + 1 < len(refit_dates)
                      else dates[-1] + pd.Timedelta(days=1))
        apply_mask = (dates >= refit_date) & (dates < next_refit)
        if apply_mask.sum() == 0:
            continue

        # Forward pass over all history up to the end of the application
        # period (the recursion needs its memory), then keep only the new rows.
        hist_mask = dates < next_refit
        filt = compute_filtered_probs(model, X_all[hist_mask])[:, order]
        probs[apply_mask] = filt[-apply_mask.sum():]

    return probs, pd.DataFrame(diagnostics).set_index("date")


# ---------------------------------------------------------------------------
# Wasserstein
# ---------------------------------------------------------------------------

def wasserstein_distance_p(x, y, p=2):
    """p-Wasserstein distance between two equal-sized 1D samples.

    In 1D optimal transport has a closed form: sort both samples, pair the
    quantiles, average the p-th powers of the gaps. Sorting is what makes the
    distance permutation-invariant -- it compares DISTRIBUTIONS, not
    time-ordered sequences.
    """
    x = np.sort(x)
    y = np.sort(y)
    return (np.mean(np.abs(x - y) ** p)) ** (1 / p)


def wasserstein_barycenter(windows):
    """Wasserstein barycenter of a set of 1D samples (closed form).

    The barycenter is the average of the quantile functions: sort each sample,
    then average position by position.
    """
    return np.sort(windows, axis=1).mean(axis=0)


def build_windows(returns, window_size=21, step=1):
    """Slice a return series into overlapping windows.

    Each window is treated as an empirical distribution -- a regime candidate.
    Each is anchored to its LAST day: the window covering days 100-120 can only
    inform a decision on day 120 or later.

    Note: with step=1 adjacent windows share 20 of 21 observations and are
    therefore highly correlated. This is inherent to the approach.
    """
    windows, end_idx = [], []
    for i in range(0, len(returns) - window_size + 1, step):
        windows.append(returns[i:i + window_size])
        end_idx.append(i + window_size - 1)
    return np.array(windows), np.array(end_idx)


def wasserstein_kmeans(windows, k=2, p=2, max_iter=50, seed=42):
    """Wasserstein k-means (Horvath, Issa & Muguruza, 2021).

    Standard k-means with the Wasserstein distance for assignment and the
    Wasserstein barycenter for the update step. This is NOT the MDS + euclidean
    k-means shortcut common in blog implementations, which distorts the
    geometry by projecting the distance matrix into 2D before clustering.
    """
    rng = np.random.default_rng(seed)
    n = len(windows)

    idx = rng.choice(n, size=k, replace=False)
    centroids = np.sort(windows[idx], axis=1)
    labels = np.zeros(n, dtype=int)

    for it in range(max_iter):
        new_labels = np.array([
            np.argmin([wasserstein_distance_p(windows[i], centroids[j], p)
                       for j in range(k)])
            for i in range(n)
        ])

        if it > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for j in range(k):
            if (labels == j).sum() > 0:
                centroids[j] = wasserstein_barycenter(windows[labels == j])

    return labels, centroids


def walk_forward_wasserstein(df, window_size=21, train_years=4,
                             refit_freq="ME", k=3, p=2):
    """Walk-forward WK-means: fit centroids on past data, assign causally.

    Mirrors the HMM walk-forward: centroids are frozen on the training window,
    then each new day is assigned by computing the distance from its trailing
    window to those centroids. Fit/assign separation guarantees causality.
    """
    returns_arr = df["returns"].values
    dates = df.index
    signal = pd.Series(np.nan, index=dates)

    refit_dates = pd.date_range(
        start=dates[0] + pd.DateOffset(years=train_years),
        end=dates[-1],
        freq=refit_freq,
    )

    for i, refit_date in enumerate(refit_dates):
        train_start = refit_date - pd.DateOffset(years=train_years)
        train_mask = (dates >= train_start) & (dates < refit_date)
        if train_mask.sum() < 250:
            continue

        w_train, _ = build_windows(returns_arr[train_mask], window_size, step=1)
        _, centroids = wasserstein_kmeans(w_train, k=k, p=p, seed=42)

        # Anti label-switching: order centroids by dispersion
        order = np.argsort([c.std() for c in centroids])
        centroids = centroids[order]

        next_refit = (refit_dates[i + 1] if i + 1 < len(refit_dates)
                      else dates[-1] + pd.Timedelta(days=1))
        apply_mask = (dates >= refit_date) & (dates < next_refit)

        for t in np.where(apply_mask)[0]:
            if t < window_size:
                continue
            w_t = returns_arr[t - window_size + 1: t + 1]
            dists = [wasserstein_distance_p(w_t, centroids[j], p)
                     for j in range(k)]
            signal.iloc[t] = np.argmin(dists)

    return signal


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

COST = 0.0005  # 5 bps per unit of turnover


def backtest(exposure, returns, cost=COST):
    """Net strategy returns from an exposure series.

    Exposure is lagged one day. A signal computed from today's close
    cannot trade today's return.
    """
    gross = exposure.shift(1) * returns
    turnover = exposure.diff().abs()
    costs = turnover.shift(1) * cost
    return gross - costs.fillna(0)


def metrics(returns, name=""):
    """CAGR, annualised vol, Sharpe and max drawdown."""
    r = returns.dropna()
    cum = (1 + r).cumprod()
    years = len(r) / 252
    cagr = cum.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = cagr / vol
    max_dd = (cum / cum.cummax() - 1).min()

    if name:
        print(f"{name:22} | {cagr:7.2%} | {vol:7.2%} | {sharpe:6.2f} | {max_dd:8.2%}")

    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": max_dd}


def metrics_header():
    print(f"{'Strategy':22} | {'CAGR':>7} | {'Vol':>7} | {'Sharpe':>6} | {'MaxDD':>8}")
    print("-" * 68)
