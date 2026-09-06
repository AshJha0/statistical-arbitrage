"""Spread construction, rolling z-score, and mean-reversion half-life."""
from __future__ import annotations

import math

import numpy as np

from .ols import ols

#: Rolling standard deviations at or below this are treated as
#: zero-variance windows: the z-score is NaN (no signal).  Pinned.
ZSCORE_STD_FLOOR: float = 1.0e-12


def spread(y: np.ndarray, x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Hedge spread s_t = y_t - (alpha + beta * x_t)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError("y and x must be 1-D arrays of equal length")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
        raise ValueError("inputs must be finite (no NaN/inf)")
    if not (math.isfinite(alpha) and math.isfinite(beta)):
        raise ValueError("alpha and beta must be finite")
    return y - (alpha + beta * x)


def rolling_zscore(s: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score with the current observation included (pinned).

    z_t = (s_t - mean(s_{t-W+1..t})) / std(s_{t-W+1..t}, ddof=1).

    The first W-1 entries are NaN (insufficient history).  A window whose
    sample standard deviation is <= ZSCORE_STD_FLOOR yields NaN (zero-
    variance guard) rather than +/-inf.

    Raises:
        ValueError: window < 2, window > len(s), or non-finite input.
    """
    s = np.asarray(s, dtype=float)
    if s.ndim != 1:
        raise ValueError("s must be 1-D")
    if not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError("window must be an integer >= 2")
    if window > s.shape[0]:
        raise ValueError(
            f"window ({window}) larger than series length ({s.shape[0]})"
        )
    if not np.all(np.isfinite(s)):
        raise ValueError("inputs must be finite (no NaN/inf)")
    T = s.shape[0]
    z = np.full(T, np.nan)
    for t in range(window - 1, T):
        w = s[t - window + 1 : t + 1]
        m = float(w.mean())
        sd = float(w.std(ddof=1))
        if sd > ZSCORE_STD_FLOOR:
            z[t] = (s[t] - m) / sd
    return z


def half_life(s: np.ndarray) -> float:
    """Half-life of mean reversion from an AR(1) fit on the spread.

    Regression (pinned):  ds_t = c + rho * s_{t-1} + e_t  (OLS, intercept).
    Half-life = -ln(2) / ln(1 + rho).

    Guards (pinned): rho >= 0 (no reversion) returns +inf; 1 + rho <= 0
    returns the sentinel 0.0.  Both sentinels mean "the AR(1) fit is not
    interpretable as a half-life — do not trade this spread": +inf is a
    random walk / explosive level, 0.0 is a sign-flipping fit (|1 + rho|
    is either an oscillating stationary root for -2 < rho < -1 or an
    explosive alternation for rho <= -2 — the sentinel deliberately does
    NOT distinguish them).  A filter such as ``0 < hl < 30`` excludes both.

    Raises:
        ValueError: fewer than 12 observations, constant series, or
            non-finite input (via the OLS validation).
    """
    s = np.asarray(s, dtype=float)
    if s.ndim != 1:
        raise ValueError("s must be 1-D")
    if s.shape[0] < 12:
        raise ValueError("need at least 12 observations for half-life")
    if not np.all(np.isfinite(s)):
        raise ValueError("inputs must be finite (no NaN/inf)")
    if np.ptp(s) == 0.0:
        raise ValueError("series is constant: half-life undefined")
    ds = np.diff(s)
    fit = ols(ds, s[:-1], intercept=True)
    rho = float(fit.params[1])
    if rho >= 0.0:
        return math.inf
    if 1.0 + rho <= 0.0:
        return 0.0
    return -math.log(2.0) / math.log(1.0 + rho)
