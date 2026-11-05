"""Research helpers: how much to trust a backtest or a pair scan.

Python only and numpy only (plus the standard library).  Not ported to the
other languages and not in the goldens.  The bootstrap is the only code in
the package that uses random numbers; it is seeded by the caller.

Sources: Politis and Romano (1994) for resampling blocks; Bailey and Lopez
de Prado (2012, 2014) for the probabilistic and deflated Sharpe ratios;
Ploberger and Kramer (1992) for the CUSUM test; Newey and West (1987) for
the long-run variance; Benjamini and Hochberg (1995) for the FDR procedure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import combinations
from statistics import NormalDist

import numpy as np

from .backtest import BacktestConfig, backtest_pair
from .engle_granger import engle_granger_both
from .kalman import kalman_hedge
from .signals import half_life

_NORMAL = NormalDist()
_EULER_GAMMA = 0.5772156649015329
_ANNUAL = math.sqrt(252.0)


def _returns(r: np.ndarray, minimum: int) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if r.ndim != 1 or r.shape[0] < minimum:
        raise ValueError(f"need a 1-D series of at least {minimum} points")
    if not np.all(np.isfinite(r)):
        raise ValueError("inputs must be finite (no NaN/inf)")
    return r


def _sharpe(r: np.ndarray) -> float:
    sd = float(r.std(ddof=1))
    return float(r.mean()) / sd * _ANNUAL if sd > 0.0 else 0.0


@dataclass(frozen=True)
class SharpeCI:
    """Annualised Sharpe and its bootstrap percentile interval."""

    sharpe: float
    lower: float
    upper: float


def block_bootstrap_sharpe_ci(
    returns: np.ndarray,
    block: int = 20,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> SharpeCI:
    """Circular block bootstrap interval for the annualised Sharpe.

    Blocks keep the short-range dependence that an i.i.d. resample destroys.
    The same seed gives the same interval.
    """
    r = _returns(returns, 3)
    n = r.shape[0]
    if not 1 <= block <= n:
        raise ValueError("block must lie in [1, len(returns)]")
    if n_boot < 1 or not 0.0 < level < 1.0:
        raise ValueError("need n_boot >= 1 and 0 < level < 1")
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    offsets = np.arange(block)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + offsets) % n).ravel()[:n]
        stats[i] = _sharpe(r[idx])
    tail = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(stats, [tail, 100.0 - tail])
    return SharpeCI(_sharpe(r), float(lo), float(hi))


def _moments(r: np.ndarray) -> tuple[float, float, float]:
    """Per-period Sharpe, skewness and (non-excess) kurtosis."""
    if np.ptp(r) == 0.0:
        raise ValueError("returns are constant: Sharpe undefined")
    sd = float(r.std(ddof=1))
    c = r - r.mean()
    m2 = float(np.mean(c**2))
    return (float(r.mean()) / sd, float(np.mean(c**3)) / m2**1.5,
            float(np.mean(c**4)) / m2**2)


def probabilistic_sharpe(returns: np.ndarray, benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark), allowing for skew and fat tails.

    ``benchmark`` is a per-period Sharpe (annual / sqrt(252)).
    """
    r = _returns(returns, 3)
    sr, skew, kurt = _moments(r)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom <= 0.0:
        raise ValueError("Sharpe variance estimate is not positive")
    return _NORMAL.cdf((sr - benchmark) * math.sqrt(r.shape[0] - 1) / math.sqrt(denom))


def expected_max_sharpe(n_trials: int, trial_sharpe_var: float) -> float:
    """Expected best per-period Sharpe of ``n_trials`` skill-free trials."""
    if n_trials < 1 or trial_sharpe_var < 0.0:
        raise ValueError("need n_trials >= 1 and trial_sharpe_var >= 0")
    if n_trials == 1:
        return 0.0
    g = _EULER_GAMMA
    return math.sqrt(trial_sharpe_var) * (
        (1.0 - g) * _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
        + g * _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe(
    returns: np.ndarray, n_trials: int, trial_sharpe_var: float
) -> float:
    """Probabilistic Sharpe against the best result luck alone would give.

    ``trial_sharpe_var`` is the variance of the per-period Sharpe across all
    the configurations that were tried.
    """
    return probabilistic_sharpe(
        returns, expected_max_sharpe(n_trials, trial_sharpe_var)
    )


def cost_sensitivity(
    y: np.ndarray,
    x: np.ndarray,
    alpha: float,
    beta: float,
    config: BacktestConfig,
    total_bps: tuple[float, ...] = (0.0, 1.5, 3.0, 5.0, 10.0, 20.0),
) -> list[dict[str, float]]:
    """Re-run the backtest at each total cost; one metrics row per level."""
    rows = []
    for bps in total_bps:
        cfg = replace(config, cost_bps=bps, slippage_bps=0.0)
        res = backtest_pair(y, x, alpha, beta, cfg)
        rows.append({"total_bps": float(bps), **res.metrics})
    return rows


@dataclass(frozen=True)
class CusumResult:
    """CUSUM break test: statistic, p-value and most likely break index."""

    stat: float
    pvalue: float
    break_index: int

    def reject(self, alpha: float = 0.05) -> bool:
        return self.pvalue < alpha


def _long_run_variance(c: np.ndarray, lags: int) -> float:
    n = c.shape[0]
    lrv = float(c @ c) / n
    for k in range(1, lags + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * float(c[k:] @ c[:-k]) / n
    return lrv


def _kolmogorov_sf(stat: float) -> float:
    """P(sup |Brownian bridge| > stat)."""
    p = 2.0 * sum((-1.0) ** (j - 1) * math.exp(-2.0 * j * j * stat * stat)
                  for j in range(1, 101))
    return min(max(p, 0.0), 1.0)


def cusum_break_test(s: np.ndarray, lags: int | None = None) -> CusumResult:
    """Test a spread for a shift in mean.

    Statistic: sup_k |sum_{t<=k}(s_t - mean)| / (sigma_LR sqrt(n)), which
    converges to the supremum of a Brownian bridge under a stable mean.  The
    Bartlett long-run variance keeps a persistent but stable spread from
    being read as a break.  Default lags: the Andrews (1991) AR(1) plug-in,
    floor(1.1447 (a n)^(1/3)) with a = 4 rho^2 / (1 - rho^2)^2 and rho capped
    at 0.97, because a fixed rule is far too short for a spread with a
    half-life of weeks.  The test still leans towards rejecting on very
    persistent spreads, so treat a small p as a prompt to look.
    """
    v = _returns(s, 12)
    n = v.shape[0]
    if np.ptp(v) == 0.0:
        raise ValueError("series is constant: CUSUM undefined")
    c = v - v.mean()
    if lags is None:
        rho = min(abs(float(c[1:] @ c[:-1]) / float(c @ c)), 0.97)
        a = 4.0 * rho * rho / (1.0 - rho * rho) ** 2
        lags = min(int(math.floor(1.1447 * (a * n) ** (1.0 / 3.0))), n - 1)
    if not 0 <= lags < n:
        raise ValueError("lags must lie in [0, len(s))")
    lrv = _long_run_variance(c, lags)
    if lrv <= 0.0:
        lrv = float(c @ c) / n  # Bartlett weights keep this rare; be safe
    path = np.abs(np.cumsum(c))
    stat = float(path.max()) / math.sqrt(lrv * n)
    return CusumResult(stat, _kolmogorov_sf(stat), int(np.argmax(path)))


def benjamini_hochberg(
    pvalues: np.ndarray, q: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up; returns (rejected mask, adjusted p-values)."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or p.shape[0] == 0:
        raise ValueError("need a non-empty 1-D array of p-values")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly inside (0, 1)")
    m = p.shape[0]
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adj_sorted
    return adjusted <= q, adjusted


@dataclass(frozen=True)
class KalmanFit:
    """Grid MLE result with innovation diagnostics at the chosen point.

    A well-specified filter has z_sd near 1 and z_ac1 near 0.
    """

    delta: float
    r: float
    loglik: float
    z_sd: float
    z_ac1: float
    grid: np.ndarray  # loglik, shape (len(deltas), len(rs))


def kalman_grid_mle(
    y: np.ndarray,
    x: np.ndarray,
    deltas: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2),
    rs: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
    burn: int = 2,
) -> KalmanFit:
    """Pick (delta, r) by Gaussian innovation log-likelihood on a grid.

    The first ``burn`` innovations are dropped: they mostly reflect the
    prior.  Fit on training data only; the result is not out-of-sample.
    """
    if burn < 0 or not deltas or not rs:
        raise ValueError("need burn >= 0 and non-empty grids")
    grid = np.empty((len(deltas), len(rs)))
    best = None
    for i, d in enumerate(deltas):
        for j, r in enumerate(rs):
            k = kalman_hedge(y, x, delta=d, r=r)
            e, sv = k.innovation[burn:], k.s_var[burn:]
            if e.shape[0] < 3:
                raise ValueError("series too short for this burn")
            ll = -0.5 * float(np.sum(np.log(2.0 * math.pi * sv) + e * e / sv))
            grid[i, j] = ll
            if best is None or ll > best[0]:
                best = (ll, d, r, k.z[burn:])
    ll, d, r, z = best  # type: ignore[misc]
    c = z - z.mean()
    denom = float(c @ c)
    ac1 = float(c[1:] @ c[:-1]) / denom if denom > 0.0 else 0.0
    return KalmanFit(d, r, ll, float(z.std(ddof=1)), ac1, grid)


@dataclass(frozen=True)
class PairScanRow:
    """One candidate pair.  ``selected`` = survives BH and the half-life filter."""

    y: str
    x: str
    pvalue: float
    adjusted_pvalue: float
    half_life: float
    selected: bool


def scan_pairs(
    prices: dict[str, np.ndarray],
    maxlag: int = 1,
    q: float = 0.05,
    max_half_life: float = 60.0,
) -> list[PairScanRow]:
    """Screen every pair: Engle-Granger both ways, BH, then half-life.

    A pair's p-value is the larger of its two directions, so it must pass
    both ways.  The p-values are the interpolated approximation of adf.py
    and clamp at 1e-4.  A pair with beta <= 0 in the y-on-x fit is never
    selected, because the backtest refuses it.  Rows come back with the
    smallest p-value first.
    """
    names = sorted(prices)
    if len(names) < 2:
        raise ValueError("need at least two series")
    pairs = list(combinations(names, 2))
    fits = [engle_granger_both(prices[a], prices[b], maxlag) for a, b in pairs]
    pvals = np.array([max(f.y_on_x.adf.pvalue, f.x_on_y.adf.pvalue) for f in fits])
    rejected, adjusted = benjamini_hochberg(pvals, q)
    rows = []
    for i, ((a, b), f) in enumerate(zip(pairs, fits, strict=True)):
        hl = half_life(f.y_on_x.resid)
        keep = (bool(rejected[i]) and 0.0 < hl < max_half_life
                and f.y_on_x.beta > 0.0)
        rows.append(PairScanRow(a, b, float(pvals[i]), float(adjusted[i]), hl, keep))
    rows.sort(key=lambda row: row.pvalue)
    return rows
