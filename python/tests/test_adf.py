"""ADF test: verdicts on bundled data, statsmodels cross-check, p-value
interpolation, error paths."""
from __future__ import annotations

import numpy as np
import pytest

from statarb import CRITICAL_VALUES, adf_test, half_life, interpolate_pvalue


def test_white_noise_strongly_rejects() -> None:
    rng = np.random.default_rng(2)
    wn = rng.normal(0.0, 1.0, 800)
    res = adf_test(wn, maxlag=1)
    assert res.tstat < CRITICAL_VALUES[1][0]  # beyond even the 1% value
    assert res.reject(0.01) and res.reject(0.05) and res.reject(0.10)
    assert res.pvalue <= 0.01 + 1e-12


def test_random_walk_does_not_reject(walks: dict) -> None:
    res = adf_test(walks["walk_a"], maxlag=1)
    assert not res.reject(0.05)
    assert res.pvalue > 0.10


def test_vs_statsmodels(pair: dict, walks: dict) -> None:
    from statsmodels.tsa.stattools import adfuller

    for series, lag in [
        (walks["walk_a"], 1),
        (walks["walk_b"], 3),
        (np.diff(pair["stock_y"]), 2),
        (pair["stock_x"][:400], 0),
    ]:
        ours = adf_test(series, maxlag=lag)
        theirs = adfuller(series, maxlag=lag, regression="c", autolag=None)[0]
        assert ours.tstat == pytest.approx(theirs, abs=1e-8)


def test_effective_nobs() -> None:
    rng = np.random.default_rng(4)
    y = rng.normal(size=100)
    for lag in (0, 1, 4):
        assert adf_test(y, maxlag=lag).nobs == 100 - 1 - lag


def test_pvalue_interpolation_pinned_points() -> None:
    for nvars, (cv1, cv5, cv10) in CRITICAL_VALUES.items():
        assert interpolate_pvalue(cv1, nvars) == pytest.approx(0.01, abs=1e-12)
        assert interpolate_pvalue(cv5, nvars) == pytest.approx(0.05, abs=1e-12)
        assert interpolate_pvalue(cv10, nvars) == pytest.approx(0.10, abs=1e-12)
    # monotone non-decreasing in the t-stat, and clamped
    grid = [interpolate_pvalue(t, 1) for t in np.linspace(-8.0, 4.0, 60)]
    assert all(a <= b + 1e-15 for a, b in zip(grid, grid[1:]))
    assert interpolate_pvalue(-50.0, 1) == pytest.approx(1e-4)
    assert interpolate_pvalue(50.0, 1) == pytest.approx(0.9999)


def test_error_paths() -> None:
    rng = np.random.default_rng(5)
    good = rng.normal(size=100)
    with pytest.raises(ValueError, match="constant"):
        adf_test(np.full(100, 1.0))
    with pytest.raises(ValueError, match="maxlag"):
        adf_test(good, maxlag=-1)
    with pytest.raises(ValueError, match="nvars"):
        adf_test(good, nvars=4)
    with pytest.raises(ValueError, match="too short"):
        adf_test(good[:10], maxlag=1)
    with pytest.raises(ValueError, match="finite"):
        adf_test(np.concatenate([good, [np.inf]]))
    with pytest.raises(ValueError, match="alpha"):
        adf_test(good).reject(0.20)


def test_half_life_on_known_ar1() -> None:
    """AR(1) with phi = 0.9 has half-life -ln2/ln(0.9) ~ 6.58 days."""
    rng = np.random.default_rng(6)
    T = 20_000
    s = np.zeros(T)
    eps = rng.normal(0.0, 1.0, T)
    for t in range(1, T):
        s[t] = 0.9 * s[t - 1] + eps[t]
    hl = half_life(s)
    expected = -np.log(2.0) / np.log(0.9)
    assert hl == pytest.approx(expected, rel=0.15)


def test_half_life_guards(walks: dict) -> None:
    # a random walk drifts: rho-hat can be >= 0 -> +inf guard, or tiny
    # negative -> huge but finite half-life; either way, no crash and > 100.
    assert half_life(walks["walk_a"]) > 100.0
    with pytest.raises(ValueError, match="constant"):
        half_life(np.full(50, 2.0))
    with pytest.raises(ValueError, match="12 observations"):
        half_life(np.arange(5.0))
