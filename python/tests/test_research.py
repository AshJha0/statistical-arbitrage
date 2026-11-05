"""research.py on cases with known answers."""
from __future__ import annotations

import math

import numpy as np
import pytest

from statarb import BacktestConfig
from statarb.research import (
    _kolmogorov_sf,
    benjamini_hochberg,
    block_bootstrap_sharpe_ci,
    cost_sensitivity,
    cusum_break_test,
    deflated_sharpe,
    expected_max_sharpe,
    kalman_grid_mle,
    probabilistic_sharpe,
    scan_pairs,
)


def _ar1(T: int, phi: float, rng) -> np.ndarray:
    v = np.zeros(T)
    for t in range(1, T):
        v[t] = phi * v[t - 1] + rng.normal()
    return v


def test_bootstrap_seeded_and_covers_truth() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 2000)
    a = block_bootstrap_sharpe_ci(r, block=20, n_boot=400, seed=7)
    b = block_bootstrap_sharpe_ci(r, block=20, n_boot=400, seed=7)
    assert a == b
    assert a.lower < a.sharpe < a.upper
    # sd of an annualised Sharpe over 2000 days is about sqrt(252/2000) = 0.35
    assert a.upper - a.lower == pytest.approx(4 * 0.355, rel=0.25)
    assert a != block_bootstrap_sharpe_ci(r, block=20, n_boot=400, seed=8)


def test_bootstrap_block_widens_on_autocorrelated_returns() -> None:
    r = 0.001 + 0.01 * _ar1(3000, 0.7, np.random.default_rng(1))
    narrow = block_bootstrap_sharpe_ci(r, block=1, n_boot=400, seed=1)
    wide = block_bootstrap_sharpe_ci(r, block=50, n_boot=400, seed=1)
    assert wide.upper - wide.lower > 1.5 * (narrow.upper - narrow.lower)


def test_bootstrap_validation() -> None:
    with pytest.raises(ValueError):
        block_bootstrap_sharpe_ci(np.ones(2))
    with pytest.raises(ValueError):
        block_bootstrap_sharpe_ci(np.arange(10.0), block=11)
    with pytest.raises(ValueError):
        block_bootstrap_sharpe_ci(np.array([1.0, math.nan, 2.0]), block=1)


def test_probabilistic_sharpe_known_value() -> None:
    """Symmetric two-point returns: skew 0, kurtosis 1, so the denominator is
    1 and PSR = Phi(sr * sqrt(n-1))."""
    r = np.array([0.02, -0.01] * 200)
    sr = r.mean() / r.std(ddof=1)
    want = 0.5 * (1.0 + math.erf(sr * math.sqrt(399) / math.sqrt(2.0)))
    assert probabilistic_sharpe(r) == pytest.approx(want, rel=1e-9)
    assert probabilistic_sharpe(r, benchmark=sr) == pytest.approx(0.5, abs=1e-12)


def test_probabilistic_sharpe_constant_rejected() -> None:
    with pytest.raises(ValueError, match="constant"):
        probabilistic_sharpe(np.full(10, 0.01))


def test_expected_max_sharpe_values() -> None:
    assert expected_max_sharpe(1, 1.0) == 0.0
    # Bailey and Lopez de Prado approximation for 1000 trials of unit variance
    assert expected_max_sharpe(1000, 1.0) == pytest.approx(3.26, abs=0.02)
    assert expected_max_sharpe(100, 4.0) == pytest.approx(
        2.0 * expected_max_sharpe(100, 1.0))


def test_deflated_sharpe_falls_with_trials() -> None:
    r = np.random.default_rng(2).normal(0.001, 0.01, 1000)
    assert deflated_sharpe(r, 1, 0.01) == pytest.approx(probabilistic_sharpe(r))
    assert deflated_sharpe(r, 100, 0.001) < deflated_sharpe(r, 10, 0.001) \
        < probabilistic_sharpe(r)


def test_cost_sensitivity_linear_in_cost() -> None:
    """Costs change neither the trades nor the gross P&L here, so total
    return falls exactly by turnover * bps."""
    rng = np.random.default_rng(3)
    x = 100.0 + np.cumsum(rng.normal(0.0, 0.2, 800))
    s = _ar1(800, 0.9, rng)
    y = 10.0 + 1.5 * x + s + 50.0
    cfg = BacktestConfig()
    rows = cost_sensitivity(y, x, 60.0, 1.5, cfg, (0.0, 10.0))
    gap = rows[0]["total_return"] - rows[1]["total_return"]
    assert rows[0]["turnover"] > 0.0
    assert gap == pytest.approx(rows[0]["turnover"] * 10.0 / 1e4, rel=1e-9)


def test_cusum_stable_vs_shift() -> None:
    rng = np.random.default_rng(4)
    stable = rng.normal(size=1000)
    assert not cusum_break_test(stable).reject(0.01)
    shifted = stable.copy()
    shifted[600:] += 1.0
    res = cusum_break_test(shifted)
    assert res.reject(0.01)
    assert abs(res.break_index - 599) < 30


def test_cusum_size_on_white_noise() -> None:
    rejected = sum(
        cusum_break_test(np.random.default_rng(i).normal(size=300), lags=0).reject(0.05)
        for i in range(300))
    assert rejected / 300 < 0.09


def test_cusum_closed_form_statistic_and_kolmogorov_tail() -> None:
    n = 1000
    v = np.concatenate([np.full(n // 2, 1.0), np.full(n // 2, -1.0)])
    res = cusum_break_test(v, lags=0)  # sup|cumsum| = n/2, variance 1
    assert res.stat == pytest.approx(math.sqrt(n) / 2.0)
    assert res.break_index == n // 2 - 1
    assert _kolmogorov_sf(1.358) == pytest.approx(0.05, abs=5e-4)
    assert _kolmogorov_sf(1.628) == pytest.approx(0.01, abs=2e-4)


def test_cusum_default_lags_handle_persistent_stable_spread() -> None:
    hits = sum(cusum_break_test(_ar1(900, 0.92, np.random.default_rng(i))).reject(0.01)
               for i in range(100))
    assert hits < 15


def test_cusum_validation() -> None:
    with pytest.raises(ValueError, match="constant"):
        cusum_break_test(np.ones(50))
    with pytest.raises(ValueError):
        cusum_break_test(np.arange(5.0))
    with pytest.raises(ValueError):
        cusum_break_test(np.arange(50.0), lags=50)


def test_bh_textbook_example() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.2])
    rejected, adjusted = benjamini_hochberg(p, 0.05)
    assert adjusted.tolist() == pytest.approx([0.04, 0.0533333333, 0.0533333333, 0.2])
    assert rejected.tolist() == [True, False, False, False]


def test_bh_step_up_rescues_earlier_ranks() -> None:
    rejected, _ = benjamini_hochberg(np.array([0.04, 0.041, 0.042, 0.043]), 0.05)
    assert rejected.all()


def test_bh_validation() -> None:
    for bad in (np.array([]), np.array([1.5]), np.array([math.nan])):
        with pytest.raises(ValueError):
            benjamini_hochberg(bad)
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([0.1]), 1.0)


def test_kalman_mle_recovers_observation_noise() -> None:
    rng = np.random.default_rng(6)
    x = 50.0 + np.cumsum(rng.normal(0.0, 0.3, 1500))
    y = 2.0 + 1.3 * x + rng.normal(0.0, 1.0, 1500)    # r = 1, constant state
    fit = kalman_grid_mle(y, x, deltas=(1e-8, 1e-6, 1e-2),
                          rs=(0.01, 1.0, 100.0))
    assert fit.r == 1.0 and fit.delta < 1e-2
    assert fit.grid.shape == (3, 3)
    assert fit.loglik == fit.grid.max()
    assert fit.z_sd == pytest.approx(1.0, abs=0.1)
    assert abs(fit.z_ac1) < 0.1


def test_kalman_mle_validation() -> None:
    y = np.arange(4.0)
    with pytest.raises(ValueError):
        kalman_grid_mle(y, y, deltas=())
    with pytest.raises(ValueError):
        kalman_grid_mle(y, y, burn=3)


def test_scan_pairs_finds_planted_pair() -> None:
    rng = np.random.default_rng(7)
    base = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 1000))
    prices = {
        "a": base,
        "b": 5.0 + 0.8 * base + _ar1(1000, 0.7, rng),
        "c": 100.0 + np.cumsum(rng.normal(0.0, 1.0, 1000)),
        "d": 100.0 + np.cumsum(rng.normal(0.0, 1.0, 1000)),
    }
    rows = scan_pairs(prices)
    assert len(rows) == 6
    assert [(r.y, r.x) for r in rows if r.selected] == [("a", "b")]
    assert rows[0].pvalue <= rows[-1].pvalue


def test_scan_pairs_half_life_filter_and_validation() -> None:
    rng = np.random.default_rng(8)
    base = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 1000))
    prices = {"a": base, "b": 0.8 * base + _ar1(1000, 0.7, rng)}
    assert scan_pairs(prices)[0].selected
    assert not scan_pairs(prices, max_half_life=0.5)[0].selected
    with pytest.raises(ValueError):
        scan_pairs({"a": base})


def test_scan_pairs_refuses_negative_hedge() -> None:
    rng = np.random.default_rng(9)
    base = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 1000))
    prices = {"a": base, "b": 300.0 - base + _ar1(1000, 0.5, rng)}
    row = scan_pairs(prices)[0]
    assert row.pvalue < 0.05 and not row.selected
