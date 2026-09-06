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


def test_finite_sample_critical_values_match_mackinnon_2010() -> None:
    """MAJ-1: cv(T) = b_inf + b1/T + b2/T^2 + b3/T^3 (MacKinnon 2010,
    constant/no-trend), cross-checked against statsmodels' embedded table
    and against a hand evaluation of the pinned coefficients."""
    from statsmodels.tsa.adfvalues import mackinnoncrit

    from statarb import RESPONSE_SURFACE, critical_values

    for nvars in (1, 2, 3):
        for T in (20, 25, 50, 100, 250, 898, 1498, 10_000):
            ours = critical_values(nvars, T)
            ref = mackinnoncrit(nvars, "c", T)
            assert np.allclose(ours, ref, atol=1e-12), (nvars, T)
        assert critical_values(nvars) == CRITICAL_VALUES[nvars]
    # hand evaluation of the pinned N=1, 5% row at T = 50
    b_inf, b1, b2, b3 = RESPONSE_SURFACE[1][1]
    assert (b_inf, b1, b2, b3) == (-2.86154, -2.8903, -4.234, -40.040)
    expect = -2.86154 - 2.8903 / 50 - 4.234 / 2500 - 40.040 / 125_000
    assert critical_values(1, 50)[1] == pytest.approx(expect, abs=1e-12)
    # finite-sample values are more negative than the asymptotic ones and
    # converge to them as T grows
    for nvars in (1, 2, 3):
        cv20 = critical_values(nvars, 20)
        cv_big = critical_values(nvars, 10**7)
        for i in range(3):
            assert cv20[i] < CRITICAL_VALUES[nvars][i]
            assert cv_big[i] == pytest.approx(CRITICAL_VALUES[nvars][i], abs=1e-5)


def test_adf_small_sample_guard() -> None:
    """Proposed test 9: T_eff < 20 is refused; at T_eff = 20 the verdict
    uses cv(20), which is the response surface evaluated at T = 20."""
    from statarb import critical_values

    t = np.arange(40.0)
    s = 0.9 ** t + 0.05 * np.sin(2.1 * t)        # deterministic AR(1)-ish
    with pytest.raises(ValueError, match="too short"):
        adf_test(s[:21], maxlag=1)               # T_eff = 19
    with pytest.raises(ValueError, match="too short"):
        adf_test(s[:15], maxlag=1)               # T_eff = 13 (old guard passed)
    res = adf_test(s[:22], maxlag=1)             # T_eff = 20
    assert res.nobs == 20
    cv1, cv5, cv10 = critical_values(1, 20)
    assert res.crit == {"1%": cv1, "5%": cv5, "10%": cv10}
    assert res.crit["5%"] == pytest.approx(
        -2.86154 - 2.8903 / 20 - 4.234 / 400 - 40.040 / 8000, abs=1e-12)
    # the verdict is a direct comparison against the finite-sample value
    assert res.reject(0.05) == (res.tstat < cv5)
    # p-value knots are the finite-sample values
    assert interpolate_pvalue(cv5, 1, 20) == pytest.approx(0.05, abs=1e-12)
    with pytest.raises(ValueError, match="effective"):
        critical_values(1, 19)
    with pytest.raises(ValueError, match="nvars"):
        critical_values(4, 100)


def test_adf_uses_finite_sample_table_on_bundled_data(walks: dict) -> None:
    """The bundled random walk (T_eff = 1498) is judged against the
    finite-sample 5% value, not the rounded asymptotic -2.86."""
    res = adf_test(walks["walk_a"], maxlag=1)
    assert res.nobs == 1498
    expect5 = -2.86154 - 2.8903 / 1498 - 4.234 / 1498**2 - 40.040 / 1498**3
    assert res.crit["5%"] == pytest.approx(expect5, abs=1e-12)
    assert res.crit["5%"] < -2.86154


def test_interpolate_pvalue_invalid_nvars_typed() -> None:
    """Proposed test 15 (MIN-8): ValueError, not KeyError."""
    with pytest.raises(ValueError, match="nvars"):
        interpolate_pvalue(-3.0, 4)
    with pytest.raises(ValueError, match="NaN"):
        interpolate_pvalue(np.nan, 1)
    # +/-inf t-stats (exact-fit regressions) clamp rather than raise
    assert interpolate_pvalue(-np.inf, 1) == 1e-4
    assert interpolate_pvalue(np.inf, 1) == 0.9999


def test_adf_maxlag_upper_bound() -> None:
    from statarb.adf import MAX_LAG

    rng = np.random.default_rng(11)
    y = rng.normal(size=100)
    with pytest.raises(ValueError, match="maxlag"):
        adf_test(y, maxlag=MAX_LAG + 1)
    with pytest.raises(ValueError, match="too short"):
        adf_test(y, maxlag=MAX_LAG)
    with pytest.raises(ValueError, match="maxlag"):
        adf_test(y, maxlag=2.0)  # type: ignore[arg-type]
