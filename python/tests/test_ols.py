"""OLS: closed-form checks, statsmodels cross-check, error paths."""
from __future__ import annotations

import numpy as np
import pytest

from statarb import ols


def test_two_var_closed_form() -> None:
    """OLS slope/intercept must match the covariance closed form."""
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 2.0, 400)
    y = 3.0 + 0.5 * x + rng.normal(0.0, 0.1, 400)
    fit = ols(y, x, intercept=True)
    b_cf = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    a_cf = float(y.mean() - b_cf * x.mean())
    assert fit.params[1] == pytest.approx(b_cf, abs=1e-12)
    assert fit.params[0] == pytest.approx(a_cf, abs=1e-12)
    r_cf = float(np.corrcoef(x, y)[0, 1] ** 2)
    assert fit.r2 == pytest.approx(r_cf, abs=1e-12)


def test_exact_fit_and_no_intercept() -> None:
    x = np.arange(1.0, 30.0)
    y = 2.0 * x
    fit = ols(y, x, intercept=False)
    assert fit.params[0] == pytest.approx(2.0, abs=1e-12)
    assert fit.r2 == pytest.approx(1.0, abs=1e-12)
    assert np.max(np.abs(fit.resid)) < 1e-10


def test_multivariate_vs_statsmodels(pair: dict) -> None:
    import statsmodels.api as sm

    y = pair["stock_y"][:500]
    X = np.column_stack([pair["stock_x"][:500], np.arange(500.0)])
    ours = ols(y, X, intercept=True)
    sm_fit = sm.OLS(y, sm.add_constant(X)).fit()
    assert np.allclose(ours.params, sm_fit.params, atol=1e-9)
    assert np.allclose(ours.se, sm_fit.bse, atol=1e-9)
    assert ours.r2 == pytest.approx(sm_fit.rsquared, abs=1e-10)
    assert np.allclose(ours.tstats, sm_fit.tvalues, atol=1e-7)


def test_property_grid_recovers_slope() -> None:
    """Property-style check: noiseless y = a + b x recovers (a, b) exactly
    over a grid of coefficients."""
    x = np.linspace(1.0, 9.0, 60)
    for a in (-5.0, 0.0, 2.5):
        for b in (-2.0, 0.5, 10.0):
            fit = ols(a + b * x, x, intercept=True)
            assert fit.params[0] == pytest.approx(a, abs=1e-9)
            assert fit.params[1] == pytest.approx(b, abs=1e-9)


def test_error_paths() -> None:
    y = np.arange(20.0)
    with pytest.raises(ValueError, match="length mismatch"):
        ols(y, np.arange(19.0))
    with pytest.raises(ValueError, match="finite"):
        ols(np.array([1.0, np.nan, 2.0, 3.0]), np.arange(4.0))
    with pytest.raises(ValueError, match="constant"):
        ols(np.full(20, 3.0), y)  # constant response
    with pytest.raises(ValueError):
        ols(y, np.full(20, 7.0))  # constant regressor + intercept
    with pytest.raises(ValueError, match="observations"):
        ols(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_near_perfect_collinearity_raises() -> None:
    rng = np.random.default_rng(3)
    x1 = rng.normal(0.0, 1.0, 200)
    x2 = x1 * (1.0 + 1e-14) + 1e-13
    y = x1 + rng.normal(0.0, 1.0, 200)
    with pytest.raises(ValueError, match="singular|collinear"):
        ols(y, np.column_stack([x1, x2]), intercept=True)
