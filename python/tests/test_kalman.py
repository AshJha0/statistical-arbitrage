"""Kalman dynamic hedge: RLS limit, PSD covariance, adaptation, errors."""
from __future__ import annotations

import numpy as np
import pytest

from statarb import kalman_hedge, ols


def test_delta_to_zero_converges_to_ols(pair: dict, config: dict) -> None:
    """With delta -> 0 the filter is recursive least squares and the final
    beta approaches the static OLS beta (loose band; P0 is finite)."""
    n_is = int(len(pair["stock_y"]) * config["insample_frac"])
    y = pair["stock_y"][:n_is]
    x = pair["stock_x"][:n_is]
    beta_ols = float(ols(y, x, intercept=True).params[1])
    km = kalman_hedge(y, x, delta=1e-16, r=1e-3)
    assert km.beta[-1] == pytest.approx(beta_ols, abs=0.05)


def test_joseph_form_covariance_stays_psd(pair: dict) -> None:
    """2x2 closed-form eigen check: PSD iff trace >= 0 and det >= 0.
    Checked on the final posterior and along a stressed run (tiny r)."""
    for r in (1e-3, 1e-8):
        km = kalman_hedge(pair["stock_y"], pair["stock_x"], delta=1e-4, r=r)
        P = km.final_cov
        tr = P[0, 0] + P[1, 1]
        det = P[0, 0] * P[1, 1] - P[0, 1] * P[1, 0]
        assert P[0, 1] == pytest.approx(P[1, 0], abs=1e-15)  # symmetric
        assert tr >= -1e-15
        assert det >= -1e-12 * max(tr * tr, 1.0)
        # closed-form 2x2 eigenvalues both >= 0 (up to rounding)
        disc = np.sqrt(max((tr / 2.0) ** 2 - det, 0.0))
        assert tr / 2.0 - disc >= -1e-10


def test_tracks_beta_through_break(pair: dict, config: dict) -> None:
    """Kalman adapts to the post-break beta ~1.1; static OLS cannot."""
    y, x = pair["stock_y"], pair["stock_x"]
    km = kalman_hedge(y, x, delta=config["kalman_delta"], r=config["kalman_r"])
    assert km.beta[500] == pytest.approx(1.7, abs=0.1)   # pre-break
    assert km.beta[-1] == pytest.approx(1.1, abs=0.1)    # post-break
    beta_ols = float(ols(y, x, intercept=True).params[1])
    # Kalman tracking error vs the piecewise-true beta beats static OLS
    true_beta = np.full(len(y), 1.7)
    true_beta[1050:1081] = np.linspace(1.7, 1.1, 31)
    true_beta[1081:] = 1.1
    te_kalman = float(np.sqrt(np.mean((km.beta[300:] - true_beta[300:]) ** 2)))
    te_ols = float(np.sqrt(np.mean((beta_ols - true_beta[300:]) ** 2)))
    assert te_kalman < te_ols


def test_innovation_zscores_finite_and_standardised() -> None:
    rng = np.random.default_rng(9)
    x = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 600))
    y = 5.0 + 1.5 * x + rng.normal(0.0, 0.5, 600)
    km = kalman_hedge(y, x, delta=1e-4, r=0.25)
    assert np.all(np.isfinite(km.z))
    # after burn-in the innovations should be roughly standardised
    assert 0.2 < float(np.std(km.z[100:])) < 5.0
    assert np.allclose(km.z, km.innovation / np.sqrt(km.s_var), atol=1e-12)


def test_error_paths() -> None:
    y = np.arange(10.0)
    x = np.arange(10.0) + 1.0
    for bad_delta in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="delta"):
            kalman_hedge(y, x, delta=bad_delta)
    with pytest.raises(ValueError, match="r must be"):
        kalman_hedge(y, x, r=0.0)
    with pytest.raises(ValueError, match="length mismatch"):
        kalman_hedge(y, x[:-1])
    with pytest.raises(ValueError, match="at least 2"):
        kalman_hedge(np.array([1.0]), np.array([2.0]))
    with pytest.raises(ValueError, match="finite"):
        kalman_hedge(np.array([1.0, np.nan, 3.0]), np.arange(3.0))
