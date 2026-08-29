"""Johansen-lite: eigen ordering, rank verdicts, Jacobi vs numpy, errors."""
from __future__ import annotations

import numpy as np
import pytest

from statarb import cholesky_lower, jacobi_eigh, johansen_trace


def test_pair_trace_ordering_and_rank(pair: dict, config: dict) -> None:
    n_is = int(len(pair["stock_y"]) * config["insample_frac"])
    Y = np.column_stack([pair["stock_y"][:n_is], pair["stock_x"][:n_is]])
    res = johansen_trace(Y)
    lam = res.eigenvalues
    assert lam[0] >= lam[1] >= 0.0
    assert lam[0] < 1.0
    # trace stats are nested sums of positive terms: trace(0) > trace(1)
    assert res.trace_stats[0] > res.trace_stats[1] >= 0.0
    assert res.rank == 1  # cointegrated pair, not stationary in levels
    # cointegrating vector recovers roughly (1, -beta_true) pre-break
    b = res.beta[:, 0]
    assert b[0] == pytest.approx(1.0, abs=1e-12)
    assert -b[1] == pytest.approx(1.7, abs=0.25)


def test_spurious_pair_rank_zero(walks: dict) -> None:
    Y = np.column_stack([walks["walk_a"], walks["walk_b"]])
    assert johansen_trace(Y).rank == 0


def test_fx_triangle_three_asset(fx: dict) -> None:
    """log EURUSD - log EURGBP - log GBPUSD is stationary: rank >= 1 and
    the leading vector has signs (+, -, -) after normalisation."""
    Y = np.column_stack(
        [np.log(fx["eurusd"]), np.log(fx["gbpusd"]), np.log(fx["eurgbp"])]
    )
    res = johansen_trace(Y)
    assert res.rank >= 1
    lam = res.eigenvalues
    assert lam[0] >= lam[1] >= lam[2] >= 0.0
    b = res.beta[:, 0]
    assert b[0] == pytest.approx(1.0, abs=1e-12)
    assert b[1] == pytest.approx(-1.0, abs=0.15)
    assert b[2] == pytest.approx(-1.0, abs=0.15)


def test_jacobi_matches_numpy_eigh() -> None:
    rng = np.random.default_rng(8)
    for n in (2, 3):
        for _ in range(5):
            m = rng.normal(size=(n, n))
            a = m @ m.T  # symmetric PSD
            vals, vecs = jacobi_eigh(a)
            ref = np.sort(np.linalg.eigvalsh(a))[::-1]
            assert np.allclose(vals, ref, atol=1e-10)
            # eigenvector property A v = lambda v
            for i in range(n):
                assert np.allclose(a @ vecs[:, i], vals[i] * vecs[:, i], atol=1e-8)


def test_cholesky_native() -> None:
    a = np.array([[4.0, 2.0, 0.6], [2.0, 5.0, 1.0], [0.6, 1.0, 3.0]])
    L = cholesky_lower(a)
    assert np.allclose(L @ L.T, a, atol=1e-12)
    assert np.allclose(L, np.tril(L))
    with pytest.raises(ValueError, match="positive definite"):
        cholesky_lower(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_error_paths(pair: dict) -> None:
    y = pair["stock_y"][:100]
    x = pair["stock_x"][:100]
    with pytest.raises(ValueError, match="2 or 3"):
        johansen_trace(np.column_stack([y, x, y, x]))
    with pytest.raises(ValueError, match="2-D"):
        johansen_trace(y)
    with pytest.raises(ValueError, match="20 observations"):
        johansen_trace(np.column_stack([y[:10], x[:10]]))
    with pytest.raises(ValueError, match="constant"):
        johansen_trace(np.column_stack([y, np.full(100, 5.0)]))
    with pytest.raises(ValueError, match="pinned"):
        johansen_trace(np.column_stack([y, x]), maxlag_diff=2)
    bad = np.column_stack([y, x]).copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        johansen_trace(bad)
