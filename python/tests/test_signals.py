"""Rolling z-score and spread: windowing, guards, error paths."""
from __future__ import annotations

import numpy as np
import pytest

from statarb import rolling_zscore, spread


def test_zscore_matches_manual_window() -> None:
    rng = np.random.default_rng(10)
    s = rng.normal(0.0, 1.0, 200)
    W = 30
    z = rolling_zscore(s, W)
    assert np.all(np.isnan(z[: W - 1]))
    for t in (W - 1, 100, 199):
        w = s[t - W + 1 : t + 1]
        expected = (s[t] - w.mean()) / w.std(ddof=1)
        assert z[t] == pytest.approx(expected, abs=1e-12)


def test_zero_variance_window_guard() -> None:
    s = np.concatenate([np.full(50, 2.0), np.array([2.0, 3.0, 4.0])])
    z = rolling_zscore(s, 10)
    assert np.isnan(z[49])            # flat window -> NaN, not inf
    assert np.isfinite(z[52])


def test_window_errors() -> None:
    s = np.arange(20.0)
    with pytest.raises(ValueError, match="larger than series"):
        rolling_zscore(s, 21)
    with pytest.raises(ValueError, match=">= 2"):
        rolling_zscore(s, 1)
    with pytest.raises(ValueError, match="finite"):
        rolling_zscore(np.array([1.0, np.nan] * 10), 4)


def test_spread_definition_and_errors(pair: dict) -> None:
    y, x = pair["stock_y"][:50], pair["stock_x"][:50]
    s = spread(y, x, 12.0, 1.7)
    assert np.allclose(s, y - 12.0 - 1.7 * x, atol=1e-14)
    with pytest.raises(ValueError, match="equal length"):
        spread(y, x[:-1], 0.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        spread(y, x, np.nan, 1.0)
