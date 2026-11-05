"""Engle-Granger two-step cointegration test.

Step 1: static OLS hedge regression  y_t = a + b * x_t + e_t  (normal
equations, see :mod:`statarb.ols`).
Step 2: ADF test (constant, fixed maxlag) on the residuals e_t, judged
against the Engle-Granger 2-variable critical-value table
(:data:`statarb.adf.CRITICAL_VALUES` with ``nvars=2``) — the residuals are
estimated, so the plain Dickey-Fuller table would over-reject.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adf import ADFResult, adf_test
from .ols import ols


@dataclass(frozen=True)
class EGResult:
    """Engle-Granger two-step result.

    Attributes:
        alpha: Intercept of the hedge regression.
        beta: Hedge ratio b.
        resid: Cointegrating residuals ``y - a - b x``.
        r2: R^2 of the hedge regression.
        adf: ADF result on the residuals (nvars=2 table).
    """

    alpha: float
    beta: float
    resid: np.ndarray
    r2: float
    adf: ADFResult

    def cointegrated(self, alpha: float = 0.05) -> bool:
        """True when the residual ADF rejects a unit root at ``alpha``."""
        return self.adf.reject(alpha)


def engle_granger(y: np.ndarray, x: np.ndarray, maxlag: int = 1) -> EGResult:
    """Run the Engle-Granger two-step test of cointegration of y on x.

    Args:
        y: Dependent price series (1-D).
        x: Regressor price series, same length.
        maxlag: Fixed ADF augmentation lag (default 1).

    Raises:
        ValueError: propagated from OLS/ADF validation (length mismatch,
            constant series, non-finite values, short series).
    """
    fit = ols(y, x, intercept=True)
    adf = adf_test(fit.resid, maxlag=maxlag, nvars=2)
    return EGResult(
        alpha=float(fit.params[0]),
        beta=float(fit.params[1]),
        resid=fit.resid,
        r2=fit.r2,
        adf=adf,
    )


@dataclass(frozen=True)
class EGBothResult:
    """Engle-Granger run in both directions (the test is not symmetric)."""

    y_on_x: EGResult
    x_on_y: EGResult

    def cointegrated(self, alpha: float = 0.05) -> bool:
        """True only when BOTH directions reject at ``alpha``."""
        return self.y_on_x.cointegrated(alpha) and self.x_on_y.cointegrated(alpha)


def engle_granger_both(y: np.ndarray, x: np.ndarray, maxlag: int = 1) -> EGBothResult:
    """Run :func:`engle_granger` as y on x and as x on y."""
    return EGBothResult(y_on_x=engle_granger(y, x, maxlag),
                        x_on_y=engle_granger(x, y, maxlag))
