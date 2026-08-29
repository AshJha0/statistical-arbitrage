"""Augmented Dickey-Fuller test with embedded critical-value tables.

Regression form (PINNED, constant / no-trend case):

    dy_t = c + rho * y_{t-1} + sum_{i=1..p} phi_i * dy_{t-i} + e_t

with a FIXED augmentation lag p = ``maxlag`` (default 1; no automatic lag
selection).  The test statistic is the plain OLS t-statistic on rho.  With
T raw observations the regression uses the T - 1 - p rows for which all
lagged differences exist.

Critical values are MacKinnon-style ASYMPTOTIC values, constant no-trend
case, embedded verbatim below for the 1-variable ADF test and for the
Engle-Granger residual test with 2 and 3 variables.  The p-value is a
documented approximation: piecewise-linear interpolation of the
significance level through the three (critical value, level) knots, with
linear extrapolation using the edge-segment slopes, clamped to
[1e-4, 0.9999].  The exact MacKinnon response-surface regression is
deliberately NOT implemented.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ols import ols

#: MacKinnon-style asymptotic critical values, constant no-trend case.
#: Key = number of variables (1 = plain ADF; 2, 3 = Engle-Granger residual
#: test on a regression with that many variables).  Values = (1%, 5%, 10%).
#: PINNED — every language embeds these exact numbers.
CRITICAL_VALUES: dict[int, tuple[float, float, float]] = {
    1: (-3.43, -2.86, -2.57),
    2: (-3.90, -3.34, -3.04),
    3: (-4.29, -3.74, -3.45),
}

#: Significance levels matching the columns of CRITICAL_VALUES.
LEVELS: tuple[float, float, float] = (0.01, 0.05, 0.10)

#: Clamp bounds for the interpolated p-value.  Pinned.
PVALUE_MIN: float = 1.0e-4
PVALUE_MAX: float = 0.9999


@dataclass(frozen=True)
class ADFResult:
    """Result of an ADF unit-root test.

    Attributes:
        tstat: OLS t-statistic on rho.
        pvalue: Interpolated approximate p-value (see module docstring).
        crit: Mapping ``{"1%": cv, "5%": cv, "10%": cv}`` used.
        maxlag: Augmentation lag p used (fixed, not selected).
        nobs: Effective regression observations (T - 1 - p).
        nvars: Table selector (1 = ADF, 2/3 = EG residual case).
    """

    tstat: float
    pvalue: float
    crit: dict[str, float]
    maxlag: int
    nobs: int
    nvars: int

    def reject(self, alpha: float = 0.05) -> bool:
        """Reject the unit-root null at ``alpha`` in {0.01, 0.05, 0.10}.

        Verdict is by direct table comparison (tstat < critical value),
        NOT via the interpolated p-value.
        """
        key = {0.01: "1%", 0.05: "5%", 0.10: "10%"}.get(alpha)
        if key is None:
            raise ValueError("alpha must be one of 0.01, 0.05, 0.10")
        return self.tstat < self.crit[key]


def interpolate_pvalue(tstat: float, nvars: int) -> float:
    """Approximate p-value by piecewise-linear interpolation.

    Knots are (cv_1%, 0.01), (cv_5%, 0.05), (cv_10%, 0.10); outside the
    knot range the edge segment's slope extrapolates linearly; the result
    is clamped to [PVALUE_MIN, PVALUE_MAX].
    """
    cv1, cv5, cv10 = CRITICAL_VALUES[nvars]
    if tstat <= cv5:
        # segment through (cv1, 0.01) and (cv5, 0.05)
        slope = (0.05 - 0.01) / (cv5 - cv1)
        p = 0.01 + slope * (tstat - cv1)
    else:
        # segment through (cv5, 0.05) and (cv10, 0.10)
        slope = (0.10 - 0.05) / (cv10 - cv5)
        p = 0.05 + slope * (tstat - cv5)
    return float(min(max(p, PVALUE_MIN), PVALUE_MAX))


def adf_test(y: np.ndarray, maxlag: int = 1, nvars: int = 1) -> ADFResult:
    """Run the ADF regression and return the unit-root test result.

    Args:
        y: Series to test (levels for nvars=1; cointegrating residuals
            for nvars=2 or 3), 1-D, finite.
        maxlag: Fixed augmentation lag p >= 0 (default 1).
        nvars: Critical-value table selector: 1 for a plain ADF test,
            2 or 3 for the Engle-Granger residual test.

    Raises:
        ValueError: non-integer/negative maxlag, unknown nvars, constant
            series, non-finite input, or too few observations
            (effective nobs must be >= nparams + 10).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1-D")
    if not isinstance(maxlag, (int, np.integer)) or maxlag < 0:
        raise ValueError("maxlag must be an integer >= 0")
    if nvars not in CRITICAL_VALUES:
        raise ValueError("nvars must be 1, 2 or 3")
    if not np.all(np.isfinite(y)):
        raise ValueError("inputs must be finite (no NaN/inf)")
    t_len = y.shape[0]
    p = int(maxlag)
    nparams = 2 + p  # c, rho, phi_1..phi_p
    eff = t_len - 1 - p
    if np.ptp(y) == 0.0:
        raise ValueError("series is constant: ADF test undefined")
    if eff < nparams + 10:
        raise ValueError(
            f"series too short for ADF with maxlag={p}: effective nobs {eff}"
        )

    dy = np.diff(y)
    # rows t = p+1 .. T-1 (0-indexed) of the level series
    lhs = dy[p:]
    cols = [y[p:-1]]
    for i in range(1, p + 1):
        cols.append(dy[p - i : -i])
    X = np.column_stack(cols)
    fit = ols(lhs, X, intercept=True)
    tstat = float(fit.tstats[1])  # rho is first regressor after intercept
    cv1, cv5, cv10 = CRITICAL_VALUES[nvars]
    return ADFResult(
        tstat=tstat,
        pvalue=interpolate_pvalue(tstat, nvars),
        crit={"1%": cv1, "5%": cv5, "10%": cv10},
        maxlag=p,
        nobs=eff,
        nvars=nvars,
    )
