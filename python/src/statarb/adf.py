"""Augmented Dickey-Fuller test with MacKinnon (2010) critical values.

Regression form (PINNED, constant / no-trend case):

    dy_t = c + rho * y_{t-1} + sum_{i=1..p} phi_i * dy_{t-i} + e_t

with a FIXED augmentation lag p = ``maxlag`` (default 1; no automatic lag
selection).  The test statistic is the plain OLS t-statistic on rho.  With
T raw observations the regression uses the T_eff = T - 1 - p rows for which
all lagged differences exist.

Critical values (PINNED) are the MacKinnon (2010) finite-sample response
surfaces for the constant / no-trend case,

    cv(T_eff) = b_inf + b_1 / T_eff + b_2 / T_eff^2 + b_3 / T_eff^3,

embedded verbatim below for N = 1 (plain ADF on one series) and for the
Engle-Granger residual test with N = 2 and 3 variables, at the 1%, 5% and
10% levels.  ``T_eff`` is the number of observations in the test
regression, exactly as statsmodels' ``adfuller`` uses it.  The surfaces
were fitted on samples of 20 or more, so ``T_eff >= 20`` is enforced.

The p-value is a documented approximation: piecewise-linear interpolation
of the significance level through the three (critical value, level) knots
at the given T_eff, with linear extrapolation using the edge-segment
slopes, clamped to [1e-4, 0.9999].  The exact MacKinnon (1994) p-value
response surface is deliberately NOT implemented; the verdict never uses
the p-value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .ols import ols

#: MacKinnon (2010) response-surface coefficients (b_inf, b_1, b_2, b_3),
#: constant / no-trend case, Table 1 of "Critical Values for Cointegration
#: Tests", Queen's Economics Dept. WP 1227.  Key = number of variables N
#: (1 = plain ADF; 2, 3 = Engle-Granger residual test on a regression with
#: that many variables); rows = (1%, 5%, 10%).  PINNED — every language
#: embeds these exact numbers.
RESPONSE_SURFACE: dict[int, tuple[tuple[float, float, float, float], ...]] = {
    1: (
        (-3.43035, -6.5393, -16.786, -79.433),
        (-2.86154, -2.8903, -4.234, -40.040),
        (-2.56677, -1.5384, -2.809, 0.0),
    ),
    2: (
        (-3.89644, -10.9519, -33.527, 0.0),
        (-3.33613, -6.1101, -6.823, 0.0),
        (-3.04445, -4.2412, -2.720, 0.0),
    ),
    3: (
        (-4.29374, -14.4354, -33.195, 47.433),
        (-3.74066, -8.5632, -10.852, 27.982),
        (-3.45218, -6.2143, -3.718, 0.0),
    ),
}

#: Asymptotic (T -> inf) critical values b_inf, (1%, 5%, 10%) per N.
#: Kept for callers that want the limiting table; ``adf_test`` always uses
#: the finite-sample values from :func:`critical_values`.
CRITICAL_VALUES: dict[int, tuple[float, float, float]] = {
    n: tuple(row[0] for row in rows) for n, rows in RESPONSE_SURFACE.items()
}

#: Significance levels matching the columns of the tables.
LEVELS: tuple[float, float, float] = (0.01, 0.05, 0.10)

#: Smallest effective sample the response surfaces are trusted for.  Pinned.
MIN_NOBS: int = 20

#: Largest augmentation lag accepted (documented upper bound; the series
#: must be far longer than this anyway).  Pinned.
MAX_LAG: int = 1000

#: Clamp bounds for the interpolated p-value.  Pinned.
PVALUE_MIN: float = 1.0e-4
PVALUE_MAX: float = 0.9999


def critical_values(nvars: int, nobs: int | None = None) -> tuple[float, float, float]:
    """MacKinnon (2010) critical values (1%, 5%, 10%) for ``nvars`` variables.

    Args:
        nvars: 1 for the plain ADF test, 2 or 3 for the Engle-Granger
            residual test.
        nobs: Effective regression sample size T_eff; ``None`` returns the
            asymptotic values (``b_inf``).

    Raises:
        ValueError: ``nvars`` outside {1, 2, 3} or ``nobs`` below MIN_NOBS.
    """
    if nvars not in RESPONSE_SURFACE:
        raise ValueError("nvars must be 1, 2 or 3")
    if nobs is None:
        return CRITICAL_VALUES[nvars]
    if not isinstance(nobs, (int, np.integer)) or nobs < MIN_NOBS:
        raise ValueError(
            f"critical values need at least {MIN_NOBS} effective observations"
        )
    inv = 1.0 / float(nobs)
    out = []
    for b_inf, b1, b2, b3 in RESPONSE_SURFACE[nvars]:
        out.append(b_inf + inv * (b1 + inv * (b2 + inv * b3)))
    return (out[0], out[1], out[2])


@dataclass(frozen=True)
class ADFResult:
    """Result of an ADF unit-root test.

    Attributes:
        tstat: OLS t-statistic on rho.
        pvalue: Interpolated approximate p-value (see module docstring).
        crit: Mapping ``{"1%": cv, "5%": cv, "10%": cv}`` actually used for
            the verdict — the MacKinnon (2010) values at this ``nobs``.
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


def interpolate_pvalue(tstat: float, nvars: int, nobs: int | None = None) -> float:
    """Approximate p-value by piecewise-linear interpolation.

    Knots are (cv_1%, 0.01), (cv_5%, 0.05), (cv_10%, 0.10) at sample size
    ``nobs`` (asymptotic when ``None``); outside the knot range the edge
    segment's slope extrapolates linearly; the result is clamped to
    [PVALUE_MIN, PVALUE_MAX].

    Raises:
        ValueError: ``nvars`` outside {1, 2, 3}, NaN ``tstat``, or ``nobs``
            below MIN_NOBS.  (An infinite t-stat — an exact-fit regression
            — is allowed and clamps to the bound on its side.)
    """
    cv1, cv5, cv10 = critical_values(nvars, nobs)
    if math.isnan(tstat):
        raise ValueError("tstat must not be NaN")
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
        maxlag: Fixed augmentation lag p, integer in [0, MAX_LAG] (default 1).
        nvars: Critical-value table selector: 1 for a plain ADF test,
            2 or 3 for the Engle-Granger residual test.

    Raises:
        ValueError: non-integer/negative/oversized maxlag, unknown nvars,
            constant series, non-finite input, or too few observations
            (effective nobs must be >= max(nparams + 10, MIN_NOBS)).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1-D")
    if not isinstance(maxlag, (int, np.integer)) or maxlag < 0:
        raise ValueError("maxlag must be an integer >= 0")
    if maxlag > MAX_LAG:
        raise ValueError(f"maxlag must be <= {MAX_LAG}")
    if nvars not in RESPONSE_SURFACE:
        raise ValueError("nvars must be 1, 2 or 3")
    if not np.all(np.isfinite(y)):
        raise ValueError("inputs must be finite (no NaN/inf)")
    t_len = y.shape[0]
    p = int(maxlag)
    nparams = 2 + p  # c, rho, phi_1..phi_p
    eff = t_len - 1 - p
    if t_len > 0 and np.ptp(y) == 0.0:
        raise ValueError("series is constant: ADF test undefined")
    if eff < max(nparams + 10, MIN_NOBS):
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
    cv1, cv5, cv10 = critical_values(nvars, eff)
    return ADFResult(
        tstat=tstat,
        pvalue=interpolate_pvalue(tstat, nvars, eff),
        crit={"1%": cv1, "5%": cv5, "10%": cv10},
        maxlag=p,
        nobs=eff,
        nvars=nvars,
    )
