"""Ordinary least squares — pinned solution method.

The solution method is PINNED for cross-language reproducibility:

    b = solve(X'X, X'y)          (normal equations, LU solve)
    cov(b) = sigma2 * inv(X'X),  sigma2 = RSS / (n - k)

No QR, no SVD.  The design matrices in this project are tiny (k <= 4) and
well-conditioned, so the normal equations lose no meaningful precision.
A conditioning guard (cond(X'X) > 1e12) rejects (near-)collinear designs
with a ValueError rather than returning garbage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Condition-number threshold on X'X above which the design is treated as
#: collinear.  Pinned.
COND_LIMIT: float = 1.0e12


@dataclass(frozen=True)
class OLSResult:
    """Result of an OLS fit.

    Attributes:
        params: Coefficient vector, intercept first when ``intercept=True``.
        se: Plain (non-robust) standard errors, same order as ``params``.
        tstats: ``params / se`` elementwise.
        resid: Residual vector ``y - X b`` (length ``nobs``).
        r2: Coefficient of determination.  Centered total sum of squares
            when an intercept is included, uncentered otherwise.
        sigma2: Residual variance estimate ``RSS / (nobs - nparams)``.
        nobs: Number of observations.
        nparams: Number of estimated coefficients (including intercept).
    """

    params: np.ndarray
    se: np.ndarray
    tstats: np.ndarray
    resid: np.ndarray
    r2: float
    sigma2: float
    nobs: int
    nparams: int


def _as_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("regressor array must be 1-D or 2-D")
    return x


def ols(y: np.ndarray, x: np.ndarray, intercept: bool = True) -> OLSResult:
    """Fit ``y = X b + e`` by ordinary least squares (normal equations).

    Args:
        y: Response, 1-D array of length n.
        x: Regressors, shape (n,) or (n, m).  The intercept column is
            prepended automatically when ``intercept`` is True; do not
            include one yourself.
        intercept: Include a constant term (default True).

    Returns:
        OLSResult with intercept coefficient first (when requested).

    Raises:
        ValueError: on length mismatch, non-finite input, too few
            observations (n <= k), constant response with intercept
            (undefined R^2), or (near-)collinear design
            (cond(X'X) > 1e12 — this also catches a constant regressor
            when an intercept is present).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1-D")
    xm = _as_matrix(x)
    n = y.shape[0]
    if xm.shape[0] != n:
        raise ValueError(
            f"series length mismatch: len(y)={n}, len(x)={xm.shape[0]}"
        )
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(xm))):
        raise ValueError("inputs must be finite (no NaN/inf)")

    if intercept:
        X = np.column_stack([np.ones(n), xm])
    else:
        X = xm
    k = X.shape[1]
    if n <= k:
        raise ValueError(f"need more observations than parameters: n={n}, k={k}")

    xtx = X.T @ X
    if np.linalg.cond(xtx) > COND_LIMIT:
        raise ValueError(
            "design matrix is (near-)singular: constant or collinear regressors"
        )
    xty = X.T @ y
    params = np.linalg.solve(xtx, xty)
    resid = y - X @ params
    rss = float(resid @ resid)
    sigma2 = rss / (n - k)
    cov = sigma2 * np.linalg.inv(xtx)
    se = np.sqrt(np.diag(cov))

    if intercept:
        tss = float(np.sum((y - y.mean()) ** 2))
    else:
        tss = float(y @ y)
    if tss <= 0.0:
        raise ValueError("y is constant: R^2 undefined")
    r2 = 1.0 - rss / tss

    with np.errstate(divide="ignore", invalid="ignore"):
        tstats = params / se
    return OLSResult(
        params=params,
        se=se,
        tstats=tstats,
        resid=resid,
        r2=r2,
        sigma2=sigma2,
        nobs=n,
        nparams=k,
    )
