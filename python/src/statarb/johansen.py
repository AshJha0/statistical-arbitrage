"""Johansen-lite trace test for 2- and 3-asset systems.

Canonical correlation setup (PINNED).  VECM with unrestricted constant and
k = 1 lagged difference:

    dY_t = c + Gamma * dY_{t-1} + Pi * Y_{t-1} + e_t

Steps (T raw observations, n = 2 or 3 assets, T_eff = T - 2 usable rows,
t = 2..T-1 zero-indexed):

1. Z0 = dY_t, Z1 = Y_{t-1}, ZD = [1, dY_{t-1}]  (partialled-out block).
2. R0 = residuals of Z0 on ZD; R1 = residuals of Z1 on ZD (multivariate
   OLS via normal equations, one solve per system).
3. Product-moment matrices S_ij = R_i' R_j / T_eff for ij in
   {00, 01, 10, 11}.
4. Eigenvalue problem  |lambda * S11 - S10 * S00^{-1} * S01| = 0, made
   SYMMETRIC via the Cholesky factor S11 = L L':

       A = L^{-1} S10 S00^{-1} S01 L^{-T}   (symmetric PSD, n x n)

   Eigen-decomposition of A by the cyclic JACOBI rotation algorithm
   implemented natively below (pinned — no library eigensolver).
   Eigenvalues sorted descending; cointegrating vectors beta_i = L^{-T} v_i,
   normalised so the first component is +1.
5. Trace statistics  trace(r) = -T_eff * sum_{i=r+1..n} ln(1 - lambda_i),
   compared against the embedded Osterwald-Lenum (1992) 95% trace table
   for the unrestricted-constant case.  The estimated rank is the smallest
   r with trace(r) < cv(n - r), else n.

Honest limitations: this is a fixed-lag (k=1), fixed-deterministic-case
implementation with only the 95% column of the critical-value table; no
small-sample correction, no maximum-eigenvalue variant, no p-values.
It is meant for 2-3 asset pairs-trading diagnostics, not general VECM work.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Osterwald-Lenum (1992) 95% critical values for the TRACE statistic,
#: model with unrestricted constant.  Index by (n - r): 1 -> 3.76,
#: 2 -> 15.41, 3 -> 29.68.  PINNED.
TRACE_CRIT_95: dict[int, float] = {1: 3.76, 2: 15.41, 3: 29.68}

#: Jacobi convergence threshold on the off-diagonal Frobenius norm and
#: maximum number of sweeps.  Pinned.
JACOBI_TOL: float = 1.0e-14
JACOBI_MAX_SWEEPS: int = 100


def cholesky_lower(a: np.ndarray) -> np.ndarray:
    """Native lower-triangular Cholesky factor of a small SPD matrix.

    Standard column-by-column algorithm; raises ValueError when a pivot
    is not strictly positive (matrix not positive definite).
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    L = np.zeros((n, n))
    for j in range(n):
        d = a[j, j] - float(L[j, :j] @ L[j, :j])
        if d <= 0.0:
            raise ValueError("matrix is not positive definite (Cholesky pivot <= 0)")
        L[j, j] = np.sqrt(d)
        for i in range(j + 1, n):
            L[i, j] = (a[i, j] - float(L[i, :j] @ L[j, :j])) / L[j, j]
    return L


def jacobi_eigh(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations.

    Pinned algorithm: sweep the upper triangle in row-major order; for each
    off-diagonal element with |a_pq| > JACOBI_TOL apply the classical
    rotation (theta = (a_qq - a_pp) / (2 a_pq),
    t = sign(theta) / (|theta| + sqrt(theta^2 + 1)), c = 1/sqrt(t^2+1),
    s = t c); stop when the off-diagonal Frobenius norm falls below
    JACOBI_TOL or after JACOBI_MAX_SWEEPS sweeps.

    Returns:
        (eigenvalues descending, eigenvector matrix with matching columns).
    """
    A = np.array(a, dtype=float, copy=True)
    n = A.shape[0]
    if A.shape != (n, n) or not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("jacobi_eigh requires a symmetric square matrix")
    V = np.eye(n)
    for _ in range(JACOBI_MAX_SWEEPS):
        off = np.sqrt(np.sum(np.tril(A, -1) ** 2) * 2.0)
        if off < JACOBI_TOL:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p, q]
                if abs(apq) <= JACOBI_TOL:
                    continue
                theta = (A[q, q] - A[p, p]) / (2.0 * apq)
                t = np.sign(theta) / (abs(theta) + np.sqrt(theta * theta + 1.0))
                if theta == 0.0:
                    t = 1.0
                c = 1.0 / np.sqrt(t * t + 1.0)
                s = t * c
                J = np.eye(n)
                J[p, p] = c
                J[q, q] = c
                J[p, q] = s
                J[q, p] = -s
                A = J.T @ A @ J
                V = V @ J
    vals = np.diag(A).copy()
    order = np.argsort(vals)[::-1]
    return vals[order], V[:, order]


@dataclass(frozen=True)
class JohansenResult:
    """Johansen-lite trace-test result.

    Attributes:
        eigenvalues: Canonical correlations lambda_i, descending, in [0, 1).
        trace_stats: trace(r) for r = 0..n-1.
        crit_95: Matching Osterwald-Lenum 95% critical values.
        rank: Estimated cointegration rank at 95%.
        beta: (n, n) matrix; column i is the cointegrating vector for
            lambda_i, normalised to first component +1.
        t_eff: Effective sample size T - 2.
    """

    eigenvalues: np.ndarray
    trace_stats: np.ndarray
    crit_95: np.ndarray
    rank: int
    beta: np.ndarray
    t_eff: int


def _residualise(z: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Residuals of each column of z on the common design d (with solve)."""
    dtd = d.T @ d
    if np.linalg.cond(dtd) > 1.0e12:
        raise ValueError("partialled-out design is (near-)singular")
    coef = np.linalg.solve(dtd, d.T @ z)
    return z - d @ coef


def johansen_trace(levels: np.ndarray, maxlag_diff: int = 1) -> JohansenResult:
    """Run the Johansen-lite trace test on an (T, n) matrix of levels.

    Args:
        levels: Price levels, shape (T, n) with n in {2, 3}, T >= 20.
        maxlag_diff: Number of lagged differences k; only k=1 is
            implemented (pinned) — any other value raises.

    Raises:
        ValueError: wrong shape, n not in {2,3}, T < 20, non-finite
            values, constant column, or maxlag_diff != 1.
    """
    Y = np.asarray(levels, dtype=float)
    if Y.ndim != 2:
        raise ValueError("levels must be a 2-D (T, n) array")
    T, n = Y.shape
    if n not in (2, 3):
        raise ValueError("johansen_trace supports n = 2 or 3 assets only")
    if maxlag_diff != 1:
        raise ValueError("only maxlag_diff = 1 is implemented (pinned)")
    if T < 20:
        raise ValueError("need at least 20 observations")
    if not np.all(np.isfinite(Y)):
        raise ValueError("inputs must be finite (no NaN/inf)")
    if np.any(np.ptp(Y, axis=0) == 0.0):
        raise ValueError("constant column in levels")

    dY = np.diff(Y, axis=0)          # length T-1
    Z0 = dY[1:]                      # dY_t,   t = 2..T-1
    Z1 = Y[1 : T - 1]                # Y_{t-1}
    ZD = np.column_stack([np.ones(T - 2), dY[:-1]])  # [1, dY_{t-1}]
    t_eff = T - 2

    R0 = _residualise(Z0, ZD)
    R1 = _residualise(Z1, ZD)
    S00 = R0.T @ R0 / t_eff
    S01 = R0.T @ R1 / t_eff
    S10 = S01.T
    S11 = R1.T @ R1 / t_eff

    L = cholesky_lower(S11)
    M = S10 @ np.linalg.solve(S00, S01)      # S10 S00^{-1} S01
    Linv = np.linalg.solve(L, np.eye(n))     # L^{-1}
    A = Linv @ M @ Linv.T
    A = 0.5 * (A + A.T)                      # kill rounding asymmetry
    lam, V = jacobi_eigh(A)
    lam = np.clip(lam, 0.0, 1.0 - 1.0e-12)
    beta = Linv.T @ V                        # beta_i = L^{-T} v_i
    for i in range(n):
        if abs(beta[0, i]) > 1.0e-12:
            beta[:, i] = beta[:, i] / beta[0, i]

    trace = np.array(
        [-t_eff * float(np.sum(np.log(1.0 - lam[r:]))) for r in range(n)]
    )
    crit = np.array([TRACE_CRIT_95[n - r] for r in range(n)])
    rank = n
    for r in range(n):
        if trace[r] < crit[r]:
            rank = r
            break
    return JohansenResult(
        eigenvalues=lam,
        trace_stats=trace,
        crit_95=crit,
        rank=rank,
        beta=beta,
        t_eff=t_eff,
    )
