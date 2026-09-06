"""Kalman-filter dynamic hedge ratio with Joseph-form covariance update.

Model (PINNED):

    state       s_t = (alpha_t, beta_t)',  s_t = s_{t-1} + w_t  (random walk)
    observation y_t = H_t s_t + eps_t,     H_t = [1, x_t]
    Q = (delta / (1 - delta)) * I_2        (RiskMetrics-style scaling)
    R = r                                  (scalar observation variance)
    s_0 = (0, 0)',  P_0 = p0 * I_2         (diffuse-ish prior, p0 = 1e3)

Recursion per observation t = 0..T-1:

    predict:  s_pred = s_prev,             P_pred = P_prev + Q
    innovate: e_t = y_t - H_t s_pred,      S_t = H_t P_pred H_t' + r
    gain:     K_t = P_pred H_t' / S_t
    update:   s_t = s_pred + K_t e_t
              P_t = (I - K_t H_t) P_pred (I - K_t H_t)' + K_t r K_t'

The covariance update is the JOSEPH form (pinned): it is algebraically
identical to (I - K H) P_pred but remains symmetric positive semi-definite
under rounding, which the plain form does not guarantee.

Innovation z-scores z_t = e_t / sqrt(S_t) provide an alternative trading
signal to the rolling z-score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Pinned prior covariance scale P_0 = P0_SCALE * I.
P0_SCALE: float = 1.0e3


@dataclass(frozen=True)
class KalmanResult:
    """Filtered dynamic hedge output.

    Attributes:
        alpha: Filtered alpha_t path, length T.
        beta: Filtered beta_t path, length T.
        z: Innovation z-scores e_t / sqrt(S_t), length T.
        innovation: Raw innovations e_t.
        s_var: Innovation variances S_t.
        final_cov: Final 2x2 posterior covariance P_{T-1}.
    """

    alpha: np.ndarray
    beta: np.ndarray
    z: np.ndarray
    innovation: np.ndarray
    s_var: np.ndarray
    final_cov: np.ndarray


def kalman_hedge(
    y: np.ndarray,
    x: np.ndarray,
    delta: float = 1.0e-4,
    r: float = 1.0e-3,
) -> KalmanResult:
    """Filter the dynamic hedge state (alpha_t, beta_t) through (y, x).

    Args:
        y: Dependent price series, 1-D.
        x: Hedge instrument price series, same length.
        delta: State-noise discount in the OPEN interval (0, 1);
            Q = delta/(1-delta) * I.  delta -> 0 recovers recursive least
            squares (static OLS in the limit).
        r: Observation noise variance, > 0.

    Raises:
        ValueError: length mismatch, fewer than 2 observations,
            non-finite input, delta outside the open interval (0, 1), or
            r not a finite number > 0 (NaN and +inf are rejected).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("y and x must be 1-D")
    if y.shape[0] != x.shape[0]:
        raise ValueError(
            f"series length mismatch: len(y)={y.shape[0]}, len(x)={x.shape[0]}"
        )
    if y.shape[0] < 2:
        raise ValueError("need at least 2 observations")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
        raise ValueError("inputs must be finite (no NaN/inf)")
    # Written as negated comparisons so NaN fails (NaN compares False).
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie strictly inside (0, 1)")
    if not (r > 0.0 and math.isfinite(r)):
        raise ValueError("observation variance r must be > 0 and finite")

    T = y.shape[0]
    q = delta / (1.0 - delta)
    Q = q * np.eye(2)
    I2 = np.eye(2)
    s = np.zeros(2)
    P = P0_SCALE * np.eye(2)

    alpha = np.empty(T)
    beta = np.empty(T)
    z = np.empty(T)
    innov = np.empty(T)
    svar = np.empty(T)

    for t in range(T):
        P_pred = P + Q
        h = np.array([1.0, x[t]])
        e = y[t] - float(h @ s)
        S = float(h @ P_pred @ h) + r
        K = (P_pred @ h) / S
        s = s + K * e
        ikh = I2 - np.outer(K, h)
        P = ikh @ P_pred @ ikh.T + r * np.outer(K, K)  # Joseph form
        P = 0.5 * (P + P.T)
        alpha[t] = s[0]
        beta[t] = s[1]
        innov[t] = e
        svar[t] = S
        z[t] = e / np.sqrt(S)

    return KalmanResult(
        alpha=alpha,
        beta=beta,
        z=z,
        innovation=innov,
        s_var=svar,
        final_cov=P,
    )
