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
    step = KalmanStep(delta, r)
    alpha = np.empty(T)
    beta = np.empty(T)
    z = np.empty(T)
    innov = np.empty(T)
    svar = np.empty(T)

    # One recursion for batch and streaming, so the two cannot diverge.
    for t in range(T):
        alpha[t], beta[t], z[t] = step.update(x[t], y[t])
        innov[t] = step.innovation
        svar[t] = step.s_var

    return KalmanResult(
        alpha=alpha,
        beta=beta,
        z=z,
        innovation=innov,
        s_var=svar,
        final_cov=step.covariance(),
    )


class KalmanStep:
    """Streaming form of :func:`kalman_hedge`: scalar state, O(1) per update.

    Same model and prior.  :func:`kalman_hedge` loops over this class, so the
    two agree exactly here; ports must keep the scalar order of ``update``.
    """

    __slots__ = ("_q", "_r", "_a", "_b", "_p00", "_p01", "_p11",
                 "innovation", "s_var")

    def __init__(self, delta: float = 1.0e-4, r: float = 1.0e-3) -> None:
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must lie strictly inside (0, 1)")
        if not (r > 0.0 and math.isfinite(r)):
            raise ValueError("observation variance r must be > 0 and finite")
        self._q = delta / (1.0 - delta)
        self._r = float(r)
        self._a = 0.0
        self._b = 0.0
        self._p00 = P0_SCALE
        self._p01 = 0.0
        self._p11 = P0_SCALE
        self.innovation = 0.0  # e_t of the latest update
        self.s_var = 0.0       # S_t of the latest update

    def covariance(self) -> np.ndarray:
        """Current 2x2 posterior covariance."""
        return np.array([[self._p00, self._p01], [self._p01, self._p11]])

    def update(self, x: float, y: float) -> tuple[float, float, float]:
        """Filter one observation; return (alpha, beta, z).

        Raises:
            ValueError: non-finite x or y (state is left unchanged).
        """
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("inputs must be finite (no NaN/inf)")
        x = float(x)
        y = float(y)
        r = self._r
        p00 = self._p00 + self._q
        p01 = self._p01
        p11 = self._p11 + self._q
        e = y - (self._a + x * self._b)
        ph0 = p00 + p01 * x          # P_pred H'
        ph1 = p01 + p11 * x
        S = ph0 + x * ph1 + r
        k0 = ph0 / S
        k1 = ph1 / S
        self._a += k0 * e
        self._b += k1 * e
        # Joseph form with A = I - K H = [[a00, a01], [a10, a11]]
        a00 = 1.0 - k0
        a01 = -k0 * x
        a10 = -k1
        a11 = 1.0 - k1 * x
        b00 = a00 * p00 + a01 * p01  # B = A P_pred
        b01 = a00 * p01 + a01 * p11
        b10 = a10 * p00 + a11 * p01
        b11 = a10 * p01 + a11 * p11
        n00 = b00 * a00 + b01 * a01 + r * (k0 * k0)
        n01 = b00 * a10 + b01 * a11 + r * (k0 * k1)
        n10 = b10 * a00 + b11 * a01 + r * (k1 * k0)
        n11 = b10 * a10 + b11 * a11 + r * (k1 * k1)
        self._p00 = n00
        self._p01 = 0.5 * (n01 + n10)
        self._p11 = n11
        self.innovation = e
        self.s_var = S
        return self._a, self._b, e / math.sqrt(S)
