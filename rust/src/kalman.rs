//! Kalman-filter dynamic hedge ratio with Joseph-form covariance update.
//!
//! Model (PINNED):
//!
//! ```text
//! state       s_t = (alpha_t, beta_t)',  s_t = s_{t-1} + w_t  (random walk)
//! observation y_t = H_t s_t + eps_t,     H_t = [1, x_t]
//! Q = (delta / (1 - delta)) * I_2        (RiskMetrics-style scaling)
//! R = r                                  (scalar observation variance)
//! s_0 = (0, 0)',  P_0 = 1e3 * I_2        (diffuse-ish prior)
//! ```
//!
//! Recursion per observation t = 0..T-1 (the very first observation is
//! already filtered against the prior):
//!
//! ```text
//! predict:  s_pred = s_prev,             P_pred = P_prev + Q
//! innovate: e_t = y_t - H_t s_pred,      S_t = H_t P_pred H_t' + r
//! gain:     K_t = P_pred H_t' / S_t
//! update:   s_t = s_pred + K_t e_t
//!           P_t = (I - K_t H_t) P_pred (I - K_t H_t)' + K_t r K_t'
//! ```
//!
//! The covariance update is the JOSEPH form (pinned): algebraically
//! identical to `(I - K H) P_pred` but it stays symmetric positive
//! semi-definite under rounding, which the plain form does not guarantee.
//! `delta -> 0` recovers recursive least squares (static OLS in the
//! limit). Innovation z-scores `z_t = e_t / sqrt(S_t)` provide an
//! alternative trading signal to the rolling z-score.

use crate::error::{Result, StatarbError};

/// Pinned prior covariance scale `P_0 = P0_SCALE * I`.
pub const P0_SCALE: f64 = 1.0e3;

/// Filtered dynamic hedge output.
#[derive(Debug, Clone)]
pub struct KalmanResult {
    /// Filtered `alpha_t` path, length T.
    pub alpha: Vec<f64>,
    /// Filtered `beta_t` path, length T.
    pub beta: Vec<f64>,
    /// Innovation z-scores `e_t / sqrt(S_t)`, length T.
    pub z: Vec<f64>,
    /// Raw innovations `e_t`.
    pub innovation: Vec<f64>,
    /// Innovation variances `S_t`.
    pub s_var: Vec<f64>,
    /// Final 2x2 posterior covariance `P_{T-1}` (row-major).
    pub final_cov: [[f64; 2]; 2],
}

/// Filter the dynamic hedge state `(alpha_t, beta_t)` through `(y, x)`.
///
/// * `delta` — state-noise discount in the OPEN interval (0, 1);
///   `Q = delta/(1-delta) * I`.
/// * `r` — observation noise variance, `> 0`.
///
/// # Errors
///
/// Length mismatch, fewer than 2 observations, non-finite input,
/// `delta` outside the open interval (0, 1), or `r` not a finite number
/// `> 0` (NaN and +inf are rejected).
pub fn kalman_hedge(y: &[f64], x: &[f64], delta: f64, r: f64) -> Result<KalmanResult> {
    if y.len() != x.len() {
        return Err(StatarbError::LengthMismatch(format!(
            "series length mismatch: len(y)={}, len(x)={}",
            y.len(),
            x.len()
        )));
    }
    if y.len() < 2 {
        return Err(StatarbError::TooShort(
            "need at least 2 observations".to_string(),
        ));
    }
    if !y.iter().chain(x.iter()).all(|v| v.is_finite()) {
        return Err(StatarbError::NonFinite(
            "inputs must be finite (no NaN/inf)".to_string(),
        ));
    }
    if !(delta > 0.0 && delta < 1.0) {
        return Err(StatarbError::InvalidParam(
            "delta must lie strictly inside (0, 1)".to_string(),
        ));
    }
    // Negated comparison so NaN fails; +inf is rejected explicitly.
    if !(r > 0.0 && r.is_finite()) {
        return Err(StatarbError::InvalidParam(
            "observation variance r must be > 0 and finite".to_string(),
        ));
    }

    let t_len = y.len();
    let mut step = KalmanStep::new(delta, r)?;
    let mut alpha = vec![0.0; t_len];
    let mut beta = vec![0.0; t_len];
    let mut z = vec![0.0; t_len];
    let mut innovation = vec![0.0; t_len];
    let mut s_var = vec![0.0; t_len];

    // One recursion for batch and streaming, so the two cannot diverge.
    for t in 0..t_len {
        let (a, b, zt) = step.update(x[t], y[t])?;
        alpha[t] = a;
        beta[t] = b;
        z[t] = zt;
        innovation[t] = step.innovation();
        s_var[t] = step.s_var();
    }

    Ok(KalmanResult {
        alpha,
        beta,
        z,
        innovation,
        s_var,
        final_cov: step.covariance(),
    })
}

/// Streaming form of [`kalman_hedge`]: scalar state, O(1) per update and
/// no allocation. [`kalman_hedge`] loops over this type; the scalar order
/// of [`KalmanStep::update`] is pinned in API_SPEC section 6.
#[derive(Debug, Clone)]
pub struct KalmanStep {
    q: f64,
    r: f64,
    a: f64,
    b: f64,
    p00: f64,
    p01: f64,
    p11: f64,
    innovation: f64,
    s_var: f64,
}

impl KalmanStep {
    /// New filter at the pinned prior.
    ///
    /// # Errors
    ///
    /// `delta` outside the open interval (0, 1) or `r` not finite and > 0.
    pub fn new(delta: f64, r: f64) -> Result<Self> {
        if !(delta > 0.0 && delta < 1.0) {
            return Err(StatarbError::InvalidParam(
                "delta must lie strictly inside (0, 1)".to_string(),
            ));
        }
        // Negated comparison so NaN fails; +inf is rejected explicitly.
        if !(r > 0.0 && r.is_finite()) {
            return Err(StatarbError::InvalidParam(
                "observation variance r must be > 0 and finite".to_string(),
            ));
        }
        Ok(KalmanStep {
            q: delta / (1.0 - delta),
            r,
            a: 0.0,
            b: 0.0,
            p00: P0_SCALE,
            p01: 0.0,
            p11: P0_SCALE,
            innovation: 0.0,
            s_var: 0.0,
        })
    }

    /// Innovation `e_t` of the latest update.
    pub fn innovation(&self) -> f64 {
        self.innovation
    }

    /// Innovation variance `S_t` of the latest update.
    pub fn s_var(&self) -> f64 {
        self.s_var
    }

    /// Current 2x2 posterior covariance (row-major).
    pub fn covariance(&self) -> [[f64; 2]; 2] {
        [[self.p00, self.p01], [self.p01, self.p11]]
    }

    /// Filter one observation (note the order: `x` first); returns
    /// `(alpha, beta, z)`.
    ///
    /// # Errors
    ///
    /// Non-finite `x` or `y` (the state is left unchanged).
    pub fn update(&mut self, x: f64, y: f64) -> Result<(f64, f64, f64)> {
        if !(x.is_finite() && y.is_finite()) {
            return Err(StatarbError::NonFinite(
                "inputs must be finite (no NaN/inf)".to_string(),
            ));
        }
        let r = self.r;
        let p00 = self.p00 + self.q;
        let p01 = self.p01;
        let p11 = self.p11 + self.q;
        let e = y - (self.a + x * self.b);
        let ph0 = p00 + p01 * x; // P_pred H'
        let ph1 = p01 + p11 * x;
        let s = ph0 + x * ph1 + r;
        let k0 = ph0 / s;
        let k1 = ph1 / s;
        self.a += k0 * e;
        self.b += k1 * e;
        // Joseph form with A = I - K H
        let a00 = 1.0 - k0;
        let a01 = -k0 * x;
        let a10 = -k1;
        let a11 = 1.0 - k1 * x;
        let b00 = a00 * p00 + a01 * p01; // B = A P_pred
        let b01 = a00 * p01 + a01 * p11;
        let b10 = a10 * p00 + a11 * p01;
        let b11 = a10 * p01 + a11 * p11;
        let n00 = b00 * a00 + b01 * a01 + r * (k0 * k0);
        let n01 = b00 * a10 + b01 * a11 + r * (k0 * k1);
        let n10 = b10 * a00 + b11 * a01 + r * (k1 * k0);
        let n11 = b10 * a10 + b11 * a11 + r * (k1 * k1);
        self.p00 = n00;
        self.p01 = 0.5 * (n01 + n10);
        self.p11 = n11;
        self.innovation = e;
        self.s_var = s;
        Ok((self.a, self.b, e / s.sqrt()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_validation() {
        let y = vec![1.0, 2.0, 3.0];
        let x = vec![1.0, 2.0, 3.0];
        assert!(kalman_hedge(&y, &x[..2], 1e-4, 1e-3).is_err());
        assert!(kalman_hedge(&y[..1], &x[..1], 1e-4, 1e-3).is_err());
        assert!(kalman_hedge(&y, &x, 0.0, 1e-3).is_err());
        assert!(kalman_hedge(&y, &x, 1.0, 1e-3).is_err());
        assert!(kalman_hedge(&y, &x, 1e-4, 0.0).is_err());
        assert!(kalman_hedge(&[1.0, f64::NAN, 3.0], &x, 1e-4, 1e-3).is_err());
    }

    #[test]
    fn innovation_variance_positive() {
        let y: Vec<f64> = (0..50).map(|t| 2.0 + 1.5 * t as f64).collect();
        let x: Vec<f64> = (0..50).map(|t| t as f64).collect();
        let km = kalman_hedge(&y, &x, 1e-3, 1e-2).unwrap();
        assert!(km.s_var.iter().all(|&v| v > 0.0));
        assert!(km.z.iter().all(|v| v.is_finite()));
    }
}
