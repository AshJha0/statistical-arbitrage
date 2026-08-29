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
/// `delta` outside (0, 1), or `r <= 0`.
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
    if r <= 0.0 {
        return Err(StatarbError::InvalidParam(
            "observation variance r must be > 0".to_string(),
        ));
    }

    let t_len = y.len();
    let q = delta / (1.0 - delta);
    let mut s = [0.0_f64; 2];
    let mut p = [[P0_SCALE, 0.0], [0.0, P0_SCALE]];

    let mut alpha = vec![0.0; t_len];
    let mut beta = vec![0.0; t_len];
    let mut z = vec![0.0; t_len];
    let mut innovation = vec![0.0; t_len];
    let mut s_var = vec![0.0; t_len];

    for t in 0..t_len {
        // predict: state unchanged, covariance inflated by Q = q I
        let pp = [[p[0][0] + q, p[0][1]], [p[1][0], p[1][1] + q]];
        let xt = x[t];
        // innovate with H = [1, x_t]
        let e = y[t] - (s[0] + xt * s[1]);
        // S = H P_pred H' + r, evaluated as (H P_pred) . H (mirrors reference)
        let hp = [pp[0][0] + xt * pp[1][0], pp[0][1] + xt * pp[1][1]];
        let s_t = hp[0] + xt * hp[1] + r;
        // gain K = P_pred H' / S
        let k = [
            (pp[0][0] + pp[0][1] * xt) / s_t,
            (pp[1][0] + pp[1][1] * xt) / s_t,
        ];
        s[0] += k[0] * e;
        s[1] += k[1] * e;
        // Joseph form: P = (I - K H) P_pred (I - K H)' + r K K'
        let ikh = [[1.0 - k[0], -k[0] * xt], [-k[1], 1.0 - k[1] * xt]];
        let mut tmp = [[0.0_f64; 2]; 2]; // ikh @ P_pred
        for i in 0..2 {
            for j in 0..2 {
                tmp[i][j] = ikh[i][0] * pp[0][j] + ikh[i][1] * pp[1][j];
            }
        }
        let mut pn = [[0.0_f64; 2]; 2]; // tmp @ ikh' + r K K'
        for i in 0..2 {
            for j in 0..2 {
                pn[i][j] = tmp[i][0] * ikh[j][0] + tmp[i][1] * ikh[j][1] + r * k[i] * k[j];
            }
        }
        // symmetrise against rounding
        let off = 0.5 * (pn[0][1] + pn[1][0]);
        p = [[pn[0][0], off], [off, pn[1][1]]];

        alpha[t] = s[0];
        beta[t] = s[1];
        innovation[t] = e;
        s_var[t] = s_t;
        z[t] = e / s_t.sqrt();
    }

    Ok(KalmanResult {
        alpha,
        beta,
        z,
        innovation,
        s_var,
        final_cov: p,
    })
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
