//! Spread construction, rolling z-score, and mean-reversion half-life.

use crate::error::{Result, StatarbError};
use crate::ols::ols;

/// Rolling standard deviations at or below this are treated as
/// zero-variance windows: the z-score is NaN (no signal). Pinned.
pub const ZSCORE_STD_FLOOR: f64 = 1.0e-12;

/// Hedge spread `s_t = y_t - (alpha + beta * x_t)`.
///
/// # Errors
///
/// Length mismatch or non-finite series / hedge parameters.
pub fn spread(y: &[f64], x: &[f64], alpha: f64, beta: f64) -> Result<Vec<f64>> {
    if y.len() != x.len() {
        return Err(StatarbError::LengthMismatch(
            "y and x must be 1-D arrays of equal length".to_string(),
        ));
    }
    if !y.iter().chain(x.iter()).all(|v| v.is_finite()) {
        return Err(StatarbError::NonFinite(
            "inputs must be finite (no NaN/inf)".to_string(),
        ));
    }
    if !(alpha.is_finite() && beta.is_finite()) {
        return Err(StatarbError::NonFinite(
            "alpha and beta must be finite".to_string(),
        ));
    }
    Ok(y.iter()
        .zip(x)
        .map(|(&yv, &xv)| yv - (alpha + beta * xv))
        .collect())
}

/// Rolling z-score with the current observation included (pinned).
///
/// ```text
/// z_t = (s_t - mean(s_{t-W+1..t})) / std(s_{t-W+1..t}, ddof=1)
/// ```
///
/// The first `W-1` entries are NaN (insufficient history). A window whose
/// sample standard deviation is `<= ZSCORE_STD_FLOOR` yields NaN (zero-
/// variance guard) rather than +/-inf.
///
/// # Errors
///
/// `window < 2`, `window > len(s)`, or non-finite input.
pub fn rolling_zscore(s: &[f64], window: usize) -> Result<Vec<f64>> {
    if window < 2 {
        return Err(StatarbError::InvalidParam(
            "window must be an integer >= 2".to_string(),
        ));
    }
    if window > s.len() {
        return Err(StatarbError::InvalidParam(format!(
            "window ({window}) larger than series length ({})",
            s.len()
        )));
    }
    if !s.iter().all(|v| v.is_finite()) {
        return Err(StatarbError::NonFinite(
            "inputs must be finite (no NaN/inf)".to_string(),
        ));
    }
    let t_len = s.len();
    let mut z = vec![f64::NAN; t_len];
    for t in (window - 1)..t_len {
        let w = &s[t + 1 - window..t + 1];
        let m = w.iter().sum::<f64>() / window as f64;
        let var = w.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (window - 1) as f64;
        let sd = var.sqrt();
        if sd > ZSCORE_STD_FLOOR {
            z[t] = (s[t] - m) / sd;
        }
    }
    Ok(z)
}

/// Half-life of mean reversion from an AR(1) fit on the spread.
///
/// Regression (pinned): `ds_t = c + rho * s_{t-1} + e_t` (OLS with
/// intercept); then `half_life = -ln(2) / ln(1 + rho)`.
///
/// Guards (pinned): `rho >= 0` (no reversion) returns `+inf`;
/// `1 + rho <= 0` (over-shooting flip) returns `0.0`.
///
/// # Errors
///
/// Fewer than 12 observations, constant series, or non-finite input.
pub fn half_life(s: &[f64]) -> Result<f64> {
    if s.len() < 12 {
        return Err(StatarbError::TooShort(
            "need at least 12 observations for half-life".to_string(),
        ));
    }
    if !s.iter().all(|v| v.is_finite()) {
        return Err(StatarbError::NonFinite(
            "inputs must be finite (no NaN/inf)".to_string(),
        ));
    }
    let (min, max) = s
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    if max - min == 0.0 {
        return Err(StatarbError::ConstantSeries(
            "series is constant: half-life undefined".to_string(),
        ));
    }
    let ds: Vec<f64> = s.windows(2).map(|w| w[1] - w[0]).collect();
    let fit = ols(&ds, &s[..s.len() - 1], true)?;
    let rho = fit.params[1];
    if rho >= 0.0 {
        return Ok(f64::INFINITY);
    }
    if 1.0 + rho <= 0.0 {
        return Ok(0.0);
    }
    Ok(-std::f64::consts::LN_2 / (1.0 + rho).ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zscore_warmup_and_hand_value() {
        // s = [1, 2, 3, 4]: window 2 -> mean of last two, std = |diff|/sqrt(2)... ddof=1 std of (a,b) = |b-a|/sqrt(2)
        let z = rolling_zscore(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();
        assert!(z[0].is_nan());
        // window (1,2): mean 1.5, sd = sqrt(0.5); z = (2 - 1.5)/sqrt(0.5)
        let expect = 0.5 / 0.5_f64.sqrt();
        assert!((z[1] - expect).abs() < 1e-14);
    }

    #[test]
    fn zscore_zero_variance_guard() {
        let z = rolling_zscore(&[1.0, 1.0, 1.0, 2.0], 3).unwrap();
        assert!(z[2].is_nan()); // constant window -> NaN, never inf
        assert!(z[3].is_finite());
    }

    #[test]
    fn half_life_guards() {
        // explosive series: rho > 0 -> +inf
        let s: Vec<f64> = (0..30).map(|t| 1.01_f64.powi(t)).collect();
        assert_eq!(half_life(&s).unwrap(), f64::INFINITY);
        // violent alternation: 1 + rho <= 0 -> 0.0
        let s: Vec<f64> = (0..30)
            .map(|t| if t % 2 == 0 { 5.0 } else { -5.0 })
            .collect();
        assert_eq!(half_life(&s).unwrap(), 0.0);
    }

    #[test]
    fn half_life_known_ar1() {
        // deterministic AR(1) decay s_{t+1} = 0.9 s_t has rho = -0.1 exactly
        let mut s = vec![10.0];
        for _ in 0..40 {
            s.push(0.9 * s.last().unwrap());
        }
        let hl = half_life(&s).unwrap();
        let expect = -std::f64::consts::LN_2 / 0.9_f64.ln();
        assert!((hl - expect).abs() < 1e-8);
    }
}
