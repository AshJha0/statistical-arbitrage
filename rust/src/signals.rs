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
/// Summation is pinned: plain sequential left-to-right sums (API_SPEC s7).
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
    // PINNED: plain left-to-right sums, one accumulator from 0.0, in every
    // port (no pairwise, Kahan or Neumaier summation).
    for t in (window - 1)..t_len {
        let lo = t + 1 - window;
        let mut total = 0.0_f64;
        for &v in &s[lo..=t] {
            total += v;
        }
        let m = total / window as f64;
        let mut ss = 0.0_f64;
        for &v in &s[lo..=t] {
            let d = v - m;
            ss += d * d;
        }
        let sd = (ss / (window - 1) as f64).sqrt();
        if sd > ZSCORE_STD_FLOOR {
            z[t] = (s[t] - m) / sd;
        }
    }
    Ok(z)
}

/// Streaming rolling z-score: ring buffer and running moments, O(1) per
/// update and no allocation in [`RollingZ::update`].
///
/// Same definition and NaN rules as [`rolling_zscore`]. While filling it
/// uses Welford's update; once full it slides the window with
/// `mean' = mean + (s - old)/W` and
/// `m2 += (s - old)*(s - mean' + old - mean)` (floored at 0). It agrees
/// with the batch function to about 1e-9 at the bundled length and is NOT
/// bitwise equal; the batch function stays the trading reference.
#[derive(Debug, Clone)]
pub struct RollingZ {
    window: usize,
    buf: Vec<f64>,
    head: usize,
    count: usize,
    mean: f64,
    m2: f64,
}

impl RollingZ {
    /// New streaming z-score with window `window` (>= 2).
    ///
    /// # Errors
    ///
    /// `window < 2`.
    pub fn new(window: usize) -> Result<Self> {
        if window < 2 {
            return Err(StatarbError::InvalidParam(
                "window must be an integer >= 2".to_string(),
            ));
        }
        Ok(RollingZ {
            window,
            buf: vec![0.0; window],
            head: 0,
            count: 0,
            mean: 0.0,
            m2: 0.0,
        })
    }

    /// Add one observation; returns z, or NaN while warming up or on a
    /// zero-variance window.
    ///
    /// # Errors
    ///
    /// Non-finite `s` (the state is left unchanged).
    pub fn update(&mut self, s: f64) -> Result<f64> {
        if !s.is_finite() {
            return Err(StatarbError::NonFinite(
                "inputs must be finite (no NaN/inf)".to_string(),
            ));
        }
        if self.count < self.window {
            self.count += 1;
            let delta = s - self.mean;
            self.mean += delta / self.count as f64;
            self.m2 += delta * (s - self.mean);
        } else {
            let old = self.buf[self.head];
            let old_mean = self.mean;
            self.mean = old_mean + (s - old) / self.window as f64;
            self.m2 += (s - old) * (s - self.mean + old - old_mean);
            if self.m2 < 0.0 {
                self.m2 = 0.0;
            }
        }
        self.buf[self.head] = s;
        self.head = (self.head + 1) % self.window;
        if self.count < self.window {
            return Ok(f64::NAN);
        }
        let sd = (self.m2 / (self.window - 1) as f64).sqrt();
        if sd > ZSCORE_STD_FLOOR {
            Ok((s - self.mean) / sd)
        } else {
            Ok(f64::NAN)
        }
    }
}

/// Half-life of mean reversion from an AR(1) fit on the spread.
///
/// Regression (pinned): `ds_t = c + rho * s_{t-1} + e_t` (OLS with
/// intercept); then `half_life = -ln(2) / ln(1 + rho)`.
///
/// Guards (pinned): `rho >= 0` (no reversion) returns `+inf`;
/// `1 + rho <= 0` returns the sentinel `0.0`. Both sentinels mean "the
/// AR(1) fit is not interpretable as a half-life — do not trade this
/// spread" (`0.0` covers both the oscillating `-2 < rho < -1` and the
/// explosive `rho <= -2` cases and deliberately does not distinguish
/// them); a filter such as `0 < hl < 30` excludes both.
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
