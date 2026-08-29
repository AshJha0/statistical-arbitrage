//! Augmented Dickey-Fuller test with embedded critical-value tables.
//!
//! Regression form (PINNED, constant / no-trend case):
//!
//! ```text
//! dy_t = c + rho * y_{t-1} + sum_{i=1..p} phi_i * dy_{t-i} + e_t
//! ```
//!
//! with a FIXED augmentation lag `p = maxlag` (default 1 in this project;
//! no automatic lag selection). The test statistic is the plain OLS
//! t-statistic on `rho`. With `T` raw observations the regression uses
//! the `T - 1 - p` rows for which all lagged differences exist.
//!
//! Critical values are MacKinnon-style ASYMPTOTIC values, constant
//! no-trend case, embedded verbatim below for the 1-variable ADF test and
//! for the Engle-Granger residual test with 2 and 3 variables. The
//! p-value is a documented approximation: piecewise-linear interpolation
//! of the significance level through the three (critical value, level)
//! knots, with linear extrapolation using the edge-segment slopes,
//! clamped to `[1e-4, 0.9999]`. The exact MacKinnon response-surface
//! regression is deliberately NOT implemented.

use crate::error::{Result, StatarbError};
use crate::ols::ols_multi;

/// Significance levels matching the columns of the critical-value table.
pub const LEVELS: [f64; 3] = [0.01, 0.05, 0.10];

/// Clamp bounds for the interpolated p-value. Pinned.
pub const PVALUE_MIN: f64 = 1.0e-4;
/// Upper clamp bound for the interpolated p-value. Pinned.
pub const PVALUE_MAX: f64 = 0.9999;

/// MacKinnon-style asymptotic critical values (1%, 5%, 10%), constant
/// no-trend case, keyed by the number of variables (1 = plain ADF;
/// 2, 3 = Engle-Granger residual test). PINNED — every language embeds
/// these exact numbers.
pub fn critical_values(nvars: usize) -> Result<[f64; 3]> {
    match nvars {
        1 => Ok([-3.43, -2.86, -2.57]),
        2 => Ok([-3.90, -3.34, -3.04]),
        3 => Ok([-4.29, -3.74, -3.45]),
        _ => Err(StatarbError::InvalidParam(
            "nvars must be 1, 2 or 3".to_string(),
        )),
    }
}

/// Result of an ADF unit-root test.
#[derive(Debug, Clone)]
pub struct AdfResult {
    /// OLS t-statistic on `rho`.
    pub tstat: f64,
    /// Interpolated approximate p-value (see module docs).
    pub pvalue: f64,
    /// Critical values `[1%, 5%, 10%]` used for the verdict.
    pub crit: [f64; 3],
    /// Augmentation lag `p` used (fixed, not selected).
    pub maxlag: usize,
    /// Effective regression observations (`T - 1 - p`).
    pub nobs: usize,
    /// Table selector (1 = ADF, 2/3 = EG residual case).
    pub nvars: usize,
}

impl AdfResult {
    /// Reject the unit-root null at `alpha` in {0.01, 0.05, 0.10}.
    ///
    /// The verdict is by direct table comparison (`tstat < critical
    /// value`), NOT via the interpolated p-value.
    ///
    /// # Errors
    ///
    /// [`StatarbError::InvalidParam`] when `alpha` is not one of the three
    /// tabulated levels.
    pub fn reject(&self, alpha: f64) -> Result<bool> {
        let idx = if alpha == 0.01 {
            0
        } else if alpha == 0.05 {
            1
        } else if alpha == 0.10 {
            2
        } else {
            return Err(StatarbError::InvalidParam(
                "alpha must be one of 0.01, 0.05, 0.10".to_string(),
            ));
        };
        Ok(self.tstat < self.crit[idx])
    }
}

/// Approximate p-value by piecewise-linear interpolation.
///
/// Knots are `(cv_1%, 0.01)`, `(cv_5%, 0.05)`, `(cv_10%, 0.10)`; at or
/// below `cv_5%` the 1%-5% segment applies, otherwise the 5%-10% segment;
/// outside the knot range the edge segment's slope extrapolates linearly;
/// the result is clamped to `[PVALUE_MIN, PVALUE_MAX]`.
pub fn interpolate_pvalue(tstat: f64, nvars: usize) -> Result<f64> {
    let [cv1, cv5, cv10] = critical_values(nvars)?;
    let p = if tstat <= cv5 {
        let slope = (0.05 - 0.01) / (cv5 - cv1);
        0.01 + slope * (tstat - cv1)
    } else {
        let slope = (0.10 - 0.05) / (cv10 - cv5);
        0.05 + slope * (tstat - cv5)
    };
    Ok(p.clamp(PVALUE_MIN, PVALUE_MAX))
}

/// Run the ADF regression and return the unit-root test result.
///
/// * `y` — series to test (levels for `nvars=1`; cointegrating residuals
///   for `nvars=2` or `3`), finite.
/// * `maxlag` — fixed augmentation lag `p >= 0` (unsigned by type).
/// * `nvars` — critical-value table selector: 1 for a plain ADF test,
///   2 or 3 for the Engle-Granger residual test.
///
/// # Errors
///
/// Unknown `nvars`, non-finite input, constant series, or too few
/// observations (effective nobs must be `>= nparams + 10`).
pub fn adf_test(y: &[f64], maxlag: usize, nvars: usize) -> Result<AdfResult> {
    let crit = critical_values(nvars)?;
    if !y.iter().all(|v| v.is_finite()) {
        return Err(StatarbError::NonFinite(
            "inputs must be finite (no NaN/inf)".to_string(),
        ));
    }
    let t_len = y.len();
    let p = maxlag;
    let nparams = 2 + p; // c, rho, phi_1..phi_p
    if t_len > 0 {
        let (min, max) = y
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        if max - min == 0.0 {
            return Err(StatarbError::ConstantSeries(
                "series is constant: ADF test undefined".to_string(),
            ));
        }
    }
    let eff = t_len as i64 - 1 - p as i64;
    if eff < (nparams + 10) as i64 {
        return Err(StatarbError::TooShort(format!(
            "series too short for ADF with maxlag={p}: effective nobs {eff}"
        )));
    }
    let eff = eff as usize;

    let dy: Vec<f64> = y.windows(2).map(|w| w[1] - w[0]).collect(); // length T-1
    let lhs = &dy[p..];
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(1 + p);
    cols.push(y[p..t_len - 1].to_vec()); // y_{t-1}
    for i in 1..=p {
        cols.push(dy[p - i..dy.len() - i].to_vec()); // dy_{t-i}
    }
    let fit = ols_multi(lhs, &cols, true)?;
    let tstat = fit.tstats[1]; // rho is the first regressor after the intercept
    Ok(AdfResult {
        tstat,
        pvalue: interpolate_pvalue(tstat, nvars)?,
        crit,
        maxlag: p,
        nobs: eff,
        nvars,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pvalue_matches_table_knots() {
        let [cv1, cv5, cv10] = critical_values(1).unwrap();
        assert!((interpolate_pvalue(cv1, 1).unwrap() - 0.01).abs() < 1e-12);
        assert!((interpolate_pvalue(cv5, 1).unwrap() - 0.05).abs() < 1e-12);
        assert!((interpolate_pvalue(cv10, 1).unwrap() - 0.10).abs() < 1e-12);
    }

    #[test]
    fn pvalue_clamps_at_bounds() {
        assert_eq!(interpolate_pvalue(-50.0, 1).unwrap(), PVALUE_MIN);
        assert_eq!(interpolate_pvalue(50.0, 1).unwrap(), PVALUE_MAX);
    }

    #[test]
    fn invalid_nvars_rejected() {
        assert!(critical_values(4).is_err());
        assert!(adf_test(&[1.0; 40], 1, 0).is_err());
    }

    #[test]
    fn short_and_constant_series_rejected() {
        let short: Vec<f64> = (0..8).map(|v| v as f64).collect();
        assert!(matches!(
            adf_test(&short, 1, 1),
            Err(StatarbError::TooShort(_))
        ));
        assert!(matches!(
            adf_test(&[3.0; 100], 1, 1),
            Err(StatarbError::ConstantSeries(_))
        ));
    }
}
