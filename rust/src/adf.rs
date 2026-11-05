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
//! Critical values (PINNED) are the MacKinnon (2010) finite-sample
//! response surfaces for the constant / no-trend case,
//!
//! ```text
//! cv(T_eff) = b_inf + b_1/T_eff + b_2/T_eff^2 + b_3/T_eff^3,
//! ```
//!
//! embedded verbatim below for the 1-variable ADF test and for the
//! Engle-Granger residual test with 2 and 3 variables, where `T_eff` is
//! the number of observations in the test regression (as in statsmodels'
//! `adfuller`). The surfaces were fitted on samples of 20 or more, so
//! `T_eff >= 20` is enforced. The p-value is a documented approximation:
//! piecewise-linear interpolation of the significance level through the
//! three (critical value, level) knots at `T_eff`, with linear
//! extrapolation using the edge-segment slopes, clamped to
//! `[1e-4, 0.9999]`. The exact MacKinnon (1994) p-value response surface
//! is deliberately NOT implemented; the verdict never uses the p-value.

use crate::error::{Result, StatarbError};
use crate::ols::ols_multi;

/// Significance levels matching the columns of the critical-value table.
pub const LEVELS: [f64; 3] = [0.01, 0.05, 0.10];

/// Smallest effective sample the response surfaces are trusted for. Pinned.
pub const MIN_NOBS: usize = 20;

/// Largest augmentation lag accepted (documented upper bound). Pinned.
pub const MAX_LAG: usize = 1000;

/// Clamp bounds for the interpolated p-value. Pinned.
pub const PVALUE_MIN: f64 = 1.0e-4;
/// Upper clamp bound for the interpolated p-value. Pinned.
pub const PVALUE_MAX: f64 = 0.9999;

/// MacKinnon (2010) response-surface coefficients `[b_inf, b1, b2, b3]`
/// for the rows (1%, 5%, 10%), constant / no-trend case, Table 1 of
/// "Critical Values for Cointegration Tests" (Queen's Economics Dept.
/// WP 1227). Keyed by the number of variables (1 = plain ADF; 2, 3 =
/// Engle-Granger residual test). PINNED — every language embeds these
/// exact numbers.
///
/// # Errors
///
/// [`StatarbError::InvalidParam`] unless `nvars` is 1, 2 or 3.
pub fn response_surface(nvars: usize) -> Result<[[f64; 4]; 3]> {
    match nvars {
        1 => Ok([
            [-3.43035, -6.5393, -16.786, -79.433],
            [-2.86154, -2.8903, -4.234, -40.040],
            [-2.56677, -1.5384, -2.809, 0.0],
        ]),
        2 => Ok([
            [-3.89644, -10.9519, -33.527, 0.0],
            [-3.33613, -6.1101, -6.823, 0.0],
            [-3.04445, -4.2412, -2.720, 0.0],
        ]),
        3 => Ok([
            [-4.29374, -14.4354, -33.195, 47.433],
            [-3.74066, -8.5632, -10.852, 27.982],
            [-3.45218, -6.2143, -3.718, 0.0],
        ]),
        _ => Err(StatarbError::InvalidParam(
            "nvars must be 1, 2 or 3".to_string(),
        )),
    }
}

/// Critical values (1%, 5%, 10%) for `nvars` variables at effective sample
/// size `nobs`; `None` returns the asymptotic values `b_inf`.
///
/// # Errors
///
/// [`StatarbError::InvalidParam`] unless `nvars` is 1, 2 or 3;
/// [`StatarbError::TooShort`] when `nobs < MIN_NOBS`.
pub fn critical_values(nvars: usize, nobs: Option<usize>) -> Result<[f64; 3]> {
    let rows = response_surface(nvars)?;
    match nobs {
        None => Ok([rows[0][0], rows[1][0], rows[2][0]]),
        Some(n) if n < MIN_NOBS => Err(StatarbError::TooShort(format!(
            "critical values need at least {MIN_NOBS} effective observations"
        ))),
        Some(n) => {
            let inv = 1.0 / n as f64;
            let mut out = [0.0; 3];
            for (o, r) in out.iter_mut().zip(rows.iter()) {
                // Horner form, identical to the Python reference.
                *o = r[0] + inv * (r[1] + inv * (r[2] + inv * r[3]));
            }
            Ok(out)
        }
    }
}

/// Result of an ADF unit-root test.
#[derive(Debug, Clone)]
pub struct AdfResult {
    /// OLS t-statistic on `rho`.
    pub tstat: f64,
    /// Interpolated approximate p-value (see module docs).
    pub pvalue: f64,
    /// Finite-sample critical values `[1%, 5%, 10%]` used for the verdict.
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
/// Knots are `(cv_1%, 0.01)`, `(cv_5%, 0.05)`, `(cv_10%, 0.10)` at sample
/// size `nobs` (asymptotic when `None`); at or below `cv_5%` the 1%-5%
/// segment applies, otherwise the 5%-10% segment; outside the knot range
/// the edge segment's slope extrapolates linearly; the result is clamped
/// to `[PVALUE_MIN, PVALUE_MAX]`.
///
/// # Errors
///
/// Bad `nvars`/`nobs` as in [`critical_values`], or a NaN `tstat`
/// ([`StatarbError::NonFinite`]); +/-inf (exact-fit regressions) clamps.
pub fn interpolate_pvalue(tstat: f64, nvars: usize, nobs: Option<usize>) -> Result<f64> {
    let [cv1, cv5, cv10] = critical_values(nvars, nobs)?;
    if tstat.is_nan() {
        return Err(StatarbError::NonFinite("tstat must not be NaN".to_string()));
    }
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
/// * `maxlag` — fixed augmentation lag `p` in `0..=MAX_LAG`.
/// * `nvars` — critical-value table selector: 1 for a plain ADF test,
///   2 or 3 for the Engle-Granger residual test.
///
/// # Errors
///
/// Unknown `nvars`, `maxlag > MAX_LAG`, non-finite input, constant
/// series, or too few observations (effective nobs must be
/// `>= max(nparams + 10, MIN_NOBS)`).
pub fn adf_test(y: &[f64], maxlag: usize, nvars: usize) -> Result<AdfResult> {
    response_surface(nvars)?;
    if maxlag > MAX_LAG {
        return Err(StatarbError::InvalidParam(format!(
            "maxlag must be <= {MAX_LAG}"
        )));
    }
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
    // i64 arithmetic: no overflow for any admissible maxlag (<= MAX_LAG).
    let eff = t_len as i64 - 1 - p as i64;
    let min_eff = ((nparams + 10) as i64).max(MIN_NOBS as i64);
    if eff < min_eff {
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
    let crit = critical_values(nvars, Some(eff))?;
    Ok(AdfResult {
        tstat,
        pvalue: interpolate_pvalue(tstat, nvars, Some(eff))?,
        crit,
        maxlag: p,
        nobs: eff,
        nvars,
    })
}

/// Lag in `0..=max_lag` with the smallest BIC, for use with [`adf_test`].
///
/// Every candidate is fitted on the SAME rows `t = max_lag+1 .. T-1`, so the
/// BICs are comparable: `BIC(p) = n ln(RSS_p / n) + (2 + p) ln n` with
/// `n = T - 1 - max_lag`. A tie goes to the smaller lag.
///
/// # Errors
///
/// Same validation as `adf_test(y, max_lag, nvars)`.
pub fn select_lag(y: &[f64], max_lag: usize, nvars: usize) -> Result<usize> {
    adf_test(y, max_lag, nvars)?; // shared validation
    let m = max_lag;
    let t_len = y.len();
    let dy: Vec<f64> = y.windows(2).map(|w| w[1] - w[0]).collect();
    let lhs = &dy[m..];
    let n = lhs.len() as f64;
    let mut cols: Vec<Vec<f64>> = vec![y[m..t_len - 1].to_vec()];
    let (mut best_lag, mut best_bic) = (0usize, f64::INFINITY);
    for p in 0..=m {
        if p > 0 {
            cols.push(dy[m - p..dy.len() - p].to_vec());
        }
        let fit = ols_multi(lhs, &cols, true)?;
        let mut rss = 0.0_f64;
        for &e in &fit.resid {
            rss += e * e;
        }
        let fit_term = if rss > 0.0 {
            n * (rss / n).ln()
        } else {
            f64::NEG_INFINITY
        };
        let bic = fit_term + (2 + p) as f64 * n.ln();
        if p == 0 || bic < best_bic {
            best_lag = p;
            best_bic = bic;
        }
    }
    Ok(best_lag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pvalue_matches_table_knots() {
        let [cv1, cv5, cv10] = critical_values(1, None).unwrap();
        assert!((interpolate_pvalue(cv1, 1, None).unwrap() - 0.01).abs() < 1e-12);
        assert!((interpolate_pvalue(cv5, 1, None).unwrap() - 0.05).abs() < 1e-12);
        assert!((interpolate_pvalue(cv10, 1, None).unwrap() - 0.10).abs() < 1e-12);
        // finite-sample knots too
        let [_, cv5f, _] = critical_values(2, Some(898)).unwrap();
        assert!((interpolate_pvalue(cv5f, 2, Some(898)).unwrap() - 0.05).abs() < 1e-12);
    }

    #[test]
    fn pvalue_clamps_at_bounds() {
        assert_eq!(interpolate_pvalue(-50.0, 1, None).unwrap(), PVALUE_MIN);
        assert_eq!(interpolate_pvalue(50.0, 1, None).unwrap(), PVALUE_MAX);
        assert_eq!(
            interpolate_pvalue(f64::NEG_INFINITY, 1, None).unwrap(),
            PVALUE_MIN
        );
        assert_eq!(
            interpolate_pvalue(f64::INFINITY, 1, None).unwrap(),
            PVALUE_MAX
        );
        assert!(matches!(
            interpolate_pvalue(f64::NAN, 1, None),
            Err(StatarbError::NonFinite(_))
        ));
    }

    #[test]
    fn invalid_nvars_rejected() {
        assert!(critical_values(4, None).is_err());
        assert!(matches!(
            interpolate_pvalue(-3.0, 4, None),
            Err(StatarbError::InvalidParam(_))
        ));
        assert!(adf_test(&[1.0; 40], 1, 0).is_err());
    }

    #[test]
    fn asymptotic_table_pinned() {
        assert_eq!(
            critical_values(1, None).unwrap(),
            [-3.43035, -2.86154, -2.56677]
        );
        assert_eq!(
            critical_values(2, None).unwrap(),
            [-3.89644, -3.33613, -3.04445]
        );
        assert_eq!(
            critical_values(3, None).unwrap(),
            [-4.29374, -3.74066, -3.45218]
        );
        assert_eq!(
            response_surface(1).unwrap()[1],
            [-2.86154, -2.8903, -4.234, -40.040]
        );
        assert_eq!(response_surface(3).unwrap()[0][3], 47.433);
    }

    #[test]
    fn finite_sample_values_follow_response_surface() {
        // hand evaluation of the pinned N=1 5% row at T = 50 and T = 20,
        // and the N=2 5% row at T = 898 (the in-sample EG regression size)
        let e50 = -2.86154 - 2.8903 / 50.0 - 4.234 / 2500.0 - 40.040 / 125_000.0;
        assert!((critical_values(1, Some(50)).unwrap()[1] - e50).abs() < 1e-12);
        let e20 = -2.86154 - 2.8903 / 20.0 - 4.234 / 400.0 - 40.040 / 8000.0;
        assert!((critical_values(1, Some(20)).unwrap()[1] - e20).abs() < 1e-12);
        let e898 = -3.33613 - 6.1101 / 898.0 - 6.823 / (898.0 * 898.0);
        assert!((critical_values(2, Some(898)).unwrap()[1] - e898).abs() < 1e-12);
        for n in 1..=3 {
            let inf = critical_values(n, None).unwrap();
            let c20 = critical_values(n, Some(20)).unwrap();
            let big = critical_values(n, Some(10_000_000)).unwrap();
            for i in 0..3 {
                assert!(c20[i] < inf[i]);
                assert!((big[i] - inf[i]).abs() < 1e-5);
            }
        }
        assert!(matches!(
            critical_values(1, Some(19)),
            Err(StatarbError::TooShort(_))
        ));
    }

    #[test]
    fn small_sample_guard_and_maxlag_bound() {
        // T_eff < 20 is refused; at T_eff = 20 the verdict uses cv(20)
        let s: Vec<f64> = (0..40)
            .map(|t| 0.9_f64.powi(t) + 0.05 * (2.1 * t as f64).sin())
            .collect();
        assert!(matches!(
            adf_test(&s[..21], 1, 1),
            Err(StatarbError::TooShort(_))
        ));
        assert!(matches!(
            adf_test(&s[..15], 1, 1),
            Err(StatarbError::TooShort(_))
        ));
        let res = adf_test(&s[..22], 1, 1).unwrap();
        assert_eq!(res.nobs, 20);
        let e20 = -2.86154 - 2.8903 / 20.0 - 4.234 / 400.0 - 40.040 / 8000.0;
        assert!((res.crit[1] - e20).abs() < 1e-12);
        assert_eq!(res.reject(0.05).unwrap(), res.tstat < res.crit[1]);
        assert!(matches!(
            adf_test(&s, MAX_LAG + 1, 1),
            Err(StatarbError::InvalidParam(_))
        ));
        assert!(matches!(
            adf_test(&s, MAX_LAG, 1),
            Err(StatarbError::TooShort(_))
        ));
        assert!(matches!(
            adf_test(&s, usize::MAX, 1),
            Err(StatarbError::InvalidParam(_))
        ));
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
