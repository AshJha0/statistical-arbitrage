//! Ordinary least squares — pinned solution method.
//!
//! The solution method is PINNED for cross-language reproducibility:
//!
//! ```text
//! b = solve(X'X, X'y)          (normal equations, LU solve)
//! cov(b) = sigma2 * inv(X'X),  sigma2 = RSS / (n - k)
//! ```
//!
//! No QR, no SVD. The design matrices in this project are tiny (k <= 4)
//! and well-conditioned, so the normal equations lose no meaningful
//! precision. A conditioning guard (`cond(X'X) > 1e12`) rejects
//! (near-)collinear designs with an error rather than returning garbage.

use crate::error::{Result, StatarbError};
use crate::linalg::{cond_sym, inv, lu_solve};

/// Condition-number threshold on `X'X` above which the design is treated
/// as collinear. Pinned.
pub const COND_LIMIT: f64 = 1.0e12;

/// Compensated (Neumaier) sum of `f(t)` over `t = 0..n`.
///
/// The golden tolerances (1e-10 absolute on the OLS coefficients) sit
/// close to the rounding noise of plain sequential accumulation over the
/// 1500-observation cross products once amplified by `cond(X'X) ~ 2e6`,
/// so the normal-equation sums are formed with compensated summation —
/// numerically as good as an exact sum, and safely inside the pinned
/// tolerance of the reference (BLAS-accumulated) values.
fn csum(n: usize, f: impl Fn(usize) -> f64) -> f64 {
    let mut s = 0.0_f64;
    let mut c = 0.0_f64;
    for t in 0..n {
        let v = f(t);
        let tmp = s + v;
        c += if s.abs() >= v.abs() {
            (s - tmp) + v
        } else {
            (v - tmp) + s
        };
        s = tmp;
    }
    s + c
}

/// Result of an OLS fit.
#[derive(Debug, Clone)]
pub struct OlsResult {
    /// Coefficient vector, intercept first when fitted with an intercept.
    pub params: Vec<f64>,
    /// Plain (non-robust) standard errors, same order as `params`.
    pub se: Vec<f64>,
    /// `params / se` elementwise (may be non-finite when an SE is 0).
    pub tstats: Vec<f64>,
    /// Residual vector `y - X b` (length `nobs`).
    pub resid: Vec<f64>,
    /// Coefficient of determination. Centered total sum of squares when
    /// an intercept is included, uncentered otherwise.
    pub r2: f64,
    /// Residual variance estimate `RSS / (nobs - nparams)`.
    pub sigma2: f64,
    /// Number of observations.
    pub nobs: usize,
    /// Number of estimated coefficients (including intercept).
    pub nparams: usize,
}

/// Fit `y = X b + e` by OLS with multiple regressor columns.
///
/// `x_cols` holds the regressors column by column (each of length `n`);
/// the intercept column is prepended automatically when `intercept` is
/// true — do not include one yourself.
///
/// # Errors
///
/// * [`StatarbError::LengthMismatch`] — a regressor column length differs
///   from `len(y)`;
/// * [`StatarbError::NonFinite`] — any NaN/inf input;
/// * [`StatarbError::TooShort`] — `n <= k`;
/// * [`StatarbError::Singular`] — `cond(X'X) > 1e12` (this also catches a
///   constant regressor when an intercept is present);
/// * [`StatarbError::ConstantSeries`] — constant response (R^2 undefined).
pub fn ols_multi(y: &[f64], x_cols: &[Vec<f64>], intercept: bool) -> Result<OlsResult> {
    let n = y.len();
    if x_cols.is_empty() {
        return Err(StatarbError::InvalidParam(
            "at least one regressor column is required".to_string(),
        ));
    }
    for col in x_cols {
        if col.len() != n {
            return Err(StatarbError::LengthMismatch(format!(
                "series length mismatch: len(y)={}, len(x)={}",
                n,
                col.len()
            )));
        }
    }
    if !y.iter().all(|v| v.is_finite()) || !x_cols.iter().all(|c| c.iter().all(|v| v.is_finite())) {
        return Err(StatarbError::NonFinite(
            "inputs must be finite (no NaN/inf)".to_string(),
        ));
    }

    // design columns, intercept first (pinned ordering)
    let mut cols: Vec<&[f64]> = Vec::with_capacity(x_cols.len() + 1);
    let ones;
    if intercept {
        ones = vec![1.0; n];
        cols.push(&ones);
    }
    for c in x_cols {
        cols.push(c);
    }
    let k = cols.len();
    if n <= k {
        return Err(StatarbError::TooShort(format!(
            "need more observations than parameters: n={n}, k={k}"
        )));
    }

    // X'X and X'y by compensated accumulation (see `csum`)
    let mut xtx = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in i..k {
            let s = csum(n, |t| cols[i][t] * cols[j][t]);
            xtx[i][j] = s;
            xtx[j][i] = s;
        }
    }
    if cond_sym(&xtx)? > COND_LIMIT {
        return Err(StatarbError::Singular(
            "design matrix is (near-)singular: constant or collinear regressors".to_string(),
        ));
    }
    let mut xty = vec![0.0; k];
    for (i, col) in cols.iter().enumerate() {
        xty[i] = csum(n, |t| col[t] * y[t]);
    }

    let params = lu_solve(&xtx, &xty)?;
    let mut resid = vec![0.0; n];
    for t in 0..n {
        let mut fit = 0.0;
        for (i, col) in cols.iter().enumerate() {
            fit += col[t] * params[i];
        }
        resid[t] = y[t] - fit;
    }
    let rss = csum(n, |t| resid[t] * resid[t]);
    let sigma2 = rss / (n - k) as f64;
    let xtx_inv = inv(&xtx)?;
    let se: Vec<f64> = (0..k).map(|i| (sigma2 * xtx_inv[i][i]).sqrt()).collect();

    let tss = if intercept {
        let mean = csum(n, |t| y[t]) / n as f64;
        csum(n, |t| (y[t] - mean) * (y[t] - mean))
    } else {
        csum(n, |t| y[t] * y[t])
    };
    if tss <= 0.0 {
        return Err(StatarbError::ConstantSeries(
            "y is constant: R^2 undefined".to_string(),
        ));
    }
    let r2 = 1.0 - rss / tss;
    let tstats: Vec<f64> = params.iter().zip(&se).map(|(b, s)| b / s).collect();

    Ok(OlsResult {
        params,
        se,
        tstats,
        resid,
        r2,
        sigma2,
        nobs: n,
        nparams: k,
    })
}

/// Fit `y = a + b x + e` (single regressor) by the pinned OLS.
///
/// Convenience wrapper over [`ols_multi`]; see it for the error contract.
pub fn ols(y: &[f64], x: &[f64], intercept: bool) -> Result<OlsResult> {
    ols_multi(y, std::slice::from_ref(&x.to_vec()), intercept)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_fit_no_intercept() {
        let x: Vec<f64> = (1..30).map(|v| v as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v).collect();
        let fit = ols(&y, &x, false).unwrap();
        assert!((fit.params[0] - 2.0).abs() < 1e-12);
        assert!((fit.r2 - 1.0).abs() < 1e-12);
        assert!(fit.resid.iter().all(|e| e.abs() < 1e-10));
        assert_eq!(fit.nparams, 1);
    }

    #[test]
    fn collinear_design_rejected() {
        let x: Vec<f64> = (0..50).map(|v| v as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 1.0 + v).collect();
        let cols = vec![x.clone(), x.clone()];
        assert!(matches!(
            ols_multi(&y, &cols, true),
            Err(StatarbError::Singular(_))
        ));
    }

    #[test]
    fn constant_y_rejected() {
        let x: Vec<f64> = (0..30).map(|v| v as f64).collect();
        let y = vec![5.0; 30];
        assert!(matches!(
            ols(&y, &x, true),
            Err(StatarbError::ConstantSeries(_))
        ));
    }
}
