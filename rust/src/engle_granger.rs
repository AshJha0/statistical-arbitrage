//! Engle-Granger two-step cointegration test.
//!
//! Step 1: static OLS hedge regression `y_t = a + b x_t + e_t` (normal
//! equations, see [`crate::ols`]).
//! Step 2: ADF test (constant, fixed maxlag) on the residuals `e_t`,
//! judged against the Engle-Granger 2-variable critical-value table
//! (`nvars = 2`) — the residuals are estimated, so the plain
//! Dickey-Fuller table would over-reject.

use crate::adf::{adf_test, AdfResult};
use crate::error::Result;
use crate::ols::ols;

/// Engle-Granger two-step result.
#[derive(Debug, Clone)]
pub struct EgResult {
    /// Intercept of the hedge regression.
    pub alpha: f64,
    /// Hedge ratio `b`.
    pub beta: f64,
    /// Cointegrating residuals `y - a - b x`.
    pub resid: Vec<f64>,
    /// R^2 of the hedge regression.
    pub r2: f64,
    /// ADF result on the residuals (`nvars = 2` table).
    pub adf: AdfResult,
}

impl EgResult {
    /// True when the residual ADF rejects a unit root at `alpha`.
    ///
    /// # Errors
    ///
    /// `alpha` outside {0.01, 0.05, 0.10}.
    pub fn cointegrated(&self, alpha: f64) -> Result<bool> {
        self.adf.reject(alpha)
    }
}

/// Run the Engle-Granger two-step test of cointegration of `y` on `x`.
///
/// * `maxlag` — fixed ADF augmentation lag (this project uses 1).
///
/// # Errors
///
/// Propagated from OLS/ADF validation (length mismatch, constant series,
/// non-finite values, short series).
pub fn engle_granger(y: &[f64], x: &[f64], maxlag: usize) -> Result<EgResult> {
    let fit = ols(y, x, true)?;
    let adf = adf_test(&fit.resid, maxlag, 2)?;
    Ok(EgResult {
        alpha: fit.params[0],
        beta: fit.params[1],
        resid: fit.resid,
        r2: fit.r2,
        adf,
    })
}
