//! Johansen-lite trace test for 2- and 3-asset systems.
//!
//! Canonical correlation setup (PINNED). VECM with unrestricted constant
//! and k = 1 lagged difference:
//!
//! ```text
//! dY_t = c + Gamma * dY_{t-1} + Pi * Y_{t-1} + e_t
//! ```
//!
//! Steps (T raw observations, n = 2 or 3 assets, T_eff = T - 2 usable
//! rows, t = 2..T-1 zero-indexed):
//!
//! 1. `Z0 = dY_t`, `Z1 = Y_{t-1}`, `ZD = [1, dY_{t-1}]` (partialled-out).
//! 2. `R0` = residuals of Z0 on ZD; `R1` = residuals of Z1 on ZD
//!    (multivariate OLS via normal equations, one solve per system).
//! 3. Product-moment matrices `S_ij = R_i' R_j / T_eff`.
//! 4. Eigenvalue problem `|lambda S11 - S10 S00^{-1} S01| = 0`, made
//!    SYMMETRIC via the Cholesky factor `S11 = L L'`:
//!    `A = L^{-1} S10 S00^{-1} S01 L^{-T}` (symmetric PSD, n x n),
//!    eigen-decomposed by the native cyclic Jacobi rotations of
//!    [`crate::linalg::jacobi_eigh`] (pinned — no library eigensolver).
//!    Eigenvalues sorted descending, clipped into `[0, 1 - 1e-12]`;
//!    cointegrating vectors `beta_i = L^{-T} v_i`, normalised so the
//!    first component is +1 (when its magnitude exceeds 1e-12).
//! 5. Trace statistics `trace(r) = -T_eff * sum_{i=r..n-1} ln(1 - lambda_i)`
//!    against the embedded Osterwald-Lenum (1992) 95% table for the
//!    unrestricted-constant case; the estimated rank is the smallest r
//!    with `trace(r) < cv(n - r)`, else n.
//!
//! Honest limitations: fixed lag (k=1), fixed deterministic case, only
//! the 95% column; no small-sample correction, no maximum-eigenvalue
//! variant, no p-values. Meant for 2-3 asset pairs-trading diagnostics,
//! not general VECM work.
#![allow(clippy::needless_range_loop)] // index form mirrors the pinned matrix recursions

use crate::error::{Result, StatarbError};
use crate::linalg::{
    cholesky_lower, cond_sym, identity, jacobi_eigh, lu_solve_multi, mat_mul, transpose, Matrix,
};

/// Osterwald-Lenum (1992) 95% critical values for the TRACE statistic,
/// model with unrestricted constant, indexed by `n - r`:
/// `1 -> 3.76, 2 -> 15.41, 3 -> 29.68`. PINNED.
pub fn trace_crit_95(n_minus_r: usize) -> Result<f64> {
    match n_minus_r {
        1 => Ok(3.76),
        2 => Ok(15.41),
        3 => Ok(29.68),
        _ => Err(StatarbError::InvalidParam(
            "trace critical values tabulated for n - r in {1, 2, 3} only".to_string(),
        )),
    }
}

/// Johansen-lite trace-test result.
#[derive(Debug, Clone)]
pub struct JohansenResult {
    /// Canonical correlations `lambda_i`, descending, in `[0, 1)`.
    pub eigenvalues: Vec<f64>,
    /// `trace(r)` for `r = 0..n-1`.
    pub trace_stats: Vec<f64>,
    /// Matching Osterwald-Lenum 95% critical values.
    pub crit_95: Vec<f64>,
    /// Estimated cointegration rank at 95%.
    pub rank: usize,
    /// `n x n` matrix; column i is the cointegrating vector for
    /// `lambda_i`, normalised to first component +1.
    pub beta: Matrix,
    /// Effective sample size `T - 2`.
    pub t_eff: usize,
}

/// Residuals of each column of `z` on the common design `d` (one solve).
fn residualise(z: &[Vec<f64>], d: &[Vec<f64>]) -> Result<Matrix> {
    let dt = transpose(d);
    let dtd = mat_mul(&dt, d);
    if cond_sym(&dtd)? > 1.0e12 {
        return Err(StatarbError::Singular(
            "partialled-out design is (near-)singular".to_string(),
        ));
    }
    let dtz = mat_mul(&dt, z);
    let coef = lu_solve_multi(&dtd, &dtz)?;
    let fitted = mat_mul(d, &coef);
    Ok(z.iter()
        .zip(&fitted)
        .map(|(zr, fr)| zr.iter().zip(fr).map(|(a, b)| a - b).collect())
        .collect())
}

/// Run the Johansen-lite trace test on a T x n matrix of levels.
///
/// * `levels` — price levels, T rows of length n with n in {2, 3},
///   `T >= 20`.
/// * `maxlag_diff` — number of lagged differences k; only `k = 1` is
///   implemented (pinned) — any other value errors.
///
/// # Errors
///
/// Ragged rows, n not in {2,3}, `T < 20`, non-finite values, constant
/// column, or `maxlag_diff != 1`.
pub fn johansen_trace(levels: &[Vec<f64>], maxlag_diff: usize) -> Result<JohansenResult> {
    let t_raw = levels.len();
    if t_raw == 0 {
        return Err(StatarbError::InvalidParam(
            "levels must be a 2-D (T, n) array".to_string(),
        ));
    }
    let n = levels[0].len();
    if levels.iter().any(|row| row.len() != n) {
        return Err(StatarbError::InvalidParam(
            "levels must be a 2-D (T, n) array".to_string(),
        ));
    }
    if !(n == 2 || n == 3) {
        return Err(StatarbError::InvalidParam(
            "johansen_trace supports n = 2 or 3 assets only".to_string(),
        ));
    }
    if maxlag_diff != 1 {
        return Err(StatarbError::InvalidParam(
            "only maxlag_diff = 1 is implemented (pinned)".to_string(),
        ));
    }
    if t_raw < 20 {
        return Err(StatarbError::TooShort(
            "need at least 20 observations".to_string(),
        ));
    }
    if !levels.iter().all(|r| r.iter().all(|v| v.is_finite())) {
        return Err(StatarbError::NonFinite(
            "inputs must be finite (no NaN/inf)".to_string(),
        ));
    }
    for j in 0..n {
        let (lo, hi) = levels
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), row| {
                (lo.min(row[j]), hi.max(row[j]))
            });
        if hi - lo == 0.0 {
            return Err(StatarbError::ConstantSeries(
                "constant column in levels".to_string(),
            ));
        }
    }

    let t_eff = t_raw - 2;
    // dY has T-1 rows
    let dy: Matrix = (1..t_raw)
        .map(|t| (0..n).map(|j| levels[t][j] - levels[t - 1][j]).collect())
        .collect();
    let z0: Matrix = dy[1..].to_vec(); // dY_t, t = 2..T-1
    let z1: Matrix = levels[1..t_raw - 1].to_vec(); // Y_{t-1}
    let zd: Matrix = (0..t_eff)
        .map(|i| {
            let mut row = Vec::with_capacity(n + 1);
            row.push(1.0);
            row.extend_from_slice(&dy[i]); // dY_{t-1}
            row
        })
        .collect();

    let r0 = residualise(&z0, &zd)?;
    let r1 = residualise(&z1, &zd)?;
    let scale = |m: Matrix| -> Matrix {
        m.into_iter()
            .map(|row| row.into_iter().map(|v| v / t_eff as f64).collect())
            .collect()
    };
    let r0t = transpose(&r0);
    let r1t = transpose(&r1);
    let s00 = scale(mat_mul(&r0t, &r0));
    let s01 = scale(mat_mul(&r0t, &r1));
    let s10 = transpose(&s01);
    let s11 = scale(mat_mul(&r1t, &r1));

    let l = cholesky_lower(&s11)?;
    let m = mat_mul(&s10, &lu_solve_multi(&s00, &s01)?); // S10 S00^{-1} S01
    let l_inv = lu_solve_multi(&l, &identity(n))?;
    let mut a = mat_mul(&mat_mul(&l_inv, &m), &transpose(&l_inv));
    for i in 0..n {
        for j in (i + 1)..n {
            let sym = 0.5 * (a[i][j] + a[j][i]); // kill rounding asymmetry
            a[i][j] = sym;
            a[j][i] = sym;
        }
    }
    let (lam_raw, v) = jacobi_eigh(&a)?;
    let lam: Vec<f64> = lam_raw
        .iter()
        .map(|&x| x.clamp(0.0, 1.0 - 1.0e-12))
        .collect();
    let mut beta = mat_mul(&transpose(&l_inv), &v); // beta_i = L^{-T} v_i
    for i in 0..n {
        if beta[0][i].abs() > 1.0e-12 {
            let scale = beta[0][i];
            for row in beta.iter_mut() {
                row[i] /= scale;
            }
        }
    }

    let trace_stats: Vec<f64> = (0..n)
        .map(|r| -(t_eff as f64) * lam[r..].iter().map(|&x| (1.0 - x).ln()).sum::<f64>())
        .collect();
    let crit_95: Vec<f64> = (0..n)
        .map(|r| trace_crit_95(n - r))
        .collect::<Result<_>>()?;
    let mut rank = n;
    for r in 0..n {
        if trace_stats[r] < crit_95[r] {
            rank = r;
            break;
        }
    }
    Ok(JohansenResult {
        eigenvalues: lam,
        trace_stats,
        crit_95,
        rank,
        beta,
        t_eff,
    })
}
