//! Minimal dense linear algebra for the tiny matrices of this project.
//!
//! Every matrix here is at most 4x4 (design cross-products, Johansen
//! product moments), so the routines favour clarity and exact mirroring
//! of the pinned reference algorithms over speed:
//!
//! * LU solve with partial pivoting (the pinned "normal equations with an
//!   LU solve" OLS backend, mirroring `numpy.linalg.solve`),
//! * column-algorithm Cholesky (pinned for Johansen's `S11 = L L'`),
//! * cyclic Jacobi rotations for the symmetric eigenproblem (pinned — no
//!   library eigensolver),
//! * a 2-norm condition number for symmetric matrices via the Jacobi
//!   eigenvalues (`cond = max|lambda| / min|lambda|`), used for the
//!   `cond(X'X) <= 1e12` collinearity guard.
#![allow(clippy::needless_range_loop)] // index form mirrors the pinned matrix recursions

use crate::error::{Result, StatarbError};

/// Row-major dense matrix as a vector of equal-length rows.
pub type Matrix = Vec<Vec<f64>>;

/// Jacobi convergence threshold on the off-diagonal Frobenius norm. Pinned.
pub const JACOBI_TOL: f64 = 1.0e-14;
/// Maximum number of Jacobi sweeps. Pinned.
pub const JACOBI_MAX_SWEEPS: usize = 100;

/// n x n identity matrix.
pub(crate) fn identity(n: usize) -> Matrix {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

/// Matrix transpose.
pub(crate) fn transpose(a: &[Vec<f64>]) -> Matrix {
    let (r, c) = (a.len(), a[0].len());
    (0..c).map(|j| (0..r).map(|i| a[i][j]).collect()).collect()
}

/// Plain matrix product `A B` (sequential inner-product accumulation).
pub(crate) fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Matrix {
    let (n, k, m) = (a.len(), b.len(), b[0].len());
    let mut out = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0;
            for (l, b_row) in b.iter().enumerate().take(k) {
                s += a[i][l] * b_row[j];
            }
            out[i][j] = s;
        }
    }
    out
}

/// Solve `A X = B` for square `A` by LU with partial pivoting.
///
/// `b` is an n x m right-hand-side matrix; the result has the same shape.
/// An exactly zero pivot yields [`StatarbError::Singular`] (the OLS/ADF
/// callers additionally guard with a condition-number check first).
pub(crate) fn lu_solve_multi(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Matrix> {
    let n = a.len();
    let m = b[0].len();
    let mut lu: Matrix = a.to_vec();
    let mut rhs: Matrix = b.to_vec();
    for col in 0..n {
        // partial pivoting: largest |entry| in the column at or below the diagonal
        let mut piv = col;
        for r in col + 1..n {
            if lu[r][col].abs() > lu[piv][col].abs() {
                piv = r;
            }
        }
        if lu[piv][col] == 0.0 {
            return Err(StatarbError::Singular(
                "singular matrix in linear solve".to_string(),
            ));
        }
        lu.swap(col, piv);
        rhs.swap(col, piv);
        for r in col + 1..n {
            let f = lu[r][col] / lu[col][col];
            if f != 0.0 {
                for c in col..n {
                    lu[r][c] -= f * lu[col][c];
                }
                for c in 0..m {
                    rhs[r][c] -= f * rhs[col][c];
                }
            }
        }
    }
    let mut x = vec![vec![0.0; m]; n];
    for r in (0..n).rev() {
        for c in 0..m {
            let mut s = rhs[r][c];
            for j in r + 1..n {
                s -= lu[r][j] * x[j][c];
            }
            x[r][c] = s / lu[r][r];
        }
    }
    Ok(x)
}

/// Solve `A x = b` for a single right-hand side.
pub(crate) fn lu_solve(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let cols: Matrix = b.iter().map(|&v| vec![v]).collect();
    let x = lu_solve_multi(a, &cols)?;
    Ok(x.into_iter().map(|row| row[0]).collect())
}

/// Matrix inverse via LU solves against the identity.
pub(crate) fn inv(a: &[Vec<f64>]) -> Result<Matrix> {
    lu_solve_multi(a, &identity(a.len()))
}

/// Native lower-triangular Cholesky factor of a small SPD matrix.
///
/// Standard column-by-column algorithm (pinned); a pivot `<= 0` means the
/// matrix is not positive definite and yields [`StatarbError::Singular`];
/// a non-square or non-finite input is an error too.
pub fn cholesky_lower(a: &[Vec<f64>]) -> Result<Matrix> {
    let n = a.len();
    if a.iter().any(|row| row.len() != n) {
        return Err(StatarbError::InvalidParam(
            "cholesky_lower requires a square matrix".to_string(),
        ));
    }
    if !a.iter().all(|row| row.iter().all(|v| v.is_finite())) {
        return Err(StatarbError::NonFinite(
            "inputs must be finite (no NaN/inf)".to_string(),
        ));
    }
    let mut l = vec![vec![0.0; n]; n];
    for j in 0..n {
        let mut d = a[j][j];
        for k in 0..j {
            d -= l[j][k] * l[j][k];
        }
        if d <= 0.0 {
            return Err(StatarbError::Singular(
                "matrix is not positive definite (Cholesky pivot <= 0)".to_string(),
            ));
        }
        l[j][j] = d.sqrt();
        for i in j + 1..n {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            l[i][j] = s / l[j][j];
        }
    }
    Ok(l)
}

/// Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations.
///
/// Pinned algorithm: sweep the upper triangle in row-major order; for each
/// off-diagonal element with `|a_pq| > 1e-14` apply the classical rotation
/// with `theta = (a_qq - a_pp) / (2 a_pq)`,
/// `t = sign(theta) / (|theta| + sqrt(theta^2 + 1))` (`t = 1` when
/// `theta == 0`), `c = 1/sqrt(t^2+1)`, `s = t c`; stop when the
/// off-diagonal Frobenius norm falls below the tolerance or after 100
/// sweeps (non-convergence is not an error — the best iterate is
/// returned; the 2x2/3x3 matrices here always converge).
///
/// Returns `(eigenvalues descending, eigenvector matrix with matching
/// columns)`.
///
/// # Errors
///
/// [`StatarbError::InvalidParam`] unless `a` is square and symmetric;
/// [`StatarbError::NonFinite`] on NaN/inf entries.
pub fn jacobi_eigh(a: &[Vec<f64>]) -> Result<(Vec<f64>, Matrix)> {
    let n = a.len();
    for row in a {
        if row.len() != n {
            return Err(StatarbError::InvalidParam(
                "jacobi_eigh requires a symmetric square matrix".to_string(),
            ));
        }
    }
    if !a.iter().all(|row| row.iter().all(|v| v.is_finite())) {
        return Err(StatarbError::NonFinite(
            "inputs must be finite (no NaN/inf)".to_string(),
        ));
    }
    for i in 0..n {
        for j in 0..n {
            // numpy.allclose default rtol=1e-5 with atol=1e-10
            if (a[i][j] - a[j][i]).abs() > 1.0e-10 + 1.0e-5 * a[j][i].abs() {
                return Err(StatarbError::InvalidParam(
                    "jacobi_eigh requires a symmetric square matrix".to_string(),
                ));
            }
        }
    }
    // Flat row-major working copies, allocated once; each rotation is applied
    // in place. Only the entries the dense products `J' M J` and `V J` would
    // change are touched, with the same operands in the same order (the
    // skipped terms are exact zeros), so results are bit-identical.
    let mut m: Vec<f64> = a.iter().flat_map(|r| r.iter().copied()).collect();
    let mut vf: Vec<f64> = vec![0.0; n * n];
    for i in 0..n {
        vf[i * n + i] = 1.0;
    }
    for _ in 0..JACOBI_MAX_SWEEPS {
        let mut off2 = 0.0;
        for i in 1..n {
            for j in 0..i {
                off2 += m[i * n + j] * m[i * n + j];
            }
        }
        if (off2 * 2.0).sqrt() < JACOBI_TOL {
            break;
        }
        for p in 0..n.saturating_sub(1) {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq.abs() <= JACOBI_TOL {
                    continue;
                }
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // rows p, q of J' M
                for j in 0..n {
                    let (mp, mq) = (m[p * n + j], m[q * n + j]);
                    m[p * n + j] = 0.0 + c * mp + (-s) * mq;
                    m[q * n + j] = 0.0 + s * mp + c * mq;
                }
                // columns p, q of (J' M) J and of V J
                for i in 0..n {
                    let (ap, aq) = (m[i * n + p], m[i * n + q]);
                    m[i * n + p] = 0.0 + c * ap + (-s) * aq;
                    m[i * n + q] = 0.0 + s * ap + c * aq;
                    let (vp, vq) = (vf[i * n + p], vf[i * n + q]);
                    vf[i * n + p] = 0.0 + c * vp + (-s) * vq;
                    vf[i * n + q] = 0.0 + s * vp + c * vq;
                }
            }
        }
    }
    let mat = |i: usize, j: usize| m[i * n + j];
    let v = |i: usize, j: usize| vf[i * n + j];
    // sort descending (stable ascending argsort, reversed — as the reference);
    // total_cmp cannot panic (the input is finite, so no NaN can appear)
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| mat(i, i).total_cmp(&mat(j, j)));
    order.reverse();
    let vals: Vec<f64> = order.iter().map(|&i| mat(i, i)).collect();
    let vecs: Matrix = (0..n)
        .map(|r| order.iter().map(|&c| v(r, c)).collect())
        .collect();
    Ok((vals, vecs))
}

/// 2-norm condition number of a symmetric matrix via Jacobi eigenvalues.
///
/// For a symmetric matrix the singular values are `|lambda_i|`, so
/// `cond_2 = max|lambda| / min|lambda|` (infinite when the smallest
/// magnitude vanishes) — equivalent to `numpy.linalg.cond`'s default.
pub(crate) fn cond_sym(a: &[Vec<f64>]) -> Result<f64> {
    let (vals, _) = jacobi_eigh(a)?;
    let mags: Vec<f64> = vals.iter().map(|v| v.abs()).collect();
    let hi = mags.iter().cloned().fold(0.0, f64::max);
    let lo = mags.iter().cloned().fold(f64::INFINITY, f64::min);
    if lo == 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok(hi / lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_solve_2x2_exact() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let x = lu_solve(&a, &[5.0, 10.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-14);
        assert!((x[1] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn lu_singular_is_error() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(matches!(
            lu_solve(&a, &[1.0, 2.0]),
            Err(StatarbError::Singular(_))
        ));
    }

    #[test]
    fn cholesky_known_factor() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 5.0]];
        let l = cholesky_lower(&a).unwrap();
        assert!((l[0][0] - 2.0).abs() < 1e-14);
        assert!((l[1][0] - 1.0).abs() < 1e-14);
        assert!((l[1][1] - 2.0).abs() < 1e-14);
        assert_eq!(l[0][1], 0.0);
    }

    #[test]
    fn cholesky_rejects_non_pd() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 1.0]]; // eigenvalues 3, -1
        assert!(matches!(cholesky_lower(&a), Err(StatarbError::Singular(_))));
    }

    #[test]
    fn jacobi_2x2_known_eigenvalues() {
        // eigenvalues of [[2,1],[1,2]] are 3 and 1
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (vals, vecs) = jacobi_eigh(&a).unwrap();
        assert!((vals[0] - 3.0).abs() < 1e-12);
        assert!((vals[1] - 1.0).abs() < 1e-12);
        // eigenvector columns satisfy A v = lambda v
        for k in 0..2 {
            for i in 0..2 {
                let av: f64 = (0..2).map(|j| a[i][j] * vecs[j][k]).sum();
                assert!((av - vals[k] * vecs[i][k]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn jacobi_rejects_asymmetric() {
        let a = vec![vec![1.0, 2.0], vec![0.0, 1.0]];
        assert!(jacobi_eigh(&a).is_err());
    }

    #[test]
    fn cond_of_identity_is_one() {
        assert!((cond_sym(&identity(3)).unwrap() - 1.0).abs() < 1e-12);
    }
}
