/// \file johansen.hpp
/// \brief Johansen-lite trace test for 2- and 3-asset systems.
///
/// Canonical correlation setup (PINNED).  VECM with unrestricted constant
/// and k = 1 lagged difference:
///
///     dY_t = c + Gamma * dY_{t-1} + Pi * Y_{t-1} + e_t
///
/// Steps (T raw observations, n = 2 or 3 assets, T_eff = T - 2):
///  1. Z0 = dY_t, Z1 = Y_{t-1}, ZD = [1, dY_{t-1}] for t = 2..T-1.
///  2. R0/R1 = residuals of Z0/Z1 on ZD (multivariate OLS via normal
///     equations, one solve per block; error if cond(ZD'ZD) > 1e12).
///  3. S_ij = R_i' R_j / T_eff.
///  4. |lambda S11 - S10 inv(S00) S01| = 0 made SYMMETRIC via the native
///     Cholesky S11 = L L': A = L^{-1} S10 S00^{-1} S01 L^{-T}, forced
///     symmetric, then eigen-decomposed by the native cyclic Jacobi
///     algorithm (linalg.hpp).  Eigenvalues descending, clipped into
///     [0, 1 - 1e-12]; cointegrating vectors beta_i = L^{-T} v_i,
///     normalised so the first component is +1 (when |first| > 1e-12).
///  5. trace(r) = -T_eff * sum_{i=r..n-1} ln(1 - lambda_i), judged
///     against the embedded Osterwald-Lenum (1992) 95% trace table.
///
/// Honest limitations: fixed lag (k=1), fixed deterministic case
/// (unrestricted constant), only the 95% column; no small-sample
/// correction, no maximum-eigenvalue variant, no p-values.  Meant for
/// 2-3 asset pairs-trading diagnostics, not general VECM work.
#pragma once

#include <cstddef>
#include <vector>

#include "statarb/linalg.hpp"

namespace statarb {

/// Osterwald-Lenum (1992) 95% critical value for the TRACE statistic,
/// model with unrestricted constant, keyed by (n - r): 1 -> 3.76,
/// 2 -> 15.41, 3 -> 29.68.  PINNED verbatim.
/// \throws std::invalid_argument unless n_minus_r is 1, 2 or 3.
double trace_crit_95(int n_minus_r);

/// Johansen-lite trace-test result.
struct JohansenResult {
    std::vector<double> eigenvalues;  ///< lambda_i, descending, in [0, 1).
    std::vector<double> trace_stats;  ///< trace(r) for r = 0..n-1.
    std::vector<double> crit_95;      ///< Matching 95% critical values.
    int rank = 0;                     ///< Estimated cointegration rank at 95%.
    Matrix beta;                      ///< Column i: cointegrating vector for
                                      ///< lambda_i, first component +1.
    std::size_t t_eff = 0;            ///< Effective sample size T - 2.
};

/// Run the Johansen-lite trace test on a (T, n) matrix of levels.
///
/// \param levels Price levels, T rows and n in {2, 3} columns, T >= 20.
/// \param maxlag_diff Number of lagged differences k; only k=1 is
///        implemented (pinned) — any other value throws.
/// \throws std::invalid_argument on n outside {2,3}, T < 20, non-finite
///         values, a constant column, or maxlag_diff != 1.
JohansenResult johansen_trace(const Matrix& levels, int maxlag_diff = 1);

}  // namespace statarb
