/// \file ols.hpp
/// \brief Ordinary least squares — pinned solution method.
///
/// The solution method is PINNED for cross-language reproducibility:
///
///     b = solve(X'X, X'y)          (normal equations, LU solve)
///     cov(b) = sigma2 * inv(X'X),  sigma2 = RSS / (n - k)
///
/// No QR, no SVD.  The design matrices in this project are tiny (k <= 4)
/// and well-conditioned, so the normal equations lose no meaningful
/// precision.  A conditioning guard (cond(X'X) > 1e12) rejects
/// (near-)collinear designs with std::invalid_argument rather than
/// returning garbage.
#pragma once

#include <cstddef>
#include <vector>

#include "statarb/linalg.hpp"

namespace statarb {

/// Condition-number threshold on X'X above which the design is treated
/// as collinear.  Pinned.
inline constexpr double kCondLimit = 1.0e12;

/// Result of an OLS fit.
struct OLSResult {
    std::vector<double> params;  ///< Coefficients, intercept first when requested.
    std::vector<double> se;      ///< Plain (non-robust) standard errors.
    std::vector<double> tstats;  ///< params / se elementwise.
    std::vector<double> resid;   ///< Residuals y - X b (length nobs).
    double r2 = 0.0;             ///< Centered R^2 with intercept, uncentered without.
    double sigma2 = 0.0;         ///< RSS / (nobs - nparams).
    std::size_t nobs = 0;        ///< Number of observations.
    std::size_t nparams = 0;     ///< Number of coefficients (incl. intercept).
};

/// Fit y = X b + e by ordinary least squares (normal equations).
///
/// \param y Response vector, length n.
/// \param x Regressor matrix, n rows.  The intercept column is prepended
///          automatically when `intercept` is true; do not include one.
/// \param intercept Include a constant term (default true).
/// \throws std::invalid_argument on length mismatch, non-finite input,
///         too few observations (n <= k), constant response with
///         intercept (undefined R^2), or (near-)collinear design
///         (cond(X'X) > 1e12 — this also catches a constant regressor
///         when an intercept is present).
OLSResult ols(const std::vector<double>& y, const Matrix& x, bool intercept = true);

/// Single-regressor convenience overload.
OLSResult ols(const std::vector<double>& y, const std::vector<double>& x,
              bool intercept = true);

}  // namespace statarb
