/// \file adf.hpp
/// \brief Augmented Dickey-Fuller test with embedded critical-value tables.
///
/// Regression form (PINNED, constant / no-trend case):
///
///     dy_t = c + rho * y_{t-1} + sum_{i=1..p} phi_i * dy_{t-i} + e_t
///
/// with a FIXED augmentation lag p = maxlag (default 1; no automatic lag
/// selection).  The statistic is the plain OLS t-stat on rho.  Critical
/// values are MacKinnon-style ASYMPTOTIC values (constant no-trend case)
/// embedded verbatim for the 1-variable ADF test and the Engle-Granger
/// residual test with 2 and 3 variables.  The p-value is a documented
/// approximation: piecewise-linear interpolation of the significance
/// level through the three (critical value, level) knots with edge-slope
/// extrapolation, clamped to [1e-4, 0.9999].  The exact MacKinnon
/// response-surface regression is deliberately NOT implemented.
#pragma once

#include <cstddef>
#include <vector>

namespace statarb {

/// One row of the embedded critical-value table (1%, 5%, 10%).
struct AdfCriticalRow {
    double cv1;   ///< 1% critical value.
    double cv5;   ///< 5% critical value.
    double cv10;  ///< 10% critical value.
};

/// MacKinnon-style asymptotic critical values, constant no-trend case,
/// keyed by number of variables (1 = plain ADF; 2, 3 = Engle-Granger
/// residual test).  PINNED verbatim.
/// \throws std::invalid_argument unless nvars is 1, 2 or 3.
const AdfCriticalRow& adf_critical_values(int nvars);

/// Clamp bounds for the interpolated p-value.  Pinned.
inline constexpr double kPValueMin = 1.0e-4;
inline constexpr double kPValueMax = 0.9999;

/// Approximate p-value by piecewise-linear interpolation through the
/// knots (cv_1%, 0.01), (cv_5%, 0.05), (cv_10%, 0.10); edge segments
/// extrapolate linearly; result clamped to [kPValueMin, kPValueMax].
double interpolate_pvalue(double tstat, int nvars);

/// Result of an ADF unit-root test.
struct ADFResult {
    double tstat = 0.0;   ///< OLS t-statistic on rho.
    double pvalue = 0.0;  ///< Interpolated approximate p-value.
    AdfCriticalRow crit;  ///< Critical values used (1%, 5%, 10%).
    int maxlag = 0;       ///< Augmentation lag p used (fixed, not selected).
    std::size_t nobs = 0; ///< Effective regression observations (T - 1 - p).
    int nvars = 1;        ///< Table selector (1 = ADF, 2/3 = EG residual case).

    /// Reject the unit-root null at `alpha` in {0.01, 0.05, 0.10}.
    /// Verdict by direct table comparison (tstat < critical value), NOT
    /// via the interpolated p-value.
    /// \throws std::invalid_argument for any other alpha.
    bool reject(double alpha = 0.05) const;
};

/// Run the ADF regression and return the unit-root test result.
///
/// \param y Series to test (levels for nvars=1; cointegrating residuals
///          for nvars=2 or 3), finite.
/// \param maxlag Fixed augmentation lag p >= 0 (default 1).
/// \param nvars Critical-value table selector: 1 for a plain ADF test,
///              2 or 3 for the Engle-Granger residual test.
/// \throws std::invalid_argument on negative maxlag, unknown nvars,
///         constant series, non-finite input, or too few observations
///         (effective nobs must be >= nparams + 10).
ADFResult adf_test(const std::vector<double>& y, int maxlag = 1, int nvars = 1);

}  // namespace statarb
