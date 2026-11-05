/// \file adf.hpp
/// \brief Augmented Dickey-Fuller test with MacKinnon (2010) critical values.
///
/// Regression form (PINNED, constant / no-trend case):
///
///     dy_t = c + rho * y_{t-1} + sum_{i=1..p} phi_i * dy_{t-i} + e_t
///
/// with a FIXED augmentation lag p = maxlag (default 1; no automatic lag
/// selection).  The statistic is the plain OLS t-stat on rho.  Critical
/// values are the MacKinnon (2010) finite-sample response surfaces for the
/// constant / no-trend case,
///
///     cv(T_eff) = b_inf + b_1/T_eff + b_2/T_eff^2 + b_3/T_eff^3,
///
/// embedded verbatim for the 1-variable ADF test and the Engle-Granger
/// residual test with 2 and 3 variables, where T_eff = T - 1 - p is the
/// number of observations in the test regression (as in statsmodels'
/// adfuller).  The surfaces were fitted on samples of 20 or more, so
/// T_eff >= 20 is enforced.  The p-value is a documented approximation:
/// piecewise-linear interpolation of the significance level through the
/// three (critical value, level) knots at T_eff, with edge-slope
/// extrapolation, clamped to [1e-4, 0.9999].  The exact MacKinnon (1994)
/// p-value response surface is deliberately NOT implemented; the verdict
/// never uses the p-value.
#pragma once

#include <cstddef>
#include <vector>

namespace statarb {

/// One row of critical values (1%, 5%, 10%).
struct AdfCriticalRow {
    double cv1;   ///< 1% critical value.
    double cv5;   ///< 5% critical value.
    double cv10;  ///< 10% critical value.
};

/// MacKinnon (2010) response-surface coefficients for one level:
/// cv(T) = b_inf + b1/T + b2/T^2 + b3/T^3.
struct AdfSurfaceRow {
    double b_inf;
    double b1;
    double b2;
    double b3;
};

/// The pinned response-surface coefficients (constant / no-trend case)
/// for `nvars` in {1, 2, 3}; index 0/1/2 = 1%/5%/10%.
/// \throws std::invalid_argument unless nvars is 1, 2 or 3.
const AdfSurfaceRow* adf_response_surface(int nvars);

/// Asymptotic (T -> inf) critical values b_inf, keyed by number of
/// variables (1 = plain ADF; 2, 3 = Engle-Granger residual test).
/// \throws std::invalid_argument unless nvars is 1, 2 or 3.
AdfCriticalRow adf_critical_values(int nvars);

/// Finite-sample MacKinnon (2010) critical values at effective sample
/// size `nobs` (the regression observation count).
/// \throws std::invalid_argument unless nvars is 1, 2 or 3 and
///         nobs >= kAdfMinNobs.
AdfCriticalRow adf_critical_values(int nvars, std::size_t nobs);

/// Smallest effective sample the response surfaces are trusted for.  Pinned.
inline constexpr std::size_t kAdfMinNobs = 20;

/// Largest augmentation lag accepted (documented upper bound).  Pinned.
inline constexpr int kAdfMaxLag = 1000;

/// Clamp bounds for the interpolated p-value.  Pinned.
inline constexpr double kPValueMin = 1.0e-4;
inline constexpr double kPValueMax = 0.9999;

/// Approximate p-value by piecewise-linear interpolation through the
/// asymptotic knots (cv_1%, 0.01), (cv_5%, 0.05), (cv_10%, 0.10); edge
/// segments extrapolate linearly; result clamped to [kPValueMin,
/// kPValueMax].  A NaN t-stat is an error; +/-inf clamps.
double interpolate_pvalue(double tstat, int nvars);

/// Same, through the finite-sample knots at effective sample size `nobs`.
double interpolate_pvalue(double tstat, int nvars, std::size_t nobs);

/// Result of an ADF unit-root test.
struct ADFResult {
    double tstat = 0.0;   ///< OLS t-statistic on rho.
    double pvalue = 0.0;  ///< Interpolated approximate p-value.
    AdfCriticalRow crit;  ///< Finite-sample critical values used (1%, 5%, 10%).
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
/// \param maxlag Fixed augmentation lag p in [0, kAdfMaxLag] (default 1).
/// \param nvars Critical-value table selector: 1 for a plain ADF test,
///              2 or 3 for the Engle-Granger residual test.
/// \throws std::invalid_argument on negative/oversized maxlag, unknown
///         nvars, constant series, non-finite input, or too few
///         observations (effective nobs must be
///         >= max(nparams + 10, kAdfMinNobs)).
ADFResult adf_test(const std::vector<double>& y, int maxlag = 1, int nvars = 1);

/// Lag in 0..max_lag with the smallest BIC, for use with adf_test.  Every
/// candidate is fitted on the SAME rows t = max_lag+1 .. T-1, so the BICs
/// compare: BIC(p) = n ln(RSS_p/n) + (2+p) ln n, n = T - 1 - max_lag.  A tie
/// goes to the smaller lag.
/// \throws std::invalid_argument as adf_test(y, max_lag, nvars).
int select_lag(const std::vector<double>& y, int max_lag, int nvars = 1);

}  // namespace statarb
