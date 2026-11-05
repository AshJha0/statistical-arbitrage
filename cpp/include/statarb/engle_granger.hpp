/// \file engle_granger.hpp
/// \brief Engle-Granger two-step cointegration test.
///
/// Step 1: static OLS hedge regression  y_t = a + b * x_t + e_t  (normal
/// equations, see ols.hpp).
/// Step 2: ADF test (constant, fixed maxlag) on the residuals e_t, judged
/// against the Engle-Granger 2-variable critical-value table (adf.hpp
/// with nvars=2) — the residuals are estimated, so the plain
/// Dickey-Fuller table would over-reject.
#pragma once

#include <vector>

#include "statarb/adf.hpp"

namespace statarb {

/// Engle-Granger two-step result.
struct EGResult {
    double alpha = 0.0;         ///< Intercept of the hedge regression.
    double beta = 0.0;          ///< Hedge ratio b.
    std::vector<double> resid;  ///< Cointegrating residuals y - a - b x.
    double r2 = 0.0;            ///< R^2 of the hedge regression.
    ADFResult adf;              ///< ADF result on the residuals (nvars=2 table).

    /// True when the residual ADF rejects a unit root at `alpha`.
    bool cointegrated(double alpha_level = 0.05) const { return adf.reject(alpha_level); }
};

/// Run the Engle-Granger two-step test of cointegration of y on x.
///
/// \param y Dependent price series.
/// \param x Regressor price series, same length.
/// \param maxlag Fixed ADF augmentation lag (default 1).
/// \throws std::invalid_argument propagated from OLS/ADF validation
///         (length mismatch, constant series, non-finite values, short
///         series).
EGResult engle_granger(const std::vector<double>& y,
                       const std::vector<double>& x, int maxlag = 1);

/// Engle-Granger run in both directions (the test is not symmetric).
struct EGBothResult {
    EGResult y_on_x;
    EGResult x_on_y;
    /// True only when BOTH directions reject at `alpha_level`.
    bool cointegrated(double alpha_level = 0.05) const {
        return y_on_x.cointegrated(alpha_level) && x_on_y.cointegrated(alpha_level);
    }
};

/// Run engle_granger as y on x and as x on y.
EGBothResult engle_granger_both(const std::vector<double>& y,
                                const std::vector<double>& x, int maxlag = 1);

}  // namespace statarb

