#include "statarb/adf.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "statarb/ols.hpp"

namespace statarb {

const AdfCriticalRow& adf_critical_values(int nvars) {
    // MacKinnon-style asymptotic values, constant no-trend case.  PINNED.
    static const AdfCriticalRow kTable[3] = {
        {-3.43, -2.86, -2.57},  // nvars = 1: plain ADF on one series
        {-3.90, -3.34, -3.04},  // nvars = 2: EG residual test, 2 vars
        {-4.29, -3.74, -3.45},  // nvars = 3: EG residual test, 3 vars
    };
    if (nvars < 1 || nvars > 3) {
        throw std::invalid_argument("nvars must be 1, 2 or 3");
    }
    return kTable[nvars - 1];
}

double interpolate_pvalue(double tstat, int nvars) {
    const AdfCriticalRow& cv = adf_critical_values(nvars);
    double p;
    if (tstat <= cv.cv5) {
        // segment through (cv1, 0.01) and (cv5, 0.05)
        const double slope = (0.05 - 0.01) / (cv.cv5 - cv.cv1);
        p = 0.01 + slope * (tstat - cv.cv1);
    } else {
        // segment through (cv5, 0.05) and (cv10, 0.10)
        const double slope = (0.10 - 0.05) / (cv.cv10 - cv.cv5);
        p = 0.05 + slope * (tstat - cv.cv5);
    }
    return std::min(std::max(p, kPValueMin), kPValueMax);
}

bool ADFResult::reject(double alpha) const {
    // Direct table comparison; alpha restricted to the tabulated levels.
    if (alpha == 0.01) return tstat < crit.cv1;
    if (alpha == 0.05) return tstat < crit.cv5;
    if (alpha == 0.10) return tstat < crit.cv10;
    throw std::invalid_argument("alpha must be one of 0.01, 0.05, 0.10");
}

ADFResult adf_test(const std::vector<double>& y, int maxlag, int nvars) {
    if (maxlag < 0) {
        throw std::invalid_argument("maxlag must be an integer >= 0");
    }
    const AdfCriticalRow& crit = adf_critical_values(nvars);  // validates nvars
    if (!all_finite(y)) {
        throw std::invalid_argument("inputs must be finite (no NaN/inf)");
    }
    const std::size_t t_len = y.size();
    const int p = maxlag;
    const int nparams = 2 + p;  // c, rho, phi_1..phi_p
    const long eff = static_cast<long>(t_len) - 1 - p;
    if (t_len == 0 ||
        *std::max_element(y.begin(), y.end()) ==
            *std::min_element(y.begin(), y.end())) {
        throw std::invalid_argument("series is constant: ADF test undefined");
    }
    if (eff < nparams + 10) {
        throw std::invalid_argument("series too short for ADF with maxlag=" +
                                    std::to_string(p) + ": effective nobs " +
                                    std::to_string(eff));
    }

    // dy_t = y_t - y_{t-1}; regression rows t = p+1 .. T-1 (0-indexed).
    const std::size_t rows = static_cast<std::size_t>(eff);
    std::vector<double> dy(t_len - 1);
    for (std::size_t t = 1; t < t_len; ++t) dy[t - 1] = y[t] - y[t - 1];

    std::vector<double> lhs(rows);
    for (std::size_t i = 0; i < rows; ++i) lhs[i] = dy[static_cast<std::size_t>(p) + i];

    Matrix x(rows, static_cast<std::size_t>(1 + p));
    for (std::size_t i = 0; i < rows; ++i) {
        x(i, 0) = y[static_cast<std::size_t>(p) + i];  // y_{t-1}
        for (int lag = 1; lag <= p; ++lag) {
            x(i, static_cast<std::size_t>(lag)) =
                dy[static_cast<std::size_t>(p - lag) + i];  // dy_{t-lag}
        }
    }
    const OLSResult fit = ols(lhs, x, /*intercept=*/true);

    ADFResult res;
    res.tstat = fit.tstats[1];  // rho is first regressor after the intercept
    res.pvalue = interpolate_pvalue(res.tstat, nvars);
    res.crit = crit;
    res.maxlag = p;
    res.nobs = rows;
    res.nvars = nvars;
    return res;
}

}  // namespace statarb
