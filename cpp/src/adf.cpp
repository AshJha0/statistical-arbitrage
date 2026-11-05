#include "statarb/adf.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "statarb/ols.hpp"

namespace statarb {

const AdfSurfaceRow* adf_response_surface(int nvars) {
    // MacKinnon (2010), "Critical Values for Cointegration Tests", Queen's
    // Economics Dept. WP 1227, Table 1, constant / no-trend case.
    // Rows: 1%, 5%, 10%; columns: b_inf, b1, b2, b3.  PINNED verbatim.
    static const AdfSurfaceRow kTable[3][3] = {
        {   // nvars = 1: plain ADF on one series
            {-3.43035, -6.5393, -16.786, -79.433},
            {-2.86154, -2.8903, -4.234, -40.040},
            {-2.56677, -1.5384, -2.809, 0.0},
        },
        {   // nvars = 2: EG residual test, 2 vars
            {-3.89644, -10.9519, -33.527, 0.0},
            {-3.33613, -6.1101, -6.823, 0.0},
            {-3.04445, -4.2412, -2.720, 0.0},
        },
        {   // nvars = 3: EG residual test, 3 vars
            {-4.29374, -14.4354, -33.195, 47.433},
            {-3.74066, -8.5632, -10.852, 27.982},
            {-3.45218, -6.2143, -3.718, 0.0},
        },
    };
    if (nvars < 1 || nvars > 3) {
        throw std::invalid_argument("nvars must be 1, 2 or 3");
    }
    return kTable[nvars - 1];
}

AdfCriticalRow adf_critical_values(int nvars) {
    const AdfSurfaceRow* rows = adf_response_surface(nvars);
    return AdfCriticalRow{rows[0].b_inf, rows[1].b_inf, rows[2].b_inf};
}

AdfCriticalRow adf_critical_values(int nvars, std::size_t nobs) {
    const AdfSurfaceRow* rows = adf_response_surface(nvars);
    if (nobs < kAdfMinNobs) {
        throw std::invalid_argument("critical values need at least " +
                                    std::to_string(kAdfMinNobs) +
                                    " effective observations");
    }
    const double inv = 1.0 / static_cast<double>(nobs);
    double cv[3];
    for (int i = 0; i < 3; ++i) {
        // Horner form, identical to the Python reference.
        cv[i] = rows[i].b_inf + inv * (rows[i].b1 + inv * (rows[i].b2 + inv * rows[i].b3));
    }
    return AdfCriticalRow{cv[0], cv[1], cv[2]};
}

namespace {

double interpolate_through(double tstat, const AdfCriticalRow& cv) {
    if (std::isnan(tstat)) {
        throw std::invalid_argument("tstat must not be NaN");
    }
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

}  // namespace

double interpolate_pvalue(double tstat, int nvars) {
    return interpolate_through(tstat, adf_critical_values(nvars));
}

double interpolate_pvalue(double tstat, int nvars, std::size_t nobs) {
    return interpolate_through(tstat, adf_critical_values(nvars, nobs));
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
    if (maxlag > kAdfMaxLag) {
        throw std::invalid_argument("maxlag must be <= " + std::to_string(kAdfMaxLag));
    }
    adf_response_surface(nvars);  // validates nvars
    if (!all_finite(y)) {
        throw std::invalid_argument("inputs must be finite (no NaN/inf)");
    }
    const std::size_t t_len = y.size();
    const int p = maxlag;
    // 64-bit arithmetic throughout: no overflow for any admissible maxlag.
    const long long nparams = 2LL + p;  // c, rho, phi_1..phi_p
    const long long eff = static_cast<long long>(t_len) - 1 - p;
    if (t_len > 0 &&
        *std::max_element(y.begin(), y.end()) ==
            *std::min_element(y.begin(), y.end())) {
        throw std::invalid_argument("series is constant: ADF test undefined");
    }
    const long long min_eff =
        std::max(nparams + 10, static_cast<long long>(kAdfMinNobs));
    if (eff < min_eff) {
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
    res.crit = adf_critical_values(nvars, rows);
    res.pvalue = interpolate_pvalue(res.tstat, nvars, rows);
    res.maxlag = p;
    res.nobs = rows;
    res.nvars = nvars;
    return res;
}

}  // namespace statarb

namespace statarb {

int select_lag(const std::vector<double>& y, int max_lag, int nvars) {
    adf_test(y, max_lag, nvars);  // shared validation
    const std::size_t t_len = y.size();
    const std::size_t m = static_cast<std::size_t>(max_lag);
    std::vector<double> dy(t_len - 1);
    for (std::size_t t = 1; t < t_len; ++t) dy[t - 1] = y[t] - y[t - 1];
    const std::size_t n = dy.size() - m;
    std::vector<double> lhs(n);
    for (std::size_t i = 0; i < n; ++i) lhs[i] = dy[m + i];
    const double ln_n = std::log(static_cast<double>(n));

    int best_lag = 0;
    double best_bic = std::numeric_limits<double>::infinity();
    for (std::size_t p = 0; p <= m; ++p) {
        Matrix x(n, 1 + p);
        for (std::size_t i = 0; i < n; ++i) {
            x(i, 0) = y[m + i];  // y_{t-1}
            for (std::size_t lag = 1; lag <= p; ++lag) x(i, lag) = dy[m - lag + i];
        }
        const OLSResult fit = ols(lhs, x, /*intercept=*/true);
        double rss = 0.0;
        for (double e : fit.resid) rss += e * e;
        const double fit_term = rss > 0.0 ? static_cast<double>(n) * std::log(rss / static_cast<double>(n))
                                          : -std::numeric_limits<double>::infinity();
        const double bic = fit_term + static_cast<double>(2 + p) * ln_n;
        if (bic < best_bic || p == 0) {
            best_lag = static_cast<int>(p);
            best_bic = bic;
        }
    }
    return best_lag;
}

}  // namespace statarb
